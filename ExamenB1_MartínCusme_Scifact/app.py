"""SciFact: modelos clasicos propios y API del examen.

Esta implementacion tambien se incluye completa en ExB1_MartínCusme_Scifact.ipynb.
"""
import gzip
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from threading import Lock
from pathlib import Path
from typing import Literal

import snowballstemmer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# Lista explicita para ingles. Se conservan negaciones y numeros cientificos.
STOPWORDS = frozenset("a an the and or but if then else when where who whom which what why how of to in on at by for from with as is are was were be been being am do does did doing have has had having can could would should will may might must shall it its this that these those i me my we our you your he him his she her they them their there here such than so too very also into about between during each both other some any all only same".split())
STEMMER = snowballstemmer.stemmer("english")
STEMMER_LOCK = Lock()


def preprocess(text):
    """NFKD, minusculas, tokens alfanumericos, stopwords y Porter2 ingles."""
    text = unicodedata.normalize("NFKD", text or "").lower()
    text = "".join(c for c in text if not unicodedata.combining(c))
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text)
    with STEMMER_LOCK:
        return STEMMER.stemWords([t for t in tokens if t not in STOPWORDS])


class SearchEngine:
    def __init__(self, documents):
        self.documents = {}
        self.postings = defaultdict(dict)
        self.lengths = {}
        for row in documents:
            doc_id = str(row["_id"])
            if doc_id in self.documents:
                raise ValueError(f"Documento duplicado: {doc_id}")
            title, text = row.get("title") or "", row.get("text") or ""
            self.documents[doc_id] = {"title": title, "text": text}
            counts = Counter(preprocess(title + " " + text))
            self.lengths[doc_id] = sum(counts.values())
            for term, tf in counts.items():
                self.postings[term][doc_id] = tf
        self.n = len(self.documents)
        if not self.n:
            raise ValueError("Corpus vacio")
        self.avgdl = sum(self.lengths.values()) / self.n
        self.tfidf_idf = {t: math.log((self.n + 1)/(len(p) + 1)) + 1 for t, p in self.postings.items()}
        self.bm25_idf = {t: math.log1p((self.n-len(p)+0.5)/(len(p)+0.5)) for t, p in self.postings.items()}
        squared_norms = defaultdict(float)
        for term, postings in self.postings.items():
            for doc_id, tf in postings.items():
                squared_norms[doc_id] += (tf*self.tfidf_idf[term])**2
        self.norms = {d: math.sqrt(value) for d, value in squared_norms.items()}

    def rank(self, query, model="bm25", top_k=10):
        if model not in ("bm25", "tfidf"):
            raise ValueError("Modelo invalido")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k debe ser entero positivo")
        terms = Counter(preprocess(query))
        scores = defaultdict(float)
        if model == "tfidf":
            weights = {t: tf*self.tfidf_idf[t] for t, tf in terms.items() if t in self.tfidf_idf}
            qnorm = math.sqrt(sum(w*w for w in weights.values()))
            if not qnorm:
                return []
            for term, weight in weights.items():
                for doc_id, tf in self.postings[term].items():
                    scores[doc_id] += weight*tf*self.tfidf_idf[term]
            scores = {doc_id: score/(qnorm*self.norms[doc_id]) for doc_id, score in scores.items()}
        else:
            # Parametros fijados a priori; no se ajustan con test.
            k1, b = 1.2, 0.75
            for term in sorted(terms):
                for doc_id, tf in self.postings.get(term, {}).items():
                    norm = 1-b + b*self.lengths[doc_id]/self.avgdl
                    scores[doc_id] += self.bm25_idf[term]*tf*(k1+1)/(tf+k1*norm)
        return sorted(((d, s) for d, s in scores.items() if s > 0), key=lambda x: (-x[1], x[0]))[:top_k]

    def search(self, query, model="bm25", top_k=10):
        return [{"rank": rank, "document_id": doc_id, "score": float(score),
                 "title": self.documents[doc_id]["title"],
                 "snippet": " ".join(self.documents[doc_id]["text"].split())[:350]}
                for rank, (doc_id, score) in enumerate(self.rank(query, model, top_k), 1)]


DATA_FILE = Path(__file__).resolve().parent / ".data" / "corpus.json.gz" if "__file__" in globals() else Path.cwd()/".data"/"corpus.json.gz"


@lru_cache(maxsize=1)
def get_engine():
    if not DATA_FILE.exists():
        raise HTTPException(503, "Corpus no preparado. Ejecuta python build.py.")
    with gzip.open(DATA_FILE, "rt", encoding="utf-8") as file:
        return SearchEngine(json.load(file))


app = FastAPI(title="SciFact - Examen RI", version="1.0.0")


class SearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(min_length=1, max_length=10000)
    model: Literal["bm25", "tfidf"] = "bm25"
    top_k: int = Field(default=10, ge=1, strict=True)


@app.get("/health")
def health():
    engine = get_engine()
    return {"status": "ok", "corpus": "SciFact", "documents": engine.n, "models": ["tfidf", "bm25"]}


@app.post("/search")
def search(body: SearchRequest):
    return {"model": body.model, "results": get_engine().search(body.query, body.model, body.top_k)}
