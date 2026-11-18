"""Descargar y conservar documentos completos; se ejecuta durante el build."""
import gzip
import json
from pathlib import Path
from datasets import load_dataset


def prepare():
    path = Path(__file__).resolve().parent / ".data" / "corpus.json.gz"
    if not path.exists():
        corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
        rows = [{"_id": str(r["_id"]), "title": r["title"], "text": r["text"]} for r in corpus]
        assert len(rows) == len({row["_id"] for row in rows})
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as file:
            json.dump(rows, file, ensure_ascii=False)
    with gzip.open(path, "rt", encoding="utf-8") as file:
        rows = json.load(file)
    print(f"Corpus SciFact preparado: {len(rows)} documentos originales completos.")
    return rows


if __name__ == "__main__":
    prepare()
