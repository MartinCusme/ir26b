# Examen de Recuperacion de Informacion - SciFact

Entrega: `ExB1_MartínCusme_Scifact.ipynb` ejecutado y URL publica https://exb1-martincusme-scifact.vercel.app (tambien registrada y probada en el notebook).

## Ejecutar

Python 3.11 local o 3.12 en Vercel:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python build.py
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

En PowerShell, activar con `.\.venv\Scripts\Activate.ps1`. Abrir el notebook con ese entorno y ejecutar todos sus bloques desde esta carpeta. El notebook contiene el codigo completo de la API, indexacion, evaluacion y pruebas, con resultados guardados. `app.py` contiene la misma implementacion para alojarla como servicio.

## Servicio

`GET /health`, `POST /search` y documentacion interactiva en `/docs`.

```json
{"query":"Vitamin D deficiency is associated with cardiovascular disease.","model":"bm25","top_k":10}
```

`model` acepta `bm25` y `tfidf`. Se devuelven `rank`, `document_id` original como string, `score`, `title` y `snippet`. Un snippet solo recorta la presentacion: nunca se indexan fragmentos como documentos. Consultas sin terminos del vocabulario devuelven una lista vacia. Si hay menos de k documentos con puntuacion positiva, se devuelven los disponibles sin completar con documentos irrelevantes.

## Decisiones

- Cada documento es titulo + abstract completo de `BeIR/scifact`; se mantienen los IDs originales.
- NFKD, minusculas, eliminacion de diacriticos, tokens alfanumericos, lista explicita de stopwords inglesas y stemming Porter2/Snowball. Numeros y negaciones se conservan.
- Indice invertido propio: termino -> documento -> frecuencia. Vocabulario y longitudes se derivan de todo el corpus.
- TF-IDF propio: TF cruda, IDF suavizado `ln((N+1)/(df+1))+1`, coseno con normalizacion L2. Las palabras fuera del vocabulario se ignoran.
- BM25 propio: `ln(1+(N-df+0.5)/(df+0.5))`, k1=1.2 y b=0.75; cada termino distinto de consulta aporta una vez. Parametros fijados antes de evaluar test.
- Desempate determinista por ID documental ascendente. No hay entrenamiento ni modelos preentrenados.
- Precision@10 divide por 10; Recall@10 divide por todos los relevantes conocidos; NDCG@10 usa ganancias `2^relevancia-1` y descuento log2. Se promedian todas las consultas con qrels de test. El notebook identifica cada consulta y compara con ir-measures.

## Publicar y entregar

`vercel deploy --prod --yes` desde esta carpeta. El build descarga SciFact y genera el archivo de corpus comprimido; la API construye su indice al primer uso. No depende del computador local.

Esta carpeta es un repositorio Git independiente. Revisar `git status --short` desde aqui antes de hacer commit. `.data`, entornos, caches, credenciales y configuracion local de Vercel estan excluidos. Los archivos de FiQA y los notebooks de la carpeta padre quedan fuera de este repositorio.

Fuentes: https://huggingface.co/datasets/BeIR/scifact y https://huggingface.co/datasets/BeIR/scifact-qrels (CC BY-SA 4.0). Qrels solo se usan para evaluar, nunca para responder las busquedas.
