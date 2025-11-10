MoodTune RAG Service

Servicio/API de Recuperación Aumentada por Generación (RAG) sobre OpenSearch 3.0.0 y un LLM de OpenAI. No se conecta a Spotify directamente: consume `moodtune_music` para catálogo (búsquedas, recomendaciones y audio-features). Basado en Flask con Blueprints.

Características
- Flask + Blueprints (estructura similar a `moodtune_api`).
- OpenSearch 3.0.0 con índice `moodtune_tracks` y mapeo de `valence`/`energy` (+ embeddings opcionales).
- Búsqueda por emoción con relajación automática de rangos (valence/energy) si hay pocos resultados.
- LLM (OpenAI) para generar título/descripcion de playlists y respuestas breves con contexto.
- Fallback: si no hay suficientes resultados, el LLM sugiere títulos/artistas por emoción y se resuelven contra proveedores vía `moodtune_music`.
- Docker Compose con `opensearch` y `opensearch-dashboards` 3.0.0.

Endpoints
- `GET /health` Comprobación de salud.
- `POST /rag/search` Recupera pistas por emoción.
- `POST /rag/playlist` Genera metadatos (LLM) y lista de pistas para una playlist.
- `POST /llm/playlist-meta` Genera título y descripción a partir de una emoción.
- `POST /llm/answer` Responde un prompt usando emoción + contexto recuperado por RAG.
- `POST /admin/seed` Precarga el índice desde `moodtune_music`.
- `POST /admin/rebuild` Recrea el índice y reingesta la KB.

OpenAPI y Postman
- OpenAPI: `moodtune_rag/docs/openapi/moodtune_rag.yaml`
- Postman: `moodtune_rag/docs/openapi/moodtune_rag.postman_collection.json`
- Entornos: `moodtune_rag/docs/openapi/MoodTune_RAG_Local.postman_environment.json`, `moodtune_rag/docs/openapi/MoodTune_rag.postman_environment.json`

Requisitos
- Copia `.env.example` a `.env` y ajusta valores.
- `OPENAI_API_KEY` válido y permisos de uso del modelo configurado en `.env`.
- Servicio `moodtune_music` accesible (por ejemplo `MUSIC_SERVICE_URL=http://localhost:8020`).
- `MIN_TRACKS` controla el mínimo retornado por `/rag/search` (se fuerza a 20 si se configura menos).

Ejecutar en local (venv)
- Requiere Python 3.11+.
- Windows PowerShell:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```
- macOS/Linux:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Ejecución con Docker
1) Copia `.env.example` a `.env` y completa valores.
2) `docker compose up --build`
3) Dashboards: `http://localhost:5601`
4) API: `http://localhost:8010`
5) Persistencia: los datos de OpenSearch se guardan en el host en `.data/opensearch/`. No ejecutes `docker compose down --volumes` si quieres conservar los documentos.

Inicializar índice (una vez)
- `curl -X PUT http://localhost:9200/moodtune_tracks -H "Content-Type: application/json" --data-binary @moodtune_rag/docs/opensearch_index_mapping.json`
- Campos principales: `id, title, artist, uri, preview_url, mood, valence, energy, embedding?`.

Ingesta de conocimiento (KB)
- Script local: `python -m app.src.ingest_llm_resolver`
- Endpoint: `curl -X POST http://localhost:8010/admin/seed -H "Content-Type: application/json" -d '{"per_emotion": 25}'`
- Comodidad dev: con `AUTO_SEED_ON_START=true`, si el índice está vacío se ingesta automáticamente una muestra al iniciar.

Ejemplos
```
curl -X POST http://localhost:8010/rag/search -H "Content-Type: application/json" -d '{"emotion":"happy","min_tracks":20}'
curl -X POST http://localhost:8010/rag/playlist -H "Content-Type: application/json" -d '{"emotion":"sad","min_tracks":25}'
curl -X POST http://localhost:8010/llm/playlist-meta -H "Content-Type: application/json" -d '{"emotion":"happy"}'
curl -X POST http://localhost:8010/llm/answer -H "Content-Type: application/json" -d '{"emotion":"relaxed","prompt":"Sugiere cómo usar estas canciones para estudiar"}'
```

Notas
- La creación real de playlists en la cuenta del usuario corresponde a `moodtune_music`.
- Este servicio no reproduce ni cachea audio protegido.
