MoodTune RAG Service

Servicio/API para RAG basado en OpenSearch 3.0.0 y LLM (OpenAI), con estructura tipo `moodtune_api` y enfoque OOP. Este servicio no se conecta a Spotify directamente: consume el servicio `moodtune_music` para catÃ¡logo (recomendaciones y audioâ€‘features).

CaracterÃ­sticas
- Flask + Blueprints (estructura similar a `moodtune_api`).
- RAG con OpenSearch 3.0.0 (bÃºsqueda por emociÃ³n con relajaciÃ³n de rangos valence/energy).
- LLM de OpenAI para generar tÃ­tulo/descripciÃ³n y respuestas breves.
- Fallback sin Spotify: si no hay suficientes resultados, el RAG usa OpenAI para sugerir canciones por emociÃ³n y luego las busca por tÃ­tulo+artista en OpenSearch.
- Ingesta de conocimiento a OpenSearch a travÃ©s de `moodtune_music`.
- Docker Compose con `opensearch` y `opensearch-dashboards` 3.0.0.

Endpoints
- `GET /health` Healthcheck.
- `POST /rag/search` Lista de tracks para una emociÃ³n.
- `POST /rag/playlist` Playlist sugerida (tÃ­tulo/descr. + canciones).
- `POST /llm/playlist-meta` Genera tÃ­tulo y descripciÃ³n por emociÃ³n.
- `POST /llm/answer` Responde un prompt usando emociÃ³n + contexto.
- `POST /admin/seed` Precarga la base de conocimiento (vÃ­a `moodtune_music` o sintÃ©tico).

OpenAPI y Postman
- OpenAPI: `docs/openapi/moodtune_rag.yaml`
- Postman: `docs/openapi/moodtune_rag.postman_collection.json`
- Entornos: `docs/openapi/MoodTune_RAG_Local.postman_environment.json`, `docs/openapi/MoodTune_rag.postman_environment.json`

Requisitos previos
- Variables en `.env` (usa `.env.example`).
- `OPENAI_API_KEY` configurada.
- Servicio `moodtune_music` disponible (p. ej. `MUSIC_SERVICE_URL=http://localhost:8020`).

Entorno virtual (venv) local
- Requiere Python 3.11+ instalado.
- Crear y activar entorno (Windows PowerShell):
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```
- macOS/Linux (bash/zsh):
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Ejecución local (Docker)
1. Copia `.env.example` a `.env` y completa valores.
2. `docker compose up --build`
3. Dashboards: `http://localhost:5601`.
4. API RAG: `http://localhost:8010`.

Inicializar Ã­ndice en OpenSearch (una vez)
- `curl -X PUT http://localhost:9200/moodtune_tracks -H "Content-Type: application/json" --data-binary @docs/opensearch_index_mapping.json`
- Campos: `id, title, artist, uri, preview_url, mood, valence, energy`.

Ingesta de conocimiento (KB)
- Script: `python -m app.src.ingest_llm_resolver`
- Endpoint: `curl -X POST http://localhost:8010/admin/seed -H "Content-Type: application/json" -d '{"use_spotify": true, "per_emotion": 25}'`
- Dev: con `AUTO_SEED_ON_START=true`, si el Ã­ndice estÃ¡ vacÃ­o se auto-ingestan documentos sintÃ©ticos al iniciar.

Ejemplos
```
curl -X POST http://localhost:8010/rag/search -H "Content-Type: application/json" -d '{"emotion":"happy","min_tracks":20}'
curl -X POST http://localhost:8010/rag/playlist -H "Content-Type: application/json" -d '{"emotion":"sad","min_tracks":25}'
curl -X POST http://localhost:8010/llm/playlist-meta -H "Content-Type: application/json" -d '{"emotion":"happy"}'
curl -X POST http://localhost:8010/llm/answer -H "Content-Type: application/json" -d '{"emotion":"relaxed","prompt":"Sugiere cÃ³mo usar estas canciones para estudiar"}'
```

Notas
- La creaciÃ³n real de playlists en Spotify (cuenta del usuario) corresponde al servicio `moodtune_music`.
- Este servicio no reproduce ni cachea audio protegido.
- Nota de compatibilidad: el endpoint de Spotify `/v1/recommendations` estÃ¡ deprecado. Este servicio no depende de Ã©l; cuando faltan resultados, se apoya en el LLM para sugerir tÃ­tulos y los resuelve contra el Ã­ndice de OpenSearch.




