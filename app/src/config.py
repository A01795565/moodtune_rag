"""Configuracion central del servicio MoodTune RAG.

Lee variables de entorno (via .env) y expone parametros usados por Flask,
OpenSearch, OpenAI y la logica de RAG.
"""

import os
from dotenv import load_dotenv


class Config:
    # Cargar variables desde .env si esta presente
    load_dotenv()

    # Flask
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

    # OpenSearch
    OS_HOST = os.getenv("OS_HOST", "opensearch")  # service name in docker compose
    OS_PORT = int(os.getenv("OS_PORT", "9200"))
    OS_USER = os.getenv("OS_USER", "admin")
    OS_PASSWORD = os.getenv("OS_PASSWORD", "admin")
    OS_USE_SSL = os.getenv("OS_USE_SSL", "false").lower() == "true"
    OS_VERIFY_CERTS = os.getenv("OS_VERIFY_CERTS", "false").lower() == "true"
    OS_INDEX_TRACKS = os.getenv("OS_INDEX_TRACKS", "moodtune_tracks")

    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    OPENAI_EMBEDDING_DIM = int(os.getenv("OPENAI_EMBEDDING_DIM", "1536"))

    # Servicios externos
    MUSIC_SERVICE_URL = os.getenv("MUSIC_SERVICE_URL", "http://localhost:8020")
       
    # Provider defaults    
    MIN_TRACKS = max(20, int(os.getenv("MIN_TRACKS", "20")))

    # Parametros de ingesta (limite por emocion)
    INGEST_LIMIT_PER_EMOTION = int(os.getenv("INGEST_LIMIT_PER_EMOTION", "200"))

    # Dev convenience: auto seed synthetic data on startup if index is empty
    AUTO_SEED_ON_START = os.getenv("AUTO_SEED_ON_START", "true").lower() == "true"

    # RAG defaults / mapping
    # Valores de referencia para mapear emocion -> filtros (pueden relajarse)
    EMOTION_PARAMS = {
        "happy": {"valence": (0.6, 1.0), "energy": (0.5, 1.0)},
        "sad": {"valence": (0.0, 0.4), "energy": (0.0, 0.5)},
        "angry": {"valence": (0.2, 0.6), "energy": (0.6, 1.0)},
        "relaxed": {"valence": (0.5, 1.0), "energy": (0.0, 0.5)},
    }
