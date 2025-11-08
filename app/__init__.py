"""InicializaciÃ³n de la app Flask para el servicio MoodTune RAG.

Responsabilidades principales:
- Configurar CORS segÃºn variables de entorno.
- Registrar blueprints (salud, RAG, LLM y utilidades de administraciÃ³n).
- Al iniciar en modo desarrollo, crear el Ã­ndice en OpenSearch y (opcionalmente)
  precargar datos sintÃ©ticos si el Ã­ndice estÃ¡ vacÃ­o para evitar errores 404.
"""

from flask import Flask, jsonify
try:
    from flask_cors import CORS
except Exception:
    CORS = None

from .src.config import Config
from .routes.health import bp as health_bp
from .routes.rag import bp as rag_bp
from .routes.llm_meta import bp as llm_bp
from .routes.admin import bp as admin_bp


def create_app():
    # Crear app y cargar configuraciÃ³n desde Config
    app = Flask(__name__)
    app.config.from_object(Config)

    # Habilitar CORS si estÃ¡ disponible la extensiÃ³n
    if CORS:
        origins = Config.CORS_ORIGINS if hasattr(Config, 'CORS_ORIGINS') else '*'
        CORS(app, resources={r"/*": {"origins": origins}}, supports_credentials=True)

    # Endpoints de salud
    app.register_blueprint(health_bp)

    # Endpoints de RAG y LLM
    app.register_blueprint(rag_bp, url_prefix="/rag")
    app.register_blueprint(llm_bp, url_prefix="/llm")
    # Endpoints administrativos (solo uso dev)
    app.register_blueprint(admin_bp, url_prefix="/admin")

    @app.get("/")
    def root():
        # Ruta raÃ­z simple para inspecciÃ³n rÃ¡pida
        return jsonify({"name": "moodtune_rag", "status": "ok"}), 200

    # Comodidad para desarrollo: asegurar Ã­ndice en OpenSearch y
    # precargar datos sintÃ©ticos si estÃ¡ vacÃ­o (evita errores 404/Ã­ndice no encontrado)
    try:
        from .src.opensearch_client import OpenSearchRepo
        from .src.config import Config as _Cfg
        repo = OpenSearchRepo()
        repo.ensure_index_exists()
        if _Cfg.DEBUG and _Cfg.AUTO_SEED_ON_START:
            try:
                if repo.count() == 0:
                    from .src.seed_data import seed_knowledge
                    seed_knowledge(per_emotion=max(10, _Cfg.MIN_TRACKS))
            except Exception:
                pass
    except Exception:
        pass

    return app


