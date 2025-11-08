"""Endpoints de salud (healthcheck) del servicio RAG."""
from flask import Blueprint, jsonify
from ..src.config import Config


bp = Blueprint("health", __name__)


@bp.get("/health")
def health():
    """Devuelve estado básico del servicio para monitoreo."""
    return jsonify({
        "status": "ok",
        "service": "moodtune_rag",
        "debug": Config.DEBUG,
    }), 200
