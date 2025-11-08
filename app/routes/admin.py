"""Endpoints administrativos (solo uso en desarrollo).

La precarga usa el servicio `moodtune_music` para obtener recomendaciones
y audio-features y poblar OpenSearch. Para reconstruccion completa, se
recrea el indice y se vuelve a sembrar.
"""
from flask import Blueprint, jsonify, request
from ..src.seed_data import seed_knowledge
from ..src.opensearch_client import OpenSearchRepo


bp = Blueprint("admin", __name__)


@bp.post("/seed")
def seed():
    """Precarga el indice con datos obtenidos desde moodtune_music.

    Body JSON opcional: { per_emotion?: int }
    """
    try:
        p = request.get_json(silent=True) or {}
        per_emotion = int(p.get("per_emotion", 25))
        out = seed_knowledge(per_emotion=per_emotion)
        return jsonify(out), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@bp.post("/rebuild")
def rebuild():
    """Recrea el indice y reingesta la KB desde moodtune_music.

    Body JSON opcional: { per_emotion?: int }
    """
    try:
        p = request.get_json(silent=True) or {}
        per_emotion = int(p.get("per_emotion", 50))
        repo = OpenSearchRepo()
        repo.recreate_index()
        out = seed_knowledge(per_emotion=per_emotion)
        return jsonify({"index": repo.index, **out}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400
