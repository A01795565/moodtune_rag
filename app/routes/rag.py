"""Endpoints RAG para buscar pistas y armar playlist sugerida."""
from flask import Blueprint, jsonify, request
from ..src.rag_pipeline import RAGPipeline
from ..src.opensearch_client import OpenSearchRepo
from ..src.config import Config
from ..src.llm import LLMClient


bp = Blueprint("rag", __name__)


@bp.post("/search")
def search_tracks():
    """Recupera pistas desde OpenSearch según emoción.

    Body JSON: { emotion: str, min_tracks?: int }
    """
    try:
        p = request.get_json(force=True) or {}
        emotion = (p.get("emotion") or "relaxed").lower()
        min_tracks = int(p.get("min_tracks") or Config.MIN_TRACKS)

        repo = OpenSearchRepo()
        rag = RAGPipeline(repo)
        tracks = rag.search_tracks(emotion=emotion, min_tracks=min_tracks)

        note = None
        if len(tracks) < min_tracks:
            note = f"Resultados insuficientes ({len(tracks)}/{min_tracks}); se relajaron parámetros."

        out = {
            "emotion": emotion,
            "requested_min": min_tracks,
            "returned": len(tracks),
            "items": tracks[:max(min_tracks * 2, 50)],
        }
        if note:
            out["note"] = note
        return jsonify(out), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@bp.post("/playlist")
def build_playlist():
    """Arma una playlist sugerida: metadatos (LLM) + canciones (RAG)."""
    try:
        p = request.get_json(force=True) or {}
        emotion = (p.get("emotion") or "relaxed").lower()
        min_tracks = int(p.get("min_tracks") or Config.MIN_TRACKS)

        # Recuperar candidatas vía RAG
        repo = OpenSearchRepo()
        rag = RAGPipeline(repo)
        tracks = rag.search_tracks(emotion=emotion, min_tracks=min_tracks)

        # Generar título/descripcion con LLM
        llm = LLMClient()
        title, description = llm.playlist_title_and_description(emotion)

        uris = [t.get("uri") for t in tracks if t.get("uri")]
        out = {
            "emotion": emotion,
            "title": title,
            "description": description,
            "returned": len(tracks),
            "items": tracks[: max(min_tracks * 2, 50)],
            "uris": uris[: max(min_tracks * 2, 50)],
        }
        if len(tracks) < min_tracks:
            out["note"] = f"Resultados insuficientes ({len(tracks)}/{min_tracks}); se relajaron parámetros."
        return jsonify(out), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
