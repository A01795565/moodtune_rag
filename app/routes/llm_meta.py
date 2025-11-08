"""Endpoints LLM: metadatos de playlist y respuestas con contexto."""
from flask import Blueprint, jsonify, request
from ..src.llm import LLMClient
from ..src.rag_pipeline import RAGPipeline
from ..src.opensearch_client import OpenSearchRepo


bp = Blueprint("llm", __name__)


@bp.post("/playlist-meta")
def playlist_meta():
    """Genera titulo y descripcion de playlist para una emocion."""
    try:
        p = request.get_json(force=True) or {}
        emotion = (p.get("emotion") or "relaxed").lower()

        llm = LLMClient()
        title, description = llm.playlist_title_and_description(emotion)
        return jsonify({"title": title, "description": description}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@bp.post("/answer")
def answer_with_emotion():
    """Responde un prompt usando emocion + contexto recuperado por RAG."""
    try:
        p = request.get_json(force=True) or {}
        emotion = (p.get("emotion") or "relaxed").lower()
        prompt = (p.get("prompt") or "").strip()
        if not prompt:
            return jsonify({"error": "prompt requerido"}), 400

        # Recuperar algunas pistas como contexto para la emocion
        repo = OpenSearchRepo()
        rag = RAGPipeline(repo)
        context_items = rag.search_tracks(emotion=emotion, min_tracks=20)

        llm = LLMClient()
        answer = llm.answer_with_context(emotion, prompt, context_items)
        return jsonify({"emotion": emotion, "prompt": prompt, "answer": answer}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
