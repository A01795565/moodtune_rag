"""Ingesta de KB basada 100% en LLM + resolucion publica (iTunes Search).

Flujo:
- Para cada emocion, pedir al LLM una lista de canciones (titulo/artist).
- Resolver cada sugerencia contra iTunes Search API.
- Indexar documentos normalizados en OpenSearch (sin depender de Spotify).
"""

from typing import Dict, Any, List
from datetime import datetime, timezone

from .llm import LLMClient
from .opensearch_client import OpenSearchRepo
from .config import Config
import requests


def _resolve_via_music_service(title: str, artist: str, limit: int = 1) -> List[Dict[str, Any]]:
    try:
        url = f"{Config.MUSIC_SERVICE_URL.rstrip('/')}/catalog/resolve"
        payload = {"title": title, "artist": artist, "limit": max(1, min(limit, 5))}
        r = requests.post(url, json=payload, timeout=15)
        r.raise_for_status()
        data = r.json() or {}
        return data.get("items") or []
    except Exception:
        return []


def _resolve_batch_via_music_service(pairs: List[Dict[str, str]], per_item_limit: int = 1) -> List[Dict[str, Any]]:
    """Envia una sola peticion al servicio de musica para resolver varios titulo/artista.

    pairs: lista de { title, artist }
    Devuelve una lista de objetos normalizados (uno por pair cuando haya match), en el mismo orden,
    omitiendo los que no tuvieron resultados.
    """
    try:
        url = f"{Config.MUSIC_SERVICE_URL.rstrip('/')}/catalog/resolve-batch"
        payload = {"items": pairs, "per_item_limit": max(1, per_item_limit)}
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json() or {}
        out: List[Dict[str, Any]] = []
        groups = data.get("items") or []
        # Conservar orden; tomar el primer item normalizado por cada grupo
        for g in groups:
            items = g.get("items") or []
            if items:
                out.append(items[0])
        return out
    except Exception:
        return []


def build_doc_from_resolved(emotion: str, t: Dict[str, Any]) -> Dict[str, Any]:
    title = t.get("title")
    artist = t.get("artist")
    uri = t.get("uri")
    preview_url = t.get("preview_url")
    image_url = t.get("image_url") or t.get("artworkUrl100") or t.get("thumbnail_url")
    thumbnail_url = t.get("thumbnail_url") or t.get("artworkUrl60") or image_url
    provider = (t.get("provider") or "ext").lower()
    external_id = t.get("external_id")
    params = Config.EMOTION_PARAMS.get(emotion.lower()) or {"valence": (0.4, 0.6), "energy": (0.4, 0.6)}
    v = round((params["valence"][0] + params["valence"][1]) / 2.0, 2)
    e = round((params["energy"][0] + params["energy"][1]) / 2.0, 2)
    _id = t.get("id") or f"{provider}-{emotion}-{title}-{artist}"
    return {
        "id": _id,
        "external_id": external_id,
        "provider": provider,
        "source": "llm+music",
        "title": title,
        "artist": artist,
        "uri": uri,
        "preview_url": preview_url,
        "image_url": image_url,
        "thumbnail_url": thumbnail_url,
        "mood": emotion.lower(),
        "valence": v,
        "energy": e,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def ingest_emotions_llm(emotions: List[str] | None = None, per_emotion: int = 50) -> Dict[str, Any]:
    emotions = emotions or list(Config.EMOTION_PARAMS.keys())
    llm = LLMClient()
    repo = OpenSearchRepo()
    repo.ensure_index_exists()

    results: Dict[str, Any] = {"indexed": 0, "by_emotion": {}, "source": "llm+music"}
    for emo in emotions:
        suggestions = llm.recommend_songs(emo, count=per_emotion)
        # Preparar pares validos title/artist
        pairs: List[Dict[str, str]] = []
        for s in suggestions:
            title = (s.get("title") or "").strip()
            artist = (s.get("artist") or "").strip()
            if title and artist:
                pairs.append({"title": title, "artist": artist})
        # Resolver en una sola llamada
        resolved = _resolve_batch_via_music_service(pairs, per_item_limit=1)
        # Preparar textos a embeber con info del LLM (titulo+artista+emocion)
        docs: List[Dict[str, Any]] = []
        texts: List[str] = []
        for t in resolved:
            doc = build_doc_from_resolved(emo, t)
            if doc.get("title") and doc.get("artist"):
                llm_text = f"{emo} | {doc['title']} - {doc['artist']}"
                doc["llm_text"] = llm_text
                texts.append(llm_text)
                docs.append(doc)
        # Obtener embeddings y adjuntar al documento
        if texts:
            vectors = llm.embed_texts(texts)
            if vectors and len(vectors) == len(docs):
                for i, v in enumerate(vectors):
                    docs[i]["embedding"] = v
        indexed = repo.index_tracks(docs)
        results["by_emotion"][emo] = indexed
        results["indexed"] += indexed
    return results


if __name__ == "__main__":
    print(ingest_emotions_llm())
