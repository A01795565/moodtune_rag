"""Siembra documentos via LLM + moodtune_music (resolve-batch) con enriquecimiento.

Flujo por emocion:
- Pide al LLM una lista de {title, artist} (per_emotion entradas).
- Resuelve en lote via `moodtune_music` (/catalog/resolve-batch).
- Si hay items de Spotify, consulta audio-features para obtener valence/energy.
- Construye documentos normalizados y los indexa en OpenSearch.
"""

from typing import Dict, Any, List

from .config import Config
from .opensearch_client import OpenSearchRepo
from .music_service_client import MusicServiceClient
from .llm import LLMClient


def _build_doc(emotion: str, item: Dict[str, Any], features: Dict[str, Any] | None, params_map: Dict[str, Any] | None = None) -> Dict[str, Any]:
    title = item.get("name") or item.get("title")
    if item.get("artists"):
        artist = (item.get("artists")[0] or {}).get("name")
    else:
        artist = item.get("artist")
    uri = item.get("uri")
    preview_url = item.get("preview_url")
    image_url = item.get("image_url")
    thumb = item.get("thumbnail_url")
    external_id = item.get("id") or item.get("external_id")
    provider = (item.get("provider") or "spotify").lower()
    params = (params_map or {}).get(emotion.lower()) or {"valence": (0.4, 0.6), "energy": (0.4, 0.6)}
    v_mid = round((params["valence"][0] + params["valence"][1]) / 2.0, 2)
    e_mid = round((params["energy"][0] + params["energy"][1]) / 2.0, 2)
    val = (features or {}).get("valence", v_mid)
    eng = (features or {}).get("energy", e_mid)
    return {
        "id": f"{provider}-{external_id}" if external_id else f"{provider}-{emotion}-{title}-{artist}",
        "external_id": str(external_id) if external_id else None,
        "provider": provider,
        "source": "llm+music",
        "title": title,
        "artist": artist,
        "uri": uri,
        "preview_url": preview_url,
        "image_url": image_url,
        "thumbnail_url": thumb,
        "mood": emotion.lower(),
        "valence": val,
        "energy": eng,
    }


def seed_knowledge(per_emotion: int = 25) -> Dict[str, Any]:
    """Precarga el índice usando LLM + moodtune_music (resolve-batch + audio-features)."""
    repo = OpenSearchRepo()
    repo.ensure_index_exists()
    msc = MusicServiceClient(base_url=Config.MUSIC_SERVICE_URL)
    llm = LLMClient()

    # Obtener emociones y parámetros desde el servicio de música; fallback a config local si falla
    try:
        emotion_params_map = msc.list_emotions()
    except Exception:
        emotion_params_map = {}
    emotions = list(emotion_params_map.keys()) or list(Config.EMOTION_PARAMS.keys())
    result: Dict[str, Any] = {"indexed": 0, "by_emotion": {}, "source": "llm+music"}
    for emo in emotions:
        # Pedir sugerencias al LLM
        suggestions = llm.recommend_songs(emo, count=per_emotion)
        pairs: List[Dict[str, str]] = []
        for s in suggestions:
            title = (s.get("title") or "").strip()
            artist = (s.get("artist") or "").strip()
            if title and artist:
                pairs.append({"title": title, "artist": artist})

        # Resolver en lote vía music service
        resolved: List[Dict[str, Any]] = []
        if pairs:
            try:
                batch_size = 25
                for i in range(0, len(pairs), batch_size):
                    chunk = pairs[i:i + batch_size]
                    chunk_res = msc.resolve_batch(chunk, per_item_limit=1)
                    if chunk_res:
                        resolved.extend(chunk_res)
            except Exception:
                resolved = []

        # Audio-features para items de Spotify
        feats_map: Dict[str, Dict[str, Any]] = {}
        sp_ids = [it.get("external_id") for it in resolved if (it.get("provider") == "spotify" and it.get("external_id"))]
        if sp_ids:
            try:
                feats_map = msc.audio_features(sp_ids)
            except Exception:
                feats_map = {}

        # Construir docs
        docs: List[Dict[str, Any]] = []
        for it in resolved:
            ext_id = it.get("external_id")
            feats = feats_map.get(ext_id) if ext_id else None
            doc = _build_doc(emo, it, feats, params_map=emotion_params_map)
            if doc.get("title") and doc.get("artist"):
                docs.append(doc)

        # Indexar
        indexed = repo.index_tracks(docs)
        result["by_emotion"][emo] = indexed
        result["indexed"] += indexed
    return result


if __name__ == "__main__":
    print(seed_knowledge())
