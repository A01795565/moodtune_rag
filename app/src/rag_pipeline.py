"""Pipeline de recuperación (RAG) para pistas según emoción."""
from typing import Dict, Any, List, Tuple
from .config import Config
from .opensearch_client import OpenSearchRepo
from .music_service_client import MusicServiceClient
from .utils import relax_range
from .llm import LLMClient


class RAGPipeline:
    """Orquesta búsquedas a OpenSearch aplicando reglas por emoción."""

    def __init__(self, os_repo: OpenSearchRepo | None = None):
        self.os_repo = os_repo or OpenSearchRepo()
        self._llm: LLMClient | None = None

    def emotion_to_params(self, emotion: str) -> Dict[str, Tuple[float, float]]:
        """Mapea emoción → rangos (valence, energy)."""
        params = Config.EMOTION_PARAMS.get(emotion.lower())
        if not params:
            # fallback
            params = {"valence": (0.4, 0.6), "energy": (0.4, 0.6)}
        return params

    def search_tracks(self, emotion: str, min_tracks: int = 20, max_relax_steps: int = 2) -> List[Dict[str, Any]]:
        """Busca pistas para la emoción; relaja rangos si no hay suficientes."""
        base = self.emotion_to_params(emotion)
        valence = base.get("valence")
        energy = base.get("energy")

        # Intento estricto → relajar dentro de límites
        for _ in range(max_relax_steps + 1):
            tracks = self.os_repo.search_tracks_by_emotion(
                emotion=emotion.lower(),
                valence=valence,
                energy=energy,
                limit=max(min_tracks * 2, 50),
            )
            if len(tracks) >= min_tracks:
                return tracks[: max(min_tracks * 2, 50)]
            # relajar rangos
            valence = relax_range(valence, 0.1)
            energy = relax_range(energy, 0.1)

        # Fallback: LLM sugiere títulos; resolver con servicio de música y enriquecer atributos
        try:
            if self._llm is None:
                self._llm = LLMClient()
            suggestions = self._llm.recommend_songs(emotion, count=max(min_tracks, 20))
            # preparar pares para resolución batch
            pairs: List[Dict[str, str]] = []
            for s in suggestions:
                title = (s.get("title") or "").strip()
                artist = (s.get("artist") or "").strip()
                if title and artist:
                    pairs.append({"title": title, "artist": artist})

            msc = MusicServiceClient(base_url=Config.MUSIC_SERVICE_URL)
            resolved = msc.resolve_batch(pairs, per_item_limit=1)

            # Enriquecer con audio-features cuando sea Spotify
            sp_ids = [t.get("external_id") for t in resolved if (t.get("provider") == "spotify" and t.get("external_id"))]
            feats: Dict[str, Dict[str, Any]] = {}
            if sp_ids:
                try:
                    feats = msc.audio_features(sp_ids)
                except Exception:
                    feats = {}

            # Construir salida con valence/energy (features si hay; si no, centrado por emoción)
            v_mid = round((base["valence"][0] + base["valence"][1]) / 2.0, 2)
            e_mid = round((base["energy"][0] + base["energy"][1]) / 2.0, 2)
            found: List[Dict[str, Any]] = []
            seen = set()
            for t in resolved:
                title = t.get("title")
                artist = t.get("artist")
                if not title or not artist:
                    continue
                key = (title.lower(), artist.lower())
                if key in seen:
                    continue
                seen.add(key)
                ext_id = t.get("external_id")
                f = feats.get(ext_id) if ext_id else None
                val = (f or {}).get("valence", v_mid)
                eng = (f or {}).get("energy", e_mid)
                found.append({
                    "id": t.get("id") or ext_id,
                    "title": title,
                    "artist": artist,
                    "uri": t.get("uri"),
                    "preview_url": t.get("preview_url"),
                    "image_url": t.get("image_url"),
                    "thumbnail_url": t.get("thumbnail_url"),
                    "mood": emotion.lower(),
                    "valence": val,
                    "energy": eng,
                })

            merged = tracks + [t for t in found if t not in tracks]
            return merged[: max(min_tracks * 2, 50)] if merged else found
        except Exception:
            return tracks  # fallback final
