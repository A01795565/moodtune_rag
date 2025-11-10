"""Pipeline de recuperación (RAG) para pistas según emoción."""
from random import shuffle
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
        """Mapea emoción -> rangos (valence, energy)."""
        params = Config.EMOTION_PARAMS.get(emotion.lower())
        if not params:
            # fallback
            params = {"valence": (0.4, 0.6), "energy": (0.4, 0.6)}
        return params

    def search_tracks(self, emotion: str, min_tracks: int = 20, max_relax_steps: int = 2) -> List[Dict[str, Any]]:
        """Busca pistas para la emoción; relaja rangos si no hay suficientes.

        Reglas adicionales cuando se recurre a LLM:
        - Verifica contra la KB (OpenSearch) si la canción ya existe y la omite.
        - Busca nuevas canciones (sugeridas por el LLM) para completar según la emoción.
        - Evita duplicados por (title, artist) y por URI/ID cuando es posible.
        """
        base = self.emotion_to_params(emotion)
        valence = base.get("valence")
        energy = base.get("energy")

        # Intento estricto + relajar dentro de limites
        for _ in range(max_relax_steps + 1):
            tracks = self.os_repo.search_tracks_by_emotion(
                emotion=emotion.lower(),
                valence=valence,
                energy=energy,
                limit=max(min_tracks * 2, 50),
                randomize=True,
            )
            if len(tracks) >= min_tracks:
                return tracks[: max(min_tracks * 2, 50)]
            # relajar rangos
            valence = relax_range(valence, 0.1)
            energy = relax_range(energy, 0.1)

        existing_keys = set()
        for t in (tracks or []):
            ti = (t.get("title") or "").strip().lower()
            ar = (t.get("artist") or "").strip().lower()
            if ti and ar:
                existing_keys.add((ti, ar))

        try:
            fallback = self._augment_with_llm(emotion, existing_keys, max(min_tracks, 20))
            merged = self._merge_tracks_list([tracks or [], fallback], min_tracks)
            extra_needed = min_tracks - len(merged)
            if extra_needed > 0:
                more = self._augment_with_llm(emotion, existing_keys, extra_needed)
                merged = self._merge_tracks_list([merged, more], min_tracks)
            return merged[: max(min_tracks * 2, 50)] if merged else tracks
        except Exception:
            return tracks  # fallback final

    def _merge_tracks_list(self, lists: List[List[Dict[str, Any]]], min_tracks: int) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen = set()
        for lst in lists:
            for it in lst:
                title = (it.get("title") or "").strip().lower()
                artist = (it.get("artist") or "").strip().lower()
                if not title or not artist:
                    continue
                key = (title, artist)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(it)
                if len(merged) >= min_tracks:
                    return merged
        return merged

    def _augment_with_llm(self, emotion: str, existing_keys: set[tuple[str, str]], desired: int) -> List[Dict[str, Any]]:
        if desired <= 0:
            return []
        if self._llm is None:
            self._llm = LLMClient()
        msc = MusicServiceClient(base_url=Config.MUSIC_SERVICE_URL)

        base = self.emotion_to_params(emotion)
        iter_idx = 0
        pairs: List[Dict[str, str]] = []
        max_iters = 4
        while len(pairs) < desired and iter_idx < max_iters:
            iter_idx += 1
            avoid_list = [{"title": t, "artist": a} for (t, a) in list(existing_keys)[:100]]
            guide = (
                f"valence~{base['valence'][0]:.2f}-{base['valence'][1]:.2f}, "
                f"energy~{base['energy'][0]:.2f}-{base['energy'][1]:.2f}. "
                f"Evita duplicados exactos de título+artista conocidos."
            )
            suggestions = self._llm.curate_songs(emotion, count=desired, avoid=avoid_list, guidance=guide)
            for s in suggestions:
                title = (s.get("title") or "").strip()
                artist = (s.get("artist") or "").strip()
                if not title or not artist:
                    continue
                key = (title.lower(), artist.lower())
                if key in existing_keys:
                    continue
                try:
                    found_in_kb = self.os_repo.find_by_title_artist(title, artist, limit=1)
                except Exception:
                    found_in_kb = []
                if found_in_kb:
                    existing_keys.add(key)
                    continue
                existing_keys.add(key)
                pairs.append({"title": title, "artist": artist})
                if len(pairs) >= desired:
                    break

        if not pairs:
            return []

        resolved = msc.resolve_batch(pairs, per_item_limit=1)
        sp_ids = [t.get("external_id") for t in resolved if (t.get("provider") == "spotify" and t.get("external_id"))]
        feats: Dict[str, Dict[str, Any]] = {}
        if sp_ids:
            try:
                feats = msc.audio_features(sp_ids)
            except Exception:
                feats = {}

        v_mid = round((base["valence"][0] + base["valence"][1]) / 2.0, 2)
        e_mid = round((base["energy"][0] + base["energy"][1]) / 2.0, 2)
        found: List[Dict[str, Any]] = []
        seen = set(existing_keys)
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
            found.append({
                "id": t.get("id") or ext_id,
                "title": title,
                "artist": artist,
                "uri": t.get("uri"),
                "preview_url": t.get("preview_url"),
                "image_url": t.get("image_url"),
                "thumbnail_url": t.get("thumbnail_url"),
                "mood": emotion.lower(),
                "valence": (f or {}).get("valence", v_mid),
                "energy": (f or {}).get("energy", e_mid),
            })

        return found
