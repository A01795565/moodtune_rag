"""Endpoints RAG para buscar pistas y armar playlist sugerida."""
import random

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
        requested_min = int(p.get("min_tracks") or Config.MIN_TRACKS)
        min_tracks = max(Config.MIN_TRACKS, requested_min)

        repo = OpenSearchRepo()
        rag = RAGPipeline(repo)
        # 1) Recuperar base desde KB (con relajacion y posible fallback del pipeline)
        tracks = rag.search_tracks(emotion=emotion, min_tracks=min_tracks)

        # 2) Curación/augment con LLM para variedad (es/en), mezclando nuevas + existentes
        #    - Evitar repetir (title, artist)
        #    - Usar guías de valence/energy
        base_params = rag.emotion_to_params(emotion)
        v0, v1 = base_params.get("valence", (0.4, 0.6))
        e0, e1 = base_params.get("energy", (0.4, 0.6))

        existing_keys = set()
        for t in (tracks or []):
            ti = (t.get("title") or "").strip().lower()
            ar = (t.get("artist") or "").strip().lower()
            if ti and ar:
                existing_keys.add((ti, ar))

        # Cantidad a aumentar para variedad
        extra_needed = max(0, min_tracks // 2)
        augmented = []
        if extra_needed > 0:
            try:
                guide = f"valence~{v0:.2f}-{v1:.2f}, energy~{e0:.2f}-{e1:.2f}"
                avoid_list = [{"title": t, "artist": a} for (t, a) in list(existing_keys)[:100]]

                llm = LLMClient()
                suggestions = llm.curate_songs(
                    emotion,
                    count=extra_needed,
                    avoid=avoid_list,
                    guidance=guide,
                )

                # Resolver y enriquecer
                from ..src.music_service_client import MusicServiceClient
                msc = MusicServiceClient(base_url=Config.MUSIC_SERVICE_URL)

                pairs = []
                for s in suggestions:
                    title = (s.get("title") or "").strip()
                    artist = (s.get("artist") or "").strip()
                    if not title or not artist:
                        continue
                    key = (title.lower(), artist.lower())
                    if key in existing_keys:
                        continue
                    pairs.append({"title": title, "artist": artist})
                    existing_keys.add(key)

                resolved = msc.resolve_batch(pairs, per_item_limit=1) if pairs else []

                sp_ids = [t.get("external_id") for t in resolved if (t.get("provider") == "spotify" and t.get("external_id"))]
                feats = {}
                if sp_ids:
                    try:
                        feats = msc.audio_features(sp_ids)
                    except Exception:
                        feats = {}

                v_mid = round((v0 + v1) / 2.0, 2)
                e_mid = round((e0 + e1) / 2.0, 2)
                for t in resolved:
                    title = t.get("title")
                    artist = t.get("artist")
                    if not title or not artist:
                        continue
                    key = (title.lower(), artist.lower())
                    if key in {( (x.get("title") or "").lower(), (x.get("artist") or "").lower()) for x in augmented }:
                        continue
                    ext_id = t.get("external_id")
                    f = feats.get(ext_id) if ext_id else None
                    augmented.append({
                        "id": t.get("id") or ext_id,
                        "title": title,
                        "artist": artist,
                        "uri": t.get("uri"),
                        "preview_url": t.get("preview_url"),
                        "image_url": t.get("image_url"),
                        "thumbnail_url": t.get("thumbnail_url"),
                        "mood": emotion,
                        "valence": (f or {}).get("valence", v_mid),
                        "energy": (f or {}).get("energy", e_mid),
                    })
            except Exception:
                augmented = []

        random.shuffle(augmented)
        # 3) Mezclar sin duplicados (prioriza KB, luego LLM)
        merged = []
        seen = set()
        for lst in (tracks or []), augmented:
            for it in lst:
                k = ((it.get("title") or "").strip().lower(), (it.get("artist") or "").strip().lower())
                if not k[0] or not k[1]:
                    continue
                if k in seen:
                    continue
                seen.add(k)
                merged.append(it)

        random.shuffle(merged)

        # 4) Guardar en OpenSearch (upsert). Añadir embedding/llm_text cuando sea posible
        indexed = 0
        try:
            from datetime import datetime
            to_index = []
            texts = []
            for it in merged:
                doc = dict(it)
                if not doc.get("id"):
                    # ID estable a partir de titulo+artista+emocion
                    ti = (doc.get("title") or "").strip().lower().replace(" ", "-")
                    ar = (doc.get("artist") or "").strip().lower().replace(" ", "-")
                    doc["id"] = f"llm-{emotion}-{ti}-{ar}"[:256]
                doc.setdefault("mood", emotion)
                doc.setdefault("created_at", datetime.utcnow().isoformat() + "Z")
                # Preparar texto para embedding
                llm_text = f"{emotion} | {doc.get('title','')} - {doc.get('artist','')}"
                doc["llm_text"] = llm_text
                texts.append(llm_text)
                to_index.append(doc)

            # Embeddings (best-effort)
            try:
                llm_emb = LLMClient()
                vecs = llm_emb.embed_texts(texts)
                if vecs and len(vecs) == len(to_index):
                    for d, v in zip(to_index, vecs):
                        d["embedding"] = v
            except Exception:
                pass

            indexed = repo.index_tracks(to_index)
        except Exception:
            indexed = 0

        note = None
        if len(merged) < min_tracks:
            note = f"Resultados insuficientes ({len(merged)}/{min_tracks}); se relajaron parametros."
        elif augmented:
            note = f"Se añadieron {len(augmented)} pistas nuevas via curador LLM. Indexadas: {indexed}."

        out = {
            "emotion": emotion,
            "requested_min": min_tracks,
            "returned": len(merged),
            "items": merged[:max(min_tracks * 2, 50)],
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

        # Recuperar candidatas via RAG
        repo = OpenSearchRepo()
        rag = RAGPipeline(repo)
        tracks = rag.search_tracks(emotion=emotion, min_tracks=min_tracks)

        # Generar titulo/descripcion con LLM
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
            out["note"] = f"Resultados insuficientes ({len(tracks)}/{min_tracks}); se relajaron parametros."
        return jsonify(out), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400
