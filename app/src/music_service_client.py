"""Cliente HTTP para consumir el servicio moodtune_music (Spotify wrapper)."""

from typing import Dict, Any, List
import os
import requests
from .config import Config


class MusicServiceClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or os.getenv("MUSIC_SERVICE_URL") or "http://localhost:8020"

    def recommendations(self, emotion: str, limit: int = 50) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/catalog/recommendations"
        r = requests.post(url, json={"emotion": emotion, "limit": limit}, timeout=20)
        r.raise_for_status()
        data = r.json()
        return data.get("items", [])

    def audio_features(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        url = f"{self.base_url}/catalog/audio-features"
        r = requests.post(url, json={"ids": ids}, timeout=20)
        r.raise_for_status()
        data = r.json()
        return data.get("items", {})

    def emotion_params(self, emotion: str) -> Dict[str, Any] | None:
        url = f"{self.base_url}/catalog/emotions/{emotion}"
        r = requests.get(url, timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json().get("params")

    def list_emotions(self) -> Dict[str, Any]:
        url = f"{self.base_url}/catalog/emotions"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json().get("items", {})

    def resolve(self, title: str, artist: str, limit: int = 1) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/catalog/resolve"
        r = requests.post(url, json={"title": title, "artist": artist, "limit": max(1, limit)}, timeout=20)
        r.raise_for_status()
        data = r.json() or {}
        return data.get("items") or []

    def resolve_batch(self, items: List[Dict[str, str]], per_item_limit: int = 1) -> List[Dict[str, Any]]:
        """Llama a /catalog/resolve-batch y devuelve el primer match por item de entrada.

        items: [{ title, artist }]
        """
        url = f"{self.base_url}/catalog/resolve-batch"
        payload = {"items": items, "per_item_limit": max(1, per_item_limit)}
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json() or {}
        out: List[Dict[str, Any]] = []
        for g in (data.get("items") or []):
            first = (g.get("items") or [None])[0]
            if first:
                out.append(first)
        return out
