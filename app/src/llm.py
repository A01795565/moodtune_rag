from typing import Tuple, List, Dict
import os
import openai
from .config import Config


class LLMClient:
    def __init__(self):
        api_key = Config.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY no configurado")
        openai.api_key = api_key
        self.model = Config.OPENAI_MODEL

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Obtiene embeddings con OpenAI para una lista de textos.

        Usa el modelo `text-embedding-3-small` por defecto (configurable).
        """
        if not texts:
            return []
        try:
            resp = openai.Embedding.create(
                model=Config.OPENAI_EMBEDDING_MODEL,
                input=texts,
            )
            # SDK v0 devuelve .data[i].embedding
            vectors: List[List[float]] = [item["embedding"] for item in resp["data"]]
            return vectors
        except Exception:
            return []

    def playlist_title_and_description(self, emotion: str) -> Tuple[str, str]:
        system = "Eres un asistente que crea títulos y descripciones para playlists de música, en español e ingles, concisos y atractivos."
        user = f"Genera título y descripción breve (1-2 líneas) para una playlist basada en la emoción '{emotion}'. No uses emojis."
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.7,
            max_tokens=120,
        )
        text = resp["choices"][0]["message"]["content"].strip()
        parts = [p.strip() for p in text.split("\n") if p.strip()]
        title = parts[0][:100]
        desc = parts[1] if len(parts) > 1 else f"Playlist generada para estado '{emotion}'."
        return title, desc[:300]

    def answer_with_context(self, emotion: str, prompt: str, context_items: list[dict]) -> str:
        # Build a concise context from tracks
        lines = []
        for t in context_items[:20]:  # limit context size
            title = (t.get("title") or "").strip()
            artist = (t.get("artist") or "").strip()
            mood = (t.get("mood") or "").strip()
            ve = f"v={t.get('valence', 0.0):.2f}, e={t.get('energy', 0.0):.2f}"
            if title and artist:
                lines.append(f"- {title} — {artist} [{mood}; {ve}]")

        system = (
            "Eres un asistente musical en español. Usa el contexto de pistas"
            " (título, artista, y sus métricas) para responder de forma breve,"
            " útil y enfocada a la emoción indicada. Si el contexto no cubre algo,"
            " responde de forma general sin inventar detalles."
        )
        user = (
            f"Emoción: {emotion}\n"
            f"Contexto (tracks):\n" + "\n".join(lines) + "\n\n"
            f"Pregunta: {prompt}"
        )
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.6,
            max_tokens=300,
        )
        return resp["choices"][0]["message"]["content"].strip()

    def recommend_songs(self, emotion: str, count: int = 20) -> List[Dict[str, str]]:
        """Solicita al LLM una lista de canciones sugeridas para la emoción.

        Intenta obtener una salida en JSON con objetos {"title":..., "artist":...}.
        """
        system = (
            "Eres un curador musical, tu objetivo es sugerir canciones en español e ingles. Devuelve SOLO un JSON válido con"
            " una lista de canciones adecuadas para la emoción dada."
        )
        user = (
            f"Emoción objetivo: {emotion}. Devuelve un arreglo JSON de {count} objetos con claves exactas 'title' y 'artist'. "
            f"No incluyas comentarios ni texto fuera del JSON. Ejemplo: [{{\"title\":\"...\",\"artist\":\"...\"}}]"
        )
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.6,
            max_tokens=600,
        )
        text = resp["choices"][0]["message"]["content"].strip()
        import json
        songs: List[Dict[str, str]] = []
        try:
            songs = json.loads(text)
            # sanitize
            out: List[Dict[str, str]] = []
            for it in songs:
                title = (it.get("title") or "").strip()
                artist = (it.get("artist") or "").strip()
                if title and artist:
                    out.append({"title": title, "artist": artist})
            return out[:count]
        except Exception:
            # Fallback: parse lines if model didn't return pure JSON
            for line in text.splitlines():
                if "-" in line:
                    parts = line.split("-", 1)
                    t = parts[0].strip().strip("-• ")
                    a = parts[1].strip()
                    if t and a:
                        songs.append({"title": t, "artist": a})
            return songs[:count]
