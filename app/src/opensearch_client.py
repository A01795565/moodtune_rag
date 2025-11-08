"""Cliente/repositorio de OpenSearch para documentos de pistas.

Expone operaciones comunes:
- asegurar existencia del índice con el mapping esperado,
- indexar/upsert de documentos,
- búsqueda por emoción con filtros de valence/energy,
- conteo de documentos.
"""

from typing import List, Dict, Any
from opensearchpy import OpenSearch, NotFoundError
from .config import Config


class OpenSearchRepo:
    """Encapsula las operaciones contra OpenSearch para el índice de tracks."""

    def __init__(self):
        self.client = OpenSearch(
            hosts=[{"host": Config.OS_HOST, "port": Config.OS_PORT}],
            http_auth=(Config.OS_USER, Config.OS_PASSWORD),
            use_ssl=Config.OS_USE_SSL,
            verify_certs=Config.OS_VERIFY_CERTS,
        )
        self.index = Config.OS_INDEX_TRACKS

    def search_tracks_by_emotion(self, emotion: str, valence: tuple, energy: tuple, limit: int = 20) -> List[Dict[str, Any]]:
        """Busca documentos por emoción aplicando filtros de `valence` y `energy`.

        Devuelve una lista de fuentes (_source) con los campos relevantes.
        Si el índice no existe (dev), lo crea y retorna lista vacía.
        """
        must_filters = []
        if valence:
            must_filters.append({"range": {"valence": {"gte": valence[0], "lte": valence[1]}}})
        if energy:
            must_filters.append({"range": {"energy": {"gte": energy[0], "lte": energy[1]}}})
        if emotion:
            must_filters.append({"term": {"mood": emotion}})

        query = {"bool": {"must": must_filters}} if must_filters else {"match_all": {}}
        body = {
            "size": limit,
            "query": query,
            "_source": [
                "id", "title", "artist", "preview_url", "uri",
                "valence", "energy", "mood", "image_url", "thumbnail_url"
            ],
        }
        try:
            res = self.client.search(index=self.index, body=body)
        except NotFoundError:
            # Create index on first access to avoid 404s in dev
            try:
                self.ensure_index_exists()
            except Exception:
                pass
            return []
        hits = res.get("hits", {}).get("hits", [])
        return [h.get("_source", {}) for h in hits]

    def index_tracks(self, docs: List[Dict[str, Any]]) -> int:
        """Indexa/upserta documentos. Usa `id` de track como `_id` para idempotencia."""
        if not docs:
            return 0
        # Ensure index exists (best-effort)
        self.ensure_index_exists()

        indexed = 0
        for d in docs:
            try:
                # Use track id as document id to upsert
                _id = d.get("id")
                self.client.index(index=self.index, id=_id, body=d)
                indexed += 1
            except Exception:
                continue
        # Force a refresh so that subsequent reads see the documents immediately
        try:
            self.client.indices.refresh(index=self.index)
        except Exception:
            pass
        return indexed

    def ensure_index_exists(self) -> None:
        """Crea el índice con mapping por defecto si no existe (modo dev)."""
        try:
            if not self.client.indices.exists(self.index):
                # Create with current mapping (align with docs/opensearch_index_mapping.json)
                self.client.indices.create(index=self.index, body={
                    "settings": {
                        "index": {
                            "number_of_shards": 1,
                            "number_of_replicas": 0,
                            "knn": True,
                            "analysis": {
                                "analyzer": {
                                    "folding": {
                                        "tokenizer": "standard",
                                        "filter": ["lowercase", "asciifolding"]
                                    }
                                }
                            }
                        }
                    },
                    "mappings": {
                        "properties": {
                            "id": {"type": "keyword"},
                            "external_id": {"type": "keyword"},
                            "provider": {"type": "keyword"},
                            "source": {"type": "keyword"},
                            "image_url": {"type": "keyword"},
                            "thumbnail_url": {"type": "keyword"},
                            "llm_text": {"type": "text", "analyzer": "folding"},
                            "embedding": {
                                "type": "knn_vector",
                                "dimension": Config.OPENAI_EMBEDDING_DIM,
                                "method": {
                                    "engine": "lucene",
                                    "space_type": "cosinesimil",
                                    "name": "hnsw"
                                }
                            },
                            "title": {
                                "type": "text",
                                "analyzer": "folding",
                                "fields": {"raw": {"type": "keyword", "ignore_above": 256}}
                            },
                            "artist": {
                                "type": "text",
                                "analyzer": "folding",
                                "fields": {"raw": {"type": "keyword", "ignore_above": 256}}
                            },
                            "uri": {"type": "keyword"},
                            "preview_url": {"type": "keyword"},
                            "mood": {"type": "keyword"},
                            "valence": {"type": "float"},
                            "energy": {"type": "float"},
                            "created_at": {"type": "date", "format": "strict_date_optional_time||epoch_millis"}
                        }
                    }
                })
        except Exception:
            # ignore race or permission issues; caller will handle errors
            pass

    def count(self) -> int:
        """Devuelve el total de documentos en el índice (o 0 si no existe)."""
        try:
            res = self.client.count(index=self.index)
            return int(res.get("count", 0))
        except NotFoundError:
            return 0
        except Exception:
            return 0

    def find_by_title_artist(self, title: str, artist: str, limit: int = 1) -> List[Dict[str, Any]]:
        """Busca por título y artista aproximados en el índice."""
        try:
            body = {
                "size": max(1, limit),
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"title": {"query": title, "operator": "and"}}},
                            {"match": {"artist": {"query": artist, "operator": "and"}}},
                        ]
                    }
                },
                "_source": [
                    "id", "title", "artist", "preview_url", "uri",
                    "valence", "energy", "mood", "image_url", "thumbnail_url"
                ],
            }
            res = self.client.search(index=self.index, body=body)
            hits = res.get("hits", {}).get("hits", [])
            return [h.get("_source", {}) for h in hits]
        except Exception:
            return []

    def delete_synthetic_seed(self) -> int:
        """Elimina documentos de la siembra sintética previa (prefijo 'seed-')."""
        try:
            body = {
                "query": {
                    "bool": {
                        "should": [
                            {"prefix": {"id": "seed-"}},
                            {"prefix": {"uri": "seed:track:"}}
                        ],
                        "minimum_should_match": 1
                    }
                }
            }
            res = self.client.delete_by_query(index=self.index, body=body, refresh=True, conflicts="proceed")
            return int(res.get("deleted", 0))
        except Exception:
            return 0

    def recreate_index(self) -> None:
        """Elimina el índice y lo crea de nuevo con el mapping actual."""
        try:
            if self.client.indices.exists(self.index):
                self.client.indices.delete(index=self.index)
        except Exception:
            pass
        self.ensure_index_exists()
