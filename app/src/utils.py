"""Utilidades comunes (paginacion, reintentos, helpers numericos)."""

import random
import time
from typing import Callable, Tuple


def clamp_pagination(limit, offset, max_limit=100):
    """Normaliza `limit`/`offset` a enteros y aplica tope `max_limit`."""
    try:
        limit = int(limit)
    except Exception:
        limit = 20
    try:
        offset = int(offset)
    except Exception:
        offset = 0
    return max(1, min(limit, max_limit)), max(0, offset)


def backoff_retry(fn: Callable, max_tries: int = 3, base_delay: float = 0.5, jitter: float = 0.25):
    """Ejecuta `fn` con reintentos exponenciales simple.
    Lanza la ultima excepcion si falla.
    """
    last_exc = None
    for i in range(max_tries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if i == max_tries - 1:
                break
            sleep_s = base_delay * (2 ** i) + random.random() * jitter
            time.sleep(sleep_s)
    raise last_exc


def relax_range(bounds: Tuple[float, float], step: float, min_val: float = 0.0, max_val: float = 1.0):
    """Amplia el rango `bounds` simetricamente en `step` dentro de [min_val, max_val]."""
    lo, hi = bounds
    lo = max(min_val, lo - step)
    hi = min(max_val, hi + step)
    return lo, hi
