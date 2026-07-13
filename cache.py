import hashlib
import json
import os

CACHE_FILE = os.environ.get("CACHE_FILE", "/tmp/sieve_cache.json")

_cache: dict = {}


def _load():
    global _cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                _cache = json.load(f)
        except Exception:
            _cache = {}


def _save():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(_cache, f)
    except Exception:
        pass


_load()


def cache_key(task: dict) -> str:
    payload = json.dumps({"category": task.get("category"), "input": task.get("input")}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def get(task: dict):
    return _cache.get(cache_key(task))


def set(task: dict, result: dict):
    _cache[cache_key(task)] = result
    _save()
