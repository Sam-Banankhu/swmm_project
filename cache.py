import os
import hashlib
import pickle
from pathlib import Path
from functools import wraps
from config import config

class Cache:
    """Caching utility for expensive computations."""

    def __init__(self, cache_dir=None, backend='joblib'):
        self.cache_dir = Path(cache_dir or config.get('cache.dir', './cache'))
        self.backend = backend
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, func_name, args, kwargs):
        """Generate a unique cache key from function name and arguments."""
        # Create a hash of the function arguments
        args_str = str(args) + str(sorted(kwargs.items()))
        cache_key = hashlib.md5(f"{func_name}:{args_str}".encode()).hexdigest()
        return cache_key

    def _get_cache_path(self, cache_key):
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.pkl"

    def get(self, cache_key):
        """Retrieve cached result."""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                # If loading fails, remove corrupted cache
                cache_path.unlink(missing_ok=True)
        return None

    def set(self, cache_key, value):
        """Store result in cache."""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception:
            # If saving fails, remove partial cache
            cache_path.unlink(missing_ok=True)

    def clear(self):
        """Clear all cached results."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink(missing_ok=True)

def cached(cache_instance=None):
    """Decorator to cache function results."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not config.get('cache.enabled', True):
                return func(*args, **kwargs)

            cache = cache_instance or Cache()
            cache_key = cache._get_cache_key(func.__name__, args, kwargs)

            # Try to get cached result
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Compute result and cache it
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            return result

        return wrapper
    return decorator

# Global cache instance
cache = Cache()