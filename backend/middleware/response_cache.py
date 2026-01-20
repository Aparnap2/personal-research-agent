"""Redis-based response caching middleware for Flask applications.

This module provides response caching using Redis for storage.
It supports TTL-based expiration, cache key generation, and hit/miss tracking.

Environment Variables:
    CACHE_TTL: Default cache TTL in seconds (default: 300)
    CACHE_MAX_SIZE_MB: Maximum cache size in MB (default: 100)
    CACHE_PREFIX: Key prefix for namespacing (default: cache:)
    REDIS_URL: Redis connection URL (default: redis://localhost:6379/0)
"""

import os
import json
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from functools import wraps

from flask import Flask, Request, Response, g, request

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for response caching.

    Attributes:
        ttl_seconds: Default TTL for cached responses in seconds
        max_size_mb: Maximum cache size in megabytes
        key_prefix: Key prefix for cache namespacing
        redis_url: Redis connection URL
        include_methods: HTTP methods to cache (default: GET)
        exclude_paths: Paths to exclude from caching
        enabled: Whether caching is enabled
    """
    ttl_seconds: int = field(default=300)
    max_size_mb: int = field(default=100)
    key_prefix: str = field(default="cache:")
    redis_url: str = field(default="redis://localhost:6379/0")
    include_methods: List[str] = field(default_factory=lambda: ["GET"])
    exclude_paths: List[str] = field(default_factory=lambda: ["/api/health", "/api/stream"])
    enabled: bool = field(default=True)

    def __post_init__(self):
        """Load config from environment if not set."""
        if "CACHE_TTL" in os.environ:
            self.ttl_seconds = int(os.environ["CACHE_TTL"])
        if "CACHE_MAX_SIZE_MB" in os.environ:
            self.max_size_mb = int(os.environ["CACHE_MAX_SIZE_MB"])
        if "CACHE_PREFIX" in os.environ:
            self.key_prefix = os.environ["CACHE_PREFIX"]
        if "REDIS_URL" in os.environ:
            self.redis_url = os.environ["REDIS_URL"]
        if "CACHE_ENABLED" in os.environ:
            self.enabled = os.environ["CACHE_ENABLED"].lower() == "true"


class CacheClientSingleton:
    """Singleton Redis client wrapper for caching."""

    _instance: Optional['CacheClientSingleton'] = None
    _client: Optional[Any] = None

    def __new__(cls) -> 'CacheClientSingleton':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_client(self, config: Optional[CacheConfig] = None) -> Optional[Any]:
        """Get or create Redis client.

        Args:
            config: Optional config to use for connection

        Returns:
            Redis client or None if unavailable
        """
        if self._client is not None:
            return self._client

        try:
            import redis
            from redis.connection import ConnectionPool

            url = config.redis_url if config else "redis://localhost:6379/0"
            pool = ConnectionPool.from_url(url, max_connections=10)
            self._client = redis.Redis(connection_pool=pool)
            # Test connection
            self._client.ping()
            logger.info(f"Cache Redis connected at {url}")
            return self._client
        except ImportError:
            logger.warning("redis-py not installed for caching")
            return None
        except Exception as e:
            logger.warning(f"Could not connect to Redis for caching: {e}")
            return None

    def close(self):
        """Close the Redis connection."""
        if self._client:
            self._client.close()
            self._client = None


# Global cache client singleton
_cache_client = CacheClientSingleton()


def get_cache_client(config: Optional[CacheConfig] = None) -> Optional[Any]:
    """Get Redis client for caching.

    Args:
        config: Optional config for connection

    Returns:
        Redis client or None if unavailable
    """
    return _cache_client.get_client(config)


class ResponseCache:
    """Redis-backed response cache.

    This cache stores API responses in Redis with TTL-based expiration.
    It's designed for caching GET request responses to improve performance.

    Attributes:
        config: Cache configuration
        client: Redis client instance
        available: Whether Redis is available
        _stats: Cache hit/miss statistics
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize response cache.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or CacheConfig()
        self.client = get_cache_client(self.config)
        self.available = self.client is not None
        self._stats = {"hits": 0, "misses": 0}

        if self.available:
            logger.info(f"Response cache initialized with prefix: {self.config.key_prefix}")
        else:
            logger.warning("Response cache initialized but Redis is not available")

    def _make_key(self, path: str, params: Dict[str, Any]) -> str:
        """Generate a cache key from path and parameters.

        Args:
            path: Request path
            params: Query parameters or request body

        Returns:
            Cache key string
        """
        # Sort params for consistent key generation
        if params:
            param_str = json.dumps(params, sort_keys=True)
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:16]
        else:
            param_hash = ""

        return f"{self.config.key_prefix}{path}:{param_hash}"

    def generate_key(self, path: str, **params) -> str:
        """Generate a cache key from path and keyword params.

        Args:
            path: Request path
            **params: Query parameters

        Returns:
            Cache key string
        """
        return self._make_key(path, params)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a cached response.

        Args:
            key: Cache key

        Returns:
            Cached response dict or None if not found
        """
        if not self.available:
            self._stats["misses"] += 1
            return None

        try:
            data = self.client.get(key)

            if data is None:
                self._stats["misses"] += 1
                return None

            self._stats["hits"] += 1
            return json.loads(data)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self._stats["misses"] += 1
            return None

    def set(self, key: str, response_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Cache a response.

        Args:
            key: Cache key
            response_data: Response data to cache
            ttl: Optional TTL override in seconds

        Returns:
            True if cached successfully
        """
        if not self.available:
            return False

        try:
            expire_time = ttl or self.config.ttl_seconds
            self.client.setex(key, expire_time, json.dumps(response_data))
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete a cached response.

        Args:
            key: Cache key

        Returns:
            True if deleted successfully
        """
        if not self.available:
            return False

        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    def clear(self) -> bool:
        """Clear all cached responses with our prefix.

        Returns:
            True if cleared successfully
        """
        if not self.available:
            return False

        try:
            pattern = f"{self.config.key_prefix}*"
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dict with hits and misses counts
        """
        return self._stats.copy()

    def invalidate_path(self, path: str) -> int:
        """Invalidate all cached responses for a path.

        Args:
            path: Request path to invalidate

        Returns:
            Number of keys deleted
        """
        if not self.available:
            return 0

        try:
            pattern = f"{self.config.key_prefix}{path}:*"
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache invalidate error: {e}")
            return 0


# Global cache instance
_cache: Optional[ResponseCache] = None


def get_cache() -> ResponseCache:
    """Get or create the global cache instance."""
    global _cache
    if _cache is None:
        _cache = ResponseCache()
    return _cache


def cache_middleware(
    ttl: Optional[int] = None,
    include_methods: Optional[List[str]] = None,
    exclude_paths: Optional[List[str]] = None,
    cache: Optional[ResponseCache] = None
) -> callable:
    """Flask decorator for caching GET responses.

    Args:
        ttl: Optional TTL override in seconds
        include_methods: HTTP methods to cache (default: GET)
        exclude_paths: Paths to exclude from caching
        cache: Optional cache instance

    Returns:
        Decorator function
    """
    def decorator(f: callable) -> callable:
        @wraps(f)
        def decorated_function(*args, Any, **kwargs) -> Response:
            # Get cache instance
            cache_instance = cache or get_cache()

            if not cache_instance.available or not cache_instance.config.enabled:
                return f(*args, **kwargs)

            # Check if we should cache this request
            if request.method not in (include_methods or cache_instance.config.include_methods):
                return f(*args, **kwargs)

            if any(request.path.startswith(p) for p in (exclude_paths or cache_instance.config.exclude_paths)):
                return f(*args, **kwargs)

            # Generate cache key from path and query params
            cache_key = cache_instance.generate_key(request.path, **request.args)

            # Try to get from cache
            cached = cache_instance.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache HIT for {cache_key}")
                return Response(
                    response=json.dumps(cached.get("data", {})),
                    status=cached.get("status", 200),
                    mimetype='application/json'
                )

            # Cache miss - execute the function
            logger.debug(f"Cache MISS for {cache_key}")
            response = f(*args, **kwargs)

            # Cache the response if it's successful
            if isinstance(response, Response) and response.status_code == 200:
                try:
                    response_data = {
                        "data": json.loads(response.get_data()),
                        "status": response.status_code,
                        "headers": dict(response.headers)
                    }
                    cache_instance.set(cache_key, response_data, ttl)
                except (json.JSONDecodeError, AttributeError):
                    pass  # Don't cache non-JSON responses

            return response

        return decorated_function
    return decorator


def init_cache(config: Optional[CacheConfig] = None) -> ResponseCache:
    """Initialize the global cache.

    Args:
        Optional configuration. If not provided, loads from environment.

    Returns:
        Initialized cache instance
    """
    global _cache
    _cache = ResponseCache(config)
    return _cache


if __name__ == "__main__":
    # Quick test
    config = CacheConfig()
    cache = ResponseCache(config)

    if cache.available:
        print("Response cache: Available")
        print(f"  TTL: {config.ttl_seconds}s")
        print(f"  Prefix: {config.key_prefix}")
    else:
        print("Response cache: Not available (Redis not running)")
