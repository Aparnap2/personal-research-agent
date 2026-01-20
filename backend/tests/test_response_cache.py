"""Tests for Redis-based response caching.

These tests verify that the response cache implementation works correctly
for caching API responses and improving performance.
"""

import pytest
import os
import sys
import json
from typing import Optional

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestResponseCacheImport:
    """Test that response cache module can be imported."""

    def test_response_cache_class_exists(self):
        """Test that ResponseCache can be imported."""
        from middleware.response_cache import ResponseCache

        assert ResponseCache is not None

    def test_response_cache_config_exists(self):
        """Test that CacheConfig can be imported."""
        from middleware.response_cache import CacheConfig

        assert CacheConfig is not None

    def test_cache_middleware_exists(self):
        """Test that cache_middleware can be imported."""
        from middleware.response_cache import cache_middleware

        assert cache_middleware is not None


class TestCacheConfig:
    """Test cache configuration."""

    def test_config_defaults(self):
        """Test config has correct defaults."""
        from middleware.response_cache import CacheConfig

        config = CacheConfig()

        assert config.ttl_seconds == 300  # 5 minutes
        assert config.max_size_mb == 100
        assert config.key_prefix == "cache:"

    def test_config_custom_values(self):
        """Test config with custom values."""
        from middleware.response_cache import CacheConfig

        config = CacheConfig(
            ttl_seconds=600,
            max_size_mb=50,
            key_prefix="custom:"
        )

        assert config.ttl_seconds == 600
        assert config.max_size_mb == 50
        assert config.key_prefix == "custom:"

    def test_config_from_env(self):
        """Test config loads from environment."""
        from middleware.response_cache import CacheConfig

        # Save original env
        original_ttl = os.getenv("CACHE_TTL")
        original_prefix = os.getenv("CACHE_PREFIX")

        try:
            os.environ["CACHE_TTL"] = "600"
            os.environ["CACHE_PREFIX"] = "test:"

            config = CacheConfig()

            assert config.ttl_seconds == 600
            assert config.key_prefix == "test:"
        finally:
            # Restore original env
            if original_ttl:
                os.environ["CACHE_TTL"] = original_ttl
            else:
                os.environ.pop("CACHE_TTL", None)
            if original_prefix:
                os.environ["CACHE_PREFIX"] = original_prefix
            else:
                os.environ.pop("CACHE_PREFIX", None)


class TestResponseCacheInstantiation:
    """Test response cache instantiation."""

    def test_instantiate_with_defaults(self):
        """Test creating cache with defaults."""
        from middleware.response_cache import ResponseCache, CacheConfig

        config = CacheConfig()
        cache = ResponseCache(config)

        assert cache is not None
        assert cache.config == config

    def test_instantiate_with_custom_config(self):
        """Test creating cache with custom config."""
        from middleware.response_cache import ResponseCache, CacheConfig

        config = CacheConfig(ttl_seconds=120, key_prefix="myapp:")
        cache = ResponseCache(config)

        assert cache is not None
        assert cache.config.ttl_seconds == 120


class TestResponseCacheAvailability:
    """Test cache availability checks."""

    def test_cache_has_available_attribute(self):
        """Test that cache has available attribute."""
        from middleware.response_cache import ResponseCache, CacheConfig

        config = CacheConfig()
        cache = ResponseCache(config)

        assert hasattr(cache, 'available')
        assert isinstance(cache.available, bool)


class TestResponseCacheKeyGeneration:
    """Test cache key generation."""

    def test_make_key(self):
        """Test key generation with prefix."""
        from middleware.response_cache import ResponseCache, CacheConfig

        config = CacheConfig(key_prefix="test_prefix:")
        cache = ResponseCache(config)

        key = cache._make_key("/api/research", {"q": "test"})

        assert key.startswith("test_prefix:")
        assert "/api/research" in key

    def test_make_key_default_prefix(self):
        """Test key generation with default prefix."""
        from middleware.response_cache import ResponseCache, CacheConfig

        config = CacheConfig()
        cache = ResponseCache(config)

        key = cache._make_key("/api/chat", {"id": "123"})

        # Key should start with prefix and path
        assert key.startswith("cache:/api/chat:")
        # The rest should be the hash of params
        assert len(key.split(":")[-1]) == 16  # MD5 hex

    def test_make_key_with_short_params(self):
        """Test key generation with short params."""
        from middleware.response_cache import ResponseCache, CacheConfig

        config = CacheConfig()
        cache = ResponseCache(config)

        # Short params should be included directly
        key = cache._make_key("/api/chat", {"id": "123"})

        assert key.startswith("cache:/api/chat:")


class TestResponseCacheOperations:
    """Test cache operations."""

    def test_get_returns_none_for_missing(self):
        """Test that get returns None for missing key."""
        from middleware.response_cache import ResponseCache, CacheConfig

        config = CacheConfig()
        cache = ResponseCache(config)

        result = cache.get("nonexistent_key")
        assert result is None

    def test_set_and_get(self):
        """Test that set and get work."""
        from middleware.response_cache import ResponseCache, CacheConfig

        config = CacheConfig()
        cache = ResponseCache(config)

        test_data = {"result": "test_value", "count": 42}
        cache.set("test_key", test_data, ttl=60)

        result = cache.get("test_key")
        assert result == test_data

    def test_set_with_ttl(self):
        """Test that TTL is applied."""
        from middleware.response_cache import ResponseCache, CacheConfig

        config = CacheConfig(ttl_seconds=1)  # 1 second TTL
        cache = ResponseCache(config)

        test_data = {"value": "temporary"}
        cache.set("temp_key", test_data, ttl=1)

        # Should be present immediately
        assert cache.get("temp_key") == test_data

    def test_delete(self):
        """Test that delete works."""
        from middleware.response_cache import ResponseCache, CacheConfig

        config = CacheConfig()
        cache = ResponseCache(config)

        test_data = {"to_delete": "value"}
        cache.set("delete_me", test_data)
        assert cache.get("delete_me") == test_data

        cache.delete("delete_me")
        assert cache.get("delete_me") is None

    def test_clear(self):
        """Test that clear works."""
        from middleware.response_cache import ResponseCache, CacheConfig
        import uuid

        config = CacheConfig()
        cache = ResponseCache(config)

        # Use unique keys with proper prefix
        prefix = config.key_prefix
        key1 = f"{prefix}clear_test_1_{uuid.uuid4().hex[:8]}"
        key2 = f"{prefix}clear_test_2_{uuid.uuid4().hex[:8]}"

        cache.set(key1, {"data": 1})
        cache.set(key2, {"data": 2})

        # Verify keys exist
        assert cache.get(key1) is not None
        assert cache.get(key2) is not None

        # Clear all
        cache.clear()

        # Both should be gone
        assert cache.get(key1) is None
        assert cache.get(key2) is None


class TestResponseCacheGenerateKey:
    """Test key generation from request components."""

    def test_generate_key_from_params(self):
        """Test key generation from path and params."""
        from middleware.response_cache import ResponseCache, CacheConfig

        config = CacheConfig()
        cache = ResponseCache(config)

        key = cache.generate_key("/api/research", q="python", limit=10)

        # Key should start with prefix and path
        assert key.startswith("cache:/api/research:")
        # Key should exist
        assert len(key) > len("cache:/api/research:")

    def test_generate_key_sorted_params(self):
        """Test that params are sorted for consistent keys."""
        from middleware.response_cache import ResponseCache, CacheConfig

        config = CacheConfig()
        cache = ResponseCache(config)

        key1 = cache.generate_key("/api/research", a="1", b="2")
        key2 = cache.generate_key("/api/research", b="2", a="1")

        assert key1 == key2


class TestCacheMiddleware:
    """Test Flask middleware integration."""

    def test_middleware_returns_callable(self):
        """Test that middleware decorator returns callable."""
        from middleware.response_cache import cache_middleware

        decorator = cache_middleware()
        assert callable(decorator)

    def test_middleware_with_params(self):
        """Test middleware with custom params."""
        from middleware.response_cache import cache_middleware

        decorator = cache_middleware(ttl=300, exclude_paths=["/api/health"])
        assert callable(decorator)

    def test_middleware_with_methods(self):
        """Test middleware with allowed methods."""
        from middleware.response_cache import cache_middleware

        decorator = cache_middleware(include_methods=["GET", "POST"])
        assert callable(decorator)


class TestCacheHitMiss:
    """Test cache hit/miss tracking."""

    def test_cache_stats_initial(self):
        """Test that cache stats start at zero."""
        from middleware.response_cache import ResponseCache, CacheConfig

        config = CacheConfig()
        cache = ResponseCache(config)

        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_cache_stats_on_hit(self):
        """Test that hit count increases on cache hit."""
        from middleware.response_cache import ResponseCache, CacheConfig

        config = CacheConfig()
        cache = ResponseCache(config)

        cache.set("test_key", {"data": "value"})
        cache.get("test_key")

        stats = cache.get_stats()
        assert stats["hits"] == 1

    def test_cache_stats_on_miss(self):
        """Test that miss count increases on cache miss."""
        from middleware.response_cache import ResponseCache, CacheConfig

        config = CacheConfig()
        cache = ResponseCache(config)

        cache.get("nonexistent")

        stats = cache.get_stats()
        assert stats["misses"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
