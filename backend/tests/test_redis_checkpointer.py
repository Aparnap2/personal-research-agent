"""Tests for Redis state persistence with LangGraph.

These tests verify that the Redis checkpointer implementation works correctly
for persisting agent state across sessions.
"""

import pytest
import os
import sys
from typing import Optional

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestRedisCheckpointerImport:
    """Test that Redis checkpointer module can be imported."""

    def test_redis_checkpointer_class_exists(self):
        """Test that RedisCheckpointer can be imported."""
        from agent.redis_checkpointer import RedisCheckpointer

        assert RedisCheckpointer is not None

    def test_redis_checkpointer_config_exists(self):
        """Test that RedisCheckpointerConfig can be imported."""
        from agent.redis_checkpointer import RedisCheckpointerConfig

        assert RedisCheckpointerConfig is not None

    def test_create_redis_checkpointer_exists(self):
        """Test that create_redis_checkpointer function exists."""
        from agent.redis_checkpointer import create_redis_checkpointer

        assert create_redis_checkpointer is not None

    def test_get_redis_client_exists(self):
        """Test that get_redis_client function exists."""
        from agent.redis_checkpointer import get_redis_client

        assert get_redis_client is not None


class TestRedisCheckpointerConfig:
    """Test Redis checkpointer configuration."""

    def test_config_defaults(self):
        """Test config has correct defaults."""
        from agent.redis_checkpointer import RedisCheckpointerConfig

        config = RedisCheckpointerConfig()

        assert config.redis_url == "redis://localhost:6379/0"
        assert config.prefix == "langgraph:"
        assert config.ttl_seconds == 86400  # 24 hours
        assert config.db == 0

    def test_config_custom_values(self):
        """Test config with custom values."""
        from agent.redis_checkpointer import RedisCheckpointerConfig

        config = RedisCheckpointerConfig(
            redis_url="redis://custom:6379/5",
            prefix="custom_prefix:",
            ttl_seconds=3600,
            db=5
        )

        assert config.redis_url == "redis://custom:6379/5"
        assert config.prefix == "custom_prefix:"
        assert config.ttl_seconds == 3600
        assert config.db == 5

    def test_config_from_env(self):
        """Test config loads from environment."""
        from agent.redis_checkpointer import RedisCheckpointerConfig

        # Save original env
        original_url = os.getenv("REDIS_URL")
        original_prefix = os.getenv("REDIS_PREFIX")

        try:
            os.environ["REDIS_URL"] = "redis://envtest:6379/2"
            os.environ["REDIS_PREFIX"] = "env_prefix:"

            config = RedisCheckpointerConfig()

            assert config.redis_url == "redis://envtest:6379/2"
            assert config.prefix == "env_prefix:"
        finally:
            # Restore original env
            if original_url:
                os.environ["REDIS_URL"] = original_url
            else:
                os.environ.pop("REDIS_URL", None)
            if original_prefix:
                os.environ["REDIS_PREFIX"] = original_prefix
            else:
                os.environ.pop("REDIS_PREFIX", None)


class TestRedisCheckpointerInstantiation:
    """Test Redis checkpointer instantiation."""

    def test_instantiate_with_defaults(self):
        """Test creating checkpointer with defaults."""
        from agent.redis_checkpointer import RedisCheckpointer, RedisCheckpointerConfig

        config = RedisCheckpointerConfig()
        checkpointer = RedisCheckpointer(config)

        assert checkpointer is not None
        assert checkpointer.config == config

    def test_instantiate_with_custom_config(self):
        """Test creating checkpointer with custom config."""
        from agent.redis_checkpointer import RedisCheckpointer, RedisCheckpointerConfig

        config = RedisCheckpointerConfig(
            redis_url="redis://localhost:6379/0",
            prefix="test:",
            ttl_seconds=1000
        )
        checkpointer = RedisCheckpointer(config)

        assert checkpointer is not None
        assert checkpointer.config.ttl_seconds == 1000


class TestRedisCheckpointerAvailability:
    """Test Redis availability checks."""

    def test_checkpointer_has_available_attribute(self):
        """Test that checkpointer has available attribute."""
        from agent.redis_checkpointer import RedisCheckpointer, RedisCheckpointerConfig

        config = RedisCheckpointerConfig()
        checkpointer = RedisCheckpointer(config)

        # available should be True if Redis is connected, False otherwise
        assert hasattr(checkpointer, 'available')
        assert isinstance(checkpointer.available, bool)


class TestRedisKeyGeneration:
    """Test Redis key generation."""

    def test_make_key(self):
        """Test key generation with prefix."""
        from agent.redis_checkpointer import RedisCheckpointer, RedisCheckpointerConfig

        config = RedisCheckpointerConfig(prefix="test_prefix:")
        checkpointer = RedisCheckpointer(config)

        key = checkpointer._make_key("graph123", "checkpoint456")

        assert key == "test_prefix:graph123:checkpoint456"

    def test_make_key_default_prefix(self):
        """Test key generation with default prefix."""
        from agent.redis_checkpointer import RedisCheckpointer, RedisCheckpointerConfig

        config = RedisCheckpointerConfig()
        checkpointer = RedisCheckpointer(config)

        key = checkpointer._make_key("graph_id", "checkpoint_id")

        assert key == "langgraph:graph_id:checkpoint_id"


class TestCreateRedisCheckpointerFunction:
    """Test the convenience function."""

    def test_create_with_defaults(self):
        """Test create function with defaults."""
        from agent.redis_checkpointer import create_redis_checkpointer, RedisCheckpointer

        checkpointer = create_redis_checkpointer()

        assert checkpointer is not None
        assert isinstance(checkpointer, RedisCheckpointer)

    def test_create_with_params(self):
        """Test create function with params."""
        from agent.redis_checkpointer import create_redis_checkpointer

        checkpointer = create_redis_checkpointer(
            redis_url="redis://localhost:6379/0",
            prefix="custom:",
            ttl_seconds=7200
        )

        assert checkpointer is not None
        assert checkpointer.config.ttl_seconds == 7200


class TestGetRedisClient:
    """Test get_redis_client function."""

    def test_returns_client_or_none(self):
        """Test get_redis_client returns client or None."""
        from agent.redis_checkpointer import get_redis_client, RedisCheckpointerConfig

        config = RedisCheckpointerConfig()
        client = get_redis_client(config)

        # Should return a client if available, None otherwise
        assert client is None or hasattr(client, 'ping')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
