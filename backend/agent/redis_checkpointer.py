"""Redis-based state persistence for LangGraph agents.

This module provides a Redis-backed checkpointer for persisting agent state
across sessions, enabling resume functionality and state recovery.

Usage:
    from agent.redis_checkpointer import create_redis_checkpointer

    checkpointer = create_redis_checkpointer()
    # Use with StateGraph
    graph.compile(checkpointer=checkpointer)
"""

import os
import json
import logging
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, field

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint

logger = logging.getLogger(__name__)


@dataclass
class RedisCheckpointerConfig:
    """Configuration for Redis checkpointer.

    Attributes:
        redis_url: Redis connection URL (default: redis://localhost:6379/0)
        prefix: Key prefix for namespacing (default: langgraph:)
        ttl_seconds: Time-to-live for checkpoints in seconds (default: 86400 = 24h)
        db: Redis database number (default: 0)
        max_connections: Maximum connections in pool (default: 10)
    """
    redis_url: str = field(default="redis://localhost:6379/0")
    prefix: str = field(default="langgraph:")
    ttl_seconds: int = field(default=86400)  # 24 hours
    db: int = field(default=0)
    max_connections: int = field(default=10)

    def __post_init__(self):
        """Load config from environment if not set."""
        if "REDIS_URL" in os.environ:
            self.redis_url = os.environ["REDIS_URL"]
        if "REDIS_PREFIX" in os.environ:
            self.prefix = os.environ["REDIS_PREFIX"]
        if "REDIS_TTL" in os.environ:
            self.ttl_seconds = int(os.environ["REDIS_TTL"])
        if "REDIS_DB" in os.environ:
            self.db = int(os.environ["REDIS_DB"])


class RedisClientSingleton:
    """Singleton Redis client wrapper."""

    _instance: Optional['RedisClientSingleton'] = None
    _client: Optional[Any] = None

    def __new__(cls) -> 'RedisClientSingleton':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_client(self, config: Optional[RedisCheckpointerConfig] = None) -> Optional[Any]:
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
            logger.info(f"Connected to Redis at {url}")
            return self._client
        except ImportError:
            logger.warning("redis-py not installed. Install with: pip install redis")
            return None
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}")
            return None

    def close(self):
        """Close the Redis connection."""
        if self._client:
            self._client.close()
            self._client = None


# Global client singleton
_redis_client = RedisClientSingleton()


def get_redis_client(config: Optional[RedisCheckpointerConfig] = None) -> Optional[Any]:
    """Get Redis client singleton.

    Args:
        config: Optional config for connection

    Returns:
        Redis client or None if unavailable
    """
    return _redis_client.get_client(config)


class RedisCheckpointer(BaseCheckpointSaver):
    """Redis-backed checkpoint saver for LangGraph.

    This checkpointer stores agent state in Redis, enabling:
    - Persistent state across sessions
    - Resumable research workflows
    - Distributed agent coordination
    - Automatic state expiration via TTL

    Attributes:
        config: Checkpointer configuration
        client: Redis client instance
        available: Whether Redis is available
    """

    def __init__(self, config: Optional[RedisCheckpointerConfig] = None):
        """Initialize Redis checkpointer.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or RedisCheckpointerConfig()
        self.client = get_redis_client(self.config)
        self.available = self.client is not None

        if self.available:
            logger.info(f"Redis checkpointer initialized with prefix: {self.config.prefix}")
        else:
            logger.warning("Redis checkpointer initialized but Redis is not available")

    def _make_key(self, graph_id: str, checkpoint_id: str) -> str:
        """Create a namespaced key for the checkpoint.

        Args:
            graph_id: The graph/thread ID
            checkpoint_id: The checkpoint ID

        Returns:
            Full Redis key with prefix
        """
        return f"{self.config.prefix}{graph_id}:{checkpoint_id}"

    def get(self, graph_id: str, checkpoint_id: str) -> Optional[Checkpoint]:
        """Get a checkpoint from Redis.

        Args:
            graph_id: The graph/thread ID
            checkpoint_id: The checkpoint ID

        Returns:
            Checkpoint or None if not found
        """
        if not self.available:
            return None

        try:
            key = self._make_key(graph_id, checkpoint_id)
            data = self.client.get(key)

            if data is None:
                return None

            # Deserialize the checkpoint
            checkpoint_data = json.loads(data)
            return Checkpoint(**checkpoint_data)
        except Exception as e:
            logger.error(f"Error getting checkpoint {checkpoint_id}: {e}")
            return None

    def put(
        self,
        graph_id: str,
        checkpoint_id: str,
        checkpoint: Checkpoint,
        metadata: Dict[str, Any]
    ) -> Checkpoint:
        """Store a checkpoint in Redis.

        Args:
            graph_id: The graph/thread ID
            checkpoint_id: The checkpoint ID
            checkpoint: The checkpoint to store
            metadata: Additional metadata

        Returns:
            The checkpoint that was stored
        """
        if not self.available:
            return checkpoint

        try:
            key = self._make_key(graph_id, checkpoint_id)

            # Prepare data for serialization
            checkpoint_dict = {
                "v": checkpoint.v,
                "id": checkpoint.id,
                "ts": checkpoint.ts,
                "pending_writes": checkpoint.pending_writes,
                "tasks": checkpoint.tasks,
                "state": checkpoint.state,
            }

            # Serialize and store with TTL
            self.client.setex(
                key,
                self.config.ttl_seconds,
                json.dumps(checkpoint_dict)
            )

            logger.debug(f"Stored checkpoint {checkpoint_id} with TTL {self.config.ttl_seconds}s")

            return checkpoint
        except Exception as e:
            logger.error(f"Error storing checkpoint {checkpoint_id}: {e}")
            return checkpoint

    def delete(self, graph_id: str, checkpoint_id: str) -> bool:
        """Delete a checkpoint from Redis.

        Args:
            graph_id: The graph/thread ID
            checkpoint_id: The checkpoint ID

        Returns:
            True if deleted, False otherwise
        """
        if not self.available:
            return False

        try:
            key = self._make_key(graph_id, checkpoint_id)
            result = self.client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting checkpoint {checkpoint_id}: {e}")
            return False

    def list(
        self,
        graph_id: str,
        *,
        limit: int = 10,
        before: Optional[str] = None,
        after: Optional[str] = None
    ) -> Tuple[str, ...]:
        """List checkpoint IDs for a graph.

        Args:
            graph_id: The graph/thread ID
            limit: Maximum number to return
            before: Only return checkpoints before this ID
            after: Only return checkpoints after this ID

        Returns:
            Tuple of checkpoint IDs
        """
        if not self.available:
            return ()

        try:
            pattern = f"{self.config.prefix}{graph_id}:*"
            keys = self.client.keys(pattern)

            # Extract checkpoint IDs and sort
            checkpoint_ids = []
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                # Remove prefix and graph_id to get checkpoint_id
                checkpoint_id = key_str.replace(f"{self.config.prefix}{graph_id}:", "")
                if checkpoint_id:
                    checkpoint_ids.append(checkpoint_id)

            # Sort by timestamp (most recent first)
            checkpoint_ids.sort(reverse=True)

            # Apply limit
            return tuple(checkpoint_ids[:limit])
        except Exception as e:
            logger.error(f"Error listing checkpoints for {graph_id}: {e}")
            return ()

    def get_next_checkpoint_id(self, graph_id: str) -> Optional[str]:
        """Get the most recent checkpoint ID for a graph.

        Args:
            graph_id: The graph/thread ID

        Returns:
            Most recent checkpoint ID or None
        """
        checkpoints = self.list(graph_id, limit=1)
        return checkpoints[0] if checkpoints else None


def create_redis_checkpointer(
    redis_url: Optional[str] = None,
    prefix: Optional[str] = None,
    ttl_seconds: Optional[int] = None
) -> RedisCheckpointer:
    """Create a Redis checkpointer with the given configuration.

    This is a convenience function for creating a checkpointer.

    Args:
        redis_url: Optional Redis URL
        prefix: Optional key prefix
        ttl_seconds: Optional TTL in seconds

    Returns:
        RedisCheckpointer instance
    """
    config = RedisCheckpointerConfig(
        redis_url=redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
        prefix=prefix or os.environ.get("REDIS_PREFIX", "langgraph:"),
        ttl_seconds=ttl_seconds or int(os.environ.get("REDIS_TTL", "86400"))
    )
    return RedisCheckpointer(config)


# For backward compatibility
Checkpointer = RedisCheckpointer


if __name__ == "__main__":
    # Quick test
    config = RedisCheckpointerConfig()
    checkpointer = RedisCheckpointer(config)

    if checkpointer.available:
        print("Redis checkpointer: Available")
        print(f"  URL: {config.redis_url}")
        print(f"  Prefix: {config.prefix}")
        print(f"  TTL: {config.ttl_seconds}s")
    else:
        print("Redis checkpointer: Not available (Redis not running)")
