"""Redis-based rate limiting middleware for Flask applications.

This module provides distributed rate limiting using the sliding window algorithm
with Redis for storage. It supports per-endpoint throttling with configurable limits
and returns proper 429 responses with rate limit headers.

Environment Variables:
    RATE_LIMIT_REQUESTS: Default requests per window (default: 100)
    RATE_LIMIT_WINDOW: Default window duration in seconds (default: 60)
    REDIS_URL: Redis connection URL (default: redis://localhost:6379/0)
"""

import os
import time
from typing import Optional, Dict, Any, Tuple, Callable
from functools import wraps

import redis
from flask import Flask, request, g, Response, jsonify
from pydantic import BaseModel, Field


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting.

    Attributes:
        requests_per_window: Maximum requests allowed within the window
        window_seconds: Time window duration in seconds
        redis_url: Redis connection URL
        enabled: Whether rate limiting is enabled
        endpoint_overrides: Per-endpoint configuration overrides
    """

    requests_per_window: int = Field(
        default=100,
        description="Maximum requests per window"
    )
    window_seconds: int = Field(
        default=60,
        description="Window duration in seconds"
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    enabled: bool = Field(
        default=True,
        description="Enable or disable rate limiting"
    )
    endpoint_overrides: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Per-endpoint rate limit overrides"
    )

    model_config = {
        "env_prefix": "RATE_LIMIT_",
        "extra": "ignore"
    }

    @classmethod
    def from_env(cls) -> "RateLimitConfig":
        """Create config from environment variables."""
        return cls(
            requests_per_window=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
            window_seconds=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            enabled=os.getenv("RATE_LIMIT_ENABLED", "True").lower() == "true"
        )


def get_redis_client(config: RateLimitConfig) -> Optional[redis.Redis]:
    """Create a Redis client for rate limiting.

    Args:
        config: Rate limit configuration containing Redis URL

    Returns:
        Redis client if connection succeeds, None otherwise
    """
    try:
        client = redis.from_url(config.redis_url, socket_connect_timeout=2)
        # Test connection
        client.ping()
        return client
    except (redis.ConnectionError, redis.TimeoutError):
        return None


class SlidingWindowLimiter:
    """Sliding window rate limiter using Redis sorted sets.

    This implementation uses Redis sorted sets to track request timestamps,
    providing a true sliding window algorithm rather than a fixed window.
    """

    def __init__(self, config: RateLimitConfig, redis_client: Optional[redis.Redis] = None):
        """Initialize the sliding window limiter.

        Args:
            config: Rate limit configuration
            redis_client: Optional Redis client (created from config if not provided)
        """
        self.config = config
        self._redis_client = redis_client
        self._in_fallback_mode = False

        # Try to establish Redis connection
        if self._redis_client is None:
            self._redis_client = get_redis_client(config)

        self.available = self._redis_client is not None

    def _get_redis(self) -> Optional[redis.Redis]:
        """Get Redis client, falling back to in-memory if unavailable."""
        if self.available:
            return self._redis_client
        return None

    def _get_key(self, client_id: str) -> str:
        """Generate Redis key for a client.

        Args:
            client_id: Unique identifier for the client

        Returns:
            Redis key string
        """
        return f"ratelimit:{client_id}"

    def check_rate_limit(self, client_id: str) -> Tuple[bool, int, int]:
        """Check if a request is allowed under the rate limit.

        Uses sliding window algorithm with Redis sorted sets:
        1. Remove entries older than (current_time - window_seconds)
        2. Count remaining entries
        3. If under limit, add current timestamp
        4. Return (allowed, remaining, reset_time)

        Args:
            client_id: Unique identifier for the client

        Returns:
            Tuple of (is_allowed, requests_remaining, reset_timestamp)
        """
        redis_client = self._get_redis()
        now = time.time()
        window_start = now - self.config.window_seconds
        key = self._get_key(client_id)

        if redis_client is None:
            # Fallback: simple in-memory counter (not truly distributed)
            return self._fallback_check(client_id, now)

        try:
            pipe = redis_client.pipeline()

            # Remove old entries outside the window
            pipe.zremrangebyscore(key, 0, window_start)

            # Count current entries in window
            pipe.zcard(key)

            # Execute both operations
            results = pipe.execute()
            current_count = results[1]

            if current_count < self.config.requests_per_window:
                # Add current request with timestamp as score
                pipe = redis_client.pipeline()
                pipe.zadd(key, {str(now): now})
                pipe.expire(key, self.config.window_seconds + 1)
                pipe.execute()

                remaining = self.config.requests_per_window - current_count - 1
                return (True, remaining, int(now + self.config.window_seconds))
            else:
                # Rate limited
                # Get the oldest timestamp in the window to calculate reset time
                oldest = redis_client.zrange(key, 0, 0, withscores=True)
                if oldest:
                    reset_time = int(oldest[0][1] + self.config.window_seconds)
                else:
                    reset_time = int(now + self.config.window_seconds)

                return (False, 0, reset_time)

        except (redis.ConnectionError, redis.TimeoutError) as e:
            # Fallback on Redis errors
            return self._fallback_check(client_id, now)

    def _fallback_check(self, client_id: str, now: float) -> Tuple[bool, int, int]:
        """Fallback rate limit check using in-memory storage.

        Not truly distributed but provides basic limiting when Redis is unavailable.
        """
        # Simple in-memory fallback using Flask app context
        if not hasattr(g, '_rate_limit_fallback'):
            g._rate_limit_fallback = {}

        key = f"{client_id}:{int(now // self.config.window_seconds)}"
        current = g._rate_limit_fallback.get(key, 0)

        if current < self.config.requests_per_window:
            g._rate_limit_fallback[key] = current + 1
            remaining = self.config.requests_per_window - current - 1
            return (True, remaining, int(now + self.config.window_seconds))
        else:
            return (False, 0, int(now + self.config.window_seconds))

    def get_current_count(self, client_id: str) -> int:
        """Get the current request count for a client.

        Args:
            client_id: Unique identifier for the client

        Returns:
            Current request count within the window
        """
        redis_client = self._get_redis()
        now = time.time()
        window_start = now - self.config.window_seconds
        key = self._get_key(client_id)

        if redis_client is None:
            return 0

        try:
            # Clean old entries and count
            redis_client.zremrangebyscore(key, 0, window_start)
            return redis_client.zcard(key)
        except (redis.ConnectionError, redis.TimeoutError):
            return 0


# Global rate limiter instance
rate_limiter: Optional[SlidingWindowLimiter] = None


def get_rate_limiter() -> SlidingWindowLimiter:
    """Get or create the global rate limiter instance."""
    global rate_limiter
    if rate_limiter is None:
        config = RateLimitConfig.from_env()
        rate_limiter = SlidingWindowLimiter(config)
    return rate_limiter


def rate_limit_middleware(
    endpoint_config: Optional[Dict[str, int]] = None,
    requests_per_window: Optional[int] = None,
    window_seconds: Optional[int] = None
) -> Callable:
    """Flask decorator for rate limiting endpoints.

    Args:
        endpoint_config: Optional dict with 'requests_per_window' and 'window_seconds'
        requests_per_window: Override for requests per window
        window_seconds: Override for window duration

    Returns:
        Decorator function
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args: Any, **kwargs: Any) -> Response:
            # Get rate limiter
            limiter = get_rate_limiter()

            if not limiter.available:
                # If rate limiter is unavailable, allow the request
                return f(*args, **kwargs)

            # Determine limits for this endpoint
            config = RateLimitConfig.from_env()

            # Check for endpoint-specific overrides
            if endpoint_config:
                limit_requests = endpoint_config.get('requests_per_window', config.requests_per_window)
                limit_window = endpoint_config.get('window_seconds', config.window_seconds)
            elif requests_per_window and window_seconds:
                limit_requests = requests_per_window
                limit_window = window_seconds
            else:
                limit_requests = config.requests_per_window
                limit_window = config.window_seconds

            # Get client identifier (use IP address or user ID)
            client_id = _get_client_identifier()

            # Create a temporary limiter with specific limits
            temp_config = RateLimitConfig(
                requests_per_window=limit_requests,
                window_seconds=limit_window,
                redis_url=config.redis_url
            )
            temp_limiter = SlidingWindowLimiter(temp_config, limiter._get_redis())

            # Check rate limit
            allowed, remaining, reset_time = temp_limiter.check_rate_limit(client_id)

            # Store rate limit info in g for the endpoint to access
            g.rate_limit_allowed = allowed
            g.rate_limit_remaining = remaining
            g.rate_limit_reset = reset_time
            g.rate_limit_limit = limit_requests

            # Add rate limit headers to response
            def add_headers(response: Response) -> Response:
                # Ensure response is a Flask Response object
                if not isinstance(response, Response):
                    from flask import make_response
                    response = make_response(response)
                response.headers['X-RateLimit-Limit'] = str(limit_requests)
                response.headers['X-RateLimit-Remaining'] = str(remaining)
                response.headers['X-RateLimit-Reset'] = str(reset_time)
                return response

            if not allowed:
                # Return 429 Too Many Requests
                response = jsonify({
                    'error': 'Rate limit exceeded',
                    'message': f'Too many requests. Please wait {reset_time - int(time.time())} seconds.',
                    'retry_after': reset_time - int(time.time())
                })
                response.status_code = 429
                response.headers['Retry-After'] = str(reset_time - int(time.time()))
                return add_headers(response)

            # Execute the wrapped function
            response = f(*args, **kwargs)

            # Add headers to successful response
            if isinstance(response, Response):
                return add_headers(response)
            else:
                # If response is a tuple or other format, wrap it
                return add_headers(response)

        # Store config for inspection
        decorated_function.__rate_limit_config__ = {
            'endpoint_config': endpoint_config,
            'requests_per_window': requests_per_window,
            'window_seconds': window_seconds
        }

        return decorated_function
    return decorator


def _get_client_id_from_request() -> str:
    """Extract client identifier from request.

    Uses X-Forwarded-For header if behind proxy, otherwise uses remote addr.
    """
    # Check for forwarded header (behind proxy)
    forwarded = request.headers.get('X-Forwarded-For')
    if forwarded:
        # Get the original client IP (first in the chain)
        return forwarded.split(',')[0].strip()

    # Check for real IP header
    real_ip = request.headers.get('X-Real-IP')
    if real_ip:
        return real_ip

    # Fall back to remote address
    return request.remote_addr or 'unknown'


def _get_client_identifier() -> str:
    """Get a unique client identifier for rate limiting.

    Combines IP address with user ID if authenticated.
    """
    # Check if we have a user ID (for authenticated requests)
    if hasattr(g, 'user_id') and g.user_id:
        return f"user:{g.user_id}"

    # Use IP address
    return f"ip:{_get_client_id_from_request()}"


def init_rate_limiter(config: Optional[RateLimitConfig] = None) -> SlidingWindowLimiter:
    """Initialize the global rate limiter.

    Args:
        Optional configuration. If not provided, loads from environment.

    Returns:
        Initialized rate limiter instance
    """
    global rate_limiter
    if config is None:
        config = RateLimitConfig.from_env()

    rate_limiter = SlidingWindowLimiter(config)
    return rate_limiter
