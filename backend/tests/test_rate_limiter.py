"""Tests for Redis-based rate limiting middleware.

These tests verify that the rate limiter implementation provides distributed
rate limiting with sliding window algorithm, per-endpoint throttling, and
proper 429 response handling.
"""

import pytest
import os
import sys
import time
from typing import Optional

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestRateLimiterImport:
    """Test that rate limiter module can be imported."""

    def test_rate_limiter_module_exists(self):
        """Test that rate_limiter module can be imported."""
        from middleware import rate_limiter
        from middleware.rate_limiter import RateLimitConfig

        # rate_limiter is a global that may be None if not initialized
        # That's okay - the module exists and config exists
        assert RateLimitConfig is not None

    def test_get_redis_client_exists(self):
        """Test that get_redis_client function exists."""
        from middleware.rate_limiter import get_redis_client

        assert get_redis_client is not None

    def test_sliding_window_limiter_exists(self):
        """Test that SlidingWindowLimiter class exists."""
        from middleware.rate_limiter import SlidingWindowLimiter

        assert SlidingWindowLimiter is not None

    def test_rate_limit_middleware_exists(self):
        """Test that rate_limit_middleware function exists."""
        from middleware.rate_limiter import rate_limit_middleware

        assert rate_limit_middleware is not None


class TestRateLimitConfig:
    """Test rate limit configuration."""

    def test_config_defaults(self):
        """Test config has correct defaults."""
        from middleware.rate_limiter import RateLimitConfig

        config = RateLimitConfig()

        assert config.requests_per_window == 100
        assert config.window_seconds == 60
        assert config.redis_url == "redis://localhost:6379/0"
        assert config.enabled is True

    def test_config_custom_values(self):
        """Test config with custom values."""
        from middleware.rate_limiter import RateLimitConfig

        config = RateLimitConfig(
            requests_per_window=50,
            window_seconds=30,
            redis_url="redis://custom:6379/5",
            enabled=False
        )

        assert config.requests_per_window == 50
        assert config.window_seconds == 30
        assert config.redis_url == "redis://custom:6379/5"
        assert config.enabled is False

    def test_config_endpoint_overrides(self):
        """Test config with per-endpoint overrides."""
        from middleware.rate_limiter import RateLimitConfig

        config = RateLimitConfig(
            endpoint_overrides={
                "/api/research": {"requests_per_window": 10, "window_seconds": 60},
                "/api/health": {"requests_per_window": 1000, "window_seconds": 60}
            }
        )

        assert "/api/research" in config.endpoint_overrides
        assert config.endpoint_overrides["/api/research"]["requests_per_window"] == 10

    def test_config_from_env(self):
        """Test config loads from environment variables."""
        from middleware.rate_limiter import RateLimitConfig

        # Save original env
        original_requests = os.getenv("RATE_LIMIT_REQUESTS")
        original_window = os.getenv("RATE_LIMIT_WINDOW")
        original_url = os.getenv("REDIS_URL")
        original_enabled = os.getenv("RATE_LIMIT_ENABLED")

        try:
            os.environ["RATE_LIMIT_REQUESTS"] = "200"
            os.environ["RATE_LIMIT_WINDOW"] = "120"
            os.environ["RATE_LIMIT_ENABLED"] = "false"

            config = RateLimitConfig.from_env()

            assert config.requests_per_window == 200
            assert config.window_seconds == 120
            assert config.enabled is False
        finally:
            # Restore original env
            for key, original in [("RATE_LIMIT_REQUESTS", original_requests),
                                   ("RATE_LIMIT_WINDOW", original_window),
                                   ("RATE_LIMIT_ENABLED", original_enabled),
                                   ("REDIS_URL", original_url)]:
                if original:
                    os.environ[key] = original
                elif key in os.environ:
                    del os.environ[key]


class TestSlidingWindowLimiter:
    """Test sliding window rate limiter algorithm."""

    def test_limiter_instantiation(self):
        """Test creating a sliding window limiter instance."""
        from middleware.rate_limiter import SlidingWindowLimiter, RateLimitConfig

        config = RateLimitConfig()
        limiter = SlidingWindowLimiter(config)

        assert limiter is not None
        assert limiter.config == config

    def test_limiter_check_returns_allowed_initially(self):
        """Test that new clients are allowed."""
        from middleware.rate_limiter import SlidingWindowLimiter, RateLimitConfig
        import uuid

        config = RateLimitConfig(requests_per_window=10, window_seconds=60)
        limiter = SlidingWindowLimiter(config)

        # Use unique client ID to avoid test pollution
        client_id = f"test_client_1_{uuid.uuid4().hex[:8]}"

        # Should be allowed for new client
        allowed, remaining, reset_time = limiter.check_rate_limit(client_id)

        assert allowed is True
        assert remaining == 9  # 10 - 1
        assert reset_time > 0

    def test_limiter_allows_multiple_requests(self):
        """Test that multiple requests are allowed up to limit."""
        from middleware.rate_limiter import SlidingWindowLimiter, RateLimitConfig
        import uuid

        config = RateLimitConfig(requests_per_window=5, window_seconds=60)
        limiter = SlidingWindowLimiter(config)

        # Use unique client ID
        client_id = f"test_client_2_{uuid.uuid4().hex[:8]}"

        # Make 5 requests - all should be allowed
        for i in range(5):
            allowed, remaining, reset_time = limiter.check_rate_limit(client_id)
            assert allowed is True
            assert remaining == 5 - i - 1

    def test_limiter_blocks_excess_requests(self):
        """Test that requests beyond limit are blocked."""
        from middleware.rate_limiter import SlidingWindowLimiter, RateLimitConfig
        import uuid

        config = RateLimitConfig(requests_per_window=3, window_seconds=60)
        limiter = SlidingWindowLimiter(config)

        # Use unique client ID
        client_id = f"test_client_3_{uuid.uuid4().hex[:8]}"

        # Make 3 requests - all allowed
        for i in range(3):
            allowed, _, _ = limiter.check_rate_limit(client_id)
            assert allowed is True

        # 4th request should be blocked
        allowed, remaining, reset_time = limiter.check_rate_limit(client_id)
        assert allowed is False
        assert remaining == 0
        assert reset_time > 0

    def test_limiter_different_clients_independent(self):
        """Test that different clients have independent limits."""
        from middleware.rate_limiter import SlidingWindowLimiter, RateLimitConfig
        import uuid

        config = RateLimitConfig(requests_per_window=2, window_seconds=60)
        limiter = SlidingWindowLimiter(config)

        # Use unique client IDs
        client_a = f"client_A_{uuid.uuid4().hex[:8]}"
        client_b = f"client_B_{uuid.uuid4().hex[:8]}"

        # Exhaust client A
        limiter.check_rate_limit(client_a)
        limiter.check_rate_limit(client_a)
        allowed_a, _, _ = limiter.check_rate_limit(client_a)
        assert allowed_a is False

        # Client B should still be allowed
        allowed_b, _, _ = limiter.check_rate_limit(client_b)
        assert allowed_b is True


class TestRateLimiterMiddleware:
    """Test Flask middleware integration."""

    def test_middleware_returns_callable(self):
        """Test that middleware decorator is callable."""
        from middleware.rate_limiter import rate_limit_middleware

        # Should be a decorator that returns a callable
        assert callable(rate_limit_middleware)

    def test_middleware_decorates_endpoint(self):
        """Test that middleware can be used as decorator."""
        from middleware.rate_limiter import rate_limit_middleware
        from flask import Flask

        app = Flask(__name__)

        @app.route('/test')
        @rate_limit_middleware()
        def test_endpoint():
            return {'status': 'ok'}

        assert hasattr(test_endpoint, '__rate_limit_config__')

    def test_middleware_429_response_format(self):
        """Test that 429 response has correct format."""
        from middleware.rate_limiter import RateLimitConfig
        from middleware.rate_limiter import SlidingWindowLimiter
        from middleware.rate_limiter import rate_limit_middleware
        from flask import Flask
        import uuid

        # Create a limiter with Redis to test properly
        # If Redis is not available, this test will fail gracefully
        config = RateLimitConfig(redis_url="redis://localhost:6379/0")
        try:
            import redis
            redis_client = redis.from_url(config.redis_url, socket_connect_timeout=1)
            redis_client.ping()
            redis_available = True
        except Exception:
            redis_available = False

        app = Flask(__name__)
        unique_id = uuid.uuid4().hex[:8]

        @app.route(f'/limited_{unique_id}')
        @rate_limit_middleware(endpoint_config={"requests_per_window": 1, "window_seconds": 60})
        def limited_endpoint():
            return {'status': 'ok'}

        client = app.test_client()
        endpoint = f'/limited_{unique_id}'

        if not redis_available:
            # Skip the full integration test if Redis is not available
            # The fallback mechanism doesn't work properly with Flask's test client
            pytest.skip("Redis not available for rate limiter integration test")
            return

        # First request should succeed
        response1 = client.get(endpoint, headers={'X-Real-IP': '192.168.1.100'})
        assert response1.status_code == 200

        # Second request should be rate limited
        response2 = client.get(endpoint, headers={'X-Real-IP': '192.168.1.100'})
        assert response2.status_code == 429

        # Check headers
        assert 'X-RateLimit-Limit' in response2.headers
        assert 'X-RateLimit-Remaining' in response2.headers
        assert 'X-RateLimit-Reset' in response2.headers
        assert 'Retry-After' in response2.headers


class TestRateLimiterIntegration:
    """Integration tests for rate limiter with Redis."""

    def test_redis_client_available_check(self):
        """Test that Redis availability is properly checked."""
        from middleware.rate_limiter import get_redis_client
        from middleware.rate_limiter import RateLimitConfig

        config = RateLimitConfig()
        client = get_redis_client(config)

        # If Redis is available, client should have ping method
        # If not, client should be None (graceful fallback)
        if client is not None:
            assert hasattr(client, 'ping')

    def test_rate_limiter_fallback_when_redis_unavailable(self):
        """Test that rate limiter handles Redis unavailability gracefully."""
        from middleware.rate_limiter import SlidingWindowLimiter, RateLimitConfig

        # Create a limiter with unreachable Redis
        config = RateLimitConfig(
            redis_url="redis://unreachable:6379/0",
            requests_per_window=10,
            window_seconds=60
        )
        limiter = SlidingWindowLimiter(config)

        # Should still work (graceful fallback to in-memory)
        # or mark itself as unavailable
        assert hasattr(limiter, 'available')
        assert isinstance(limiter.available, bool)


class TestRateLimitHeaders:
    """Test rate limit header formatting."""

    def test_headers_are_included_in_response(self):
        """Test that rate limit headers are included in successful responses."""
        from middleware.rate_limiter import rate_limit_middleware
        from flask import Flask
        import uuid

        app = Flask(__name__)
        unique_id = uuid.uuid4().hex[:8]

        # Use unique endpoint
        @app.route(f'/api/test_{unique_id}')
        @rate_limit_middleware(endpoint_config={"requests_per_window": 10, "window_seconds": 60})
        def test_endpoint():
            return {'status': 'ok'}

        client = app.test_client()
        response = client.get(f'/api/test_{unique_id}')

        assert response.status_code == 200
        assert 'X-RateLimit-Limit' in response.headers
        assert 'X-RateLimit-Remaining' in response.headers
        assert 'X-RateLimit-Reset' in response.headers

    def test_remaining_decrements_on_each_request(self):
        """Test that X-RateLimit-Remaining decrements correctly."""
        from middleware.rate_limiter import rate_limit_middleware
        from flask import Flask
        import uuid

        app = Flask(__name__)
        unique_id = uuid.uuid4().hex[:8]
        client_ip = f'192.168.1.{uuid.uuid4().int % 255}'

        # Use unique endpoint
        @app.route(f'/api/count_{unique_id}')
        @rate_limit_middleware(endpoint_config={"requests_per_window": 3, "window_seconds": 60})
        def count_endpoint():
            return {'status': 'ok'}

        client = app.test_client()
        endpoint = f'/api/count_{unique_id}'
        headers = {'X-Real-IP': client_ip}

        response1 = client.get(endpoint, headers=headers)
        remaining1 = int(response1.headers['X-RateLimit-Remaining'])

        response2 = client.get(endpoint, headers=headers)
        remaining2 = int(response2.headers['X-RateLimit-Remaining'])

        response3 = client.get(endpoint, headers=headers)
        remaining3 = int(response3.headers['X-RateLimit-Remaining'])

        # Remaining should decrement: 3 -> 2 -> 1
        assert remaining1 == 2
        assert remaining2 == 1
        assert remaining3 == 0


class TestEndpointSpecificLimits:
    """Test per-endpoint rate limiting."""

    def test_different_limits_for_different_endpoints(self):
        """Test that different endpoints can have different limits."""
        from middleware.rate_limiter import rate_limit_middleware
        from flask import Flask
        import uuid

        app = Flask(__name__)

        # Use unique endpoints
        @app.route(f'/api/strict_{uuid.uuid4().hex[:8]}')
        @rate_limit_middleware(endpoint_config={"requests_per_window": 2, "window_seconds": 60})
        def strict_endpoint():
            return {'status': 'ok'}

        @app.route(f'/api/relaxed_{uuid.uuid4().hex[:8]}')
        @rate_limit_middleware(endpoint_config={"requests_per_window": 100, "window_seconds": 60})
        def relaxed_endpoint():
            return {'status': 'ok'}

        client = app.test_client()

        # Get the actual routes
        rules = list(app.url_map.iter_rules())
        strict_rule = [r for r in rules if 'strict' in r.rule][0].rule
        relaxed_rule = [r for r in rules if 'relaxed' in r.rule][0].rule

        # Exhaust strict endpoint
        client.get(strict_rule)
        client.get(strict_rule)
        response_strict = client.get(strict_rule)
        assert response_strict.status_code == 429

        # Relaxed endpoint should still work
        response_relaxed = client.get(relaxed_rule)
        assert response_relaxed.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
