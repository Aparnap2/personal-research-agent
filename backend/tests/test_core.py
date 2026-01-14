"""Tests for the Research Intelligence Platform."""

import pytest
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestSyntheticData:
    """Tests for synthetic data generation."""

    def test_generate_synthetic_quantitative_data(self):
        """Test synthetic data generation."""
        from utils.synthetic_data import generate_synthetic_quantitative_data

        data = generate_synthetic_quantitative_data(count=5, include_time_series=False, include_comparison=False)

        assert isinstance(data, list)
        assert len(data) == 5
        assert all(isinstance(item, dict) for item in data)
        assert all("metric_name" in item for item in data)
        assert all("value" in item for item in data)

    def test_generate_synthetic_quantitative_data_with_time_series(self):
        """Test synthetic data with time series."""
        from utils.synthetic_data import generate_synthetic_quantitative_data

        data = generate_synthetic_quantitative_data(count=5, include_time_series=True, include_comparison=False)

        # Should have base data + time series data
        assert len(data) > 5
        # Check for year field in time series data
        years = [item.get("year") for item in data if "year" in item]
        assert len(years) > 0

    def test_generate_synthetic_quantitative_data_with_comparison(self):
        """Test synthetic data with comparison segments."""
        from utils.synthetic_data import generate_synthetic_quantitative_data

        data = generate_synthetic_quantitative_data(count=5, include_time_series=False, include_comparison=True)

        # Should have comparison data
        has_segments = any("segment" in item for item in data)
        assert has_segments

    def test_generate_synthetic_research_data(self):
        """Test synthetic research data generation."""
        from utils.synthetic_data import generate_synthetic_research_data

        data = generate_synthetic_research_data(num_sources=3)

        assert isinstance(data, list)
        assert len(data) == 3
        assert all("url" in item for item in data)
        assert all("title" in item for item in data)


class TestSettingsValidation:
    """Tests for settings validation."""

    def test_settings_update_model_exists(self):
        """Test that SettingsUpdate model exists and can be imported."""
        from app_research import SettingsUpdate

        assert SettingsUpdate is not None

    def test_settings_update_valid_api_key(self):
        """Test valid API key validation."""
        from app_research import SettingsUpdate

        # Valid Google API key format
        settings = SettingsUpdate(api_key="AIzaSyTestKey123456789")
        assert settings.api_key == "AIzaSyTestKey123456789"

    def test_settings_update_invalid_api_key(self):
        """Test invalid API key validation."""
        from pydantic import ValidationError
        from app_research import SettingsUpdate

        with pytest.raises(ValidationError):
            SettingsUpdate(api_key="short")

    def test_settings_update_temperature_validation(self):
        """Test temperature range validation."""
        from pydantic import ValidationError
        from app_research import SettingsUpdate

        # Valid temperature
        settings = SettingsUpdate(temperature=0.5)
        assert settings.temperature == 0.5

        # Invalid temperature (too high)
        with pytest.raises(ValidationError):
            SettingsUpdate(temperature=1.5)

        # Invalid temperature (negative)
        with pytest.raises(ValidationError):
            SettingsUpdate(temperature=-0.1)

    def test_settings_update_max_projects_validation(self):
        """Test max_projects range validation."""
        from pydantic import ValidationError
        from app_research import SettingsUpdate

        # Valid range
        settings = SettingsUpdate(max_projects=50)
        assert settings.max_projects == 50

        # Out of range (too high)
        with pytest.raises(ValidationError):
            SettingsUpdate(max_projects=150)

        # Out of range (too low)
        with pytest.raises(ValidationError):
            SettingsUpdate(max_projects=0)


class TestHealthCheck:
    """Tests for health check endpoint."""

    def test_health_check_endpoint_exists(self):
        """Test that health check endpoint is registered."""
        from app import app

        client = app.test_client()
        response = client.get('/api/health')

        assert response.status_code == 200
        data = response.get_json()
        assert "status" in data
        assert "version" in data


class TestDatabasePooling:
    """Tests for database connection pooling."""

    def test_get_db_context_manager(self):
        """Test that get_db context manager works."""
        from database import get_db

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1

    def test_init_db(self):
        """Test database initialization."""
        from database import init_db

        result = init_db()
        assert result is True

    def test_create_project(self):
        """Test project creation."""
        from database import create_project, delete_project

        project_id = "test_project_create_123"
        result = create_project(project_id, "Test query")

        assert result is True

        # Cleanup
        delete_project(project_id)


class TestLangGraphTools:
    """Tests for LangGraph tools."""

    def test_search_web_tool_exists(self):
        """Test that search_web tool exists."""
        from tools.langgraph_tools import search_web

        assert search_web is not None

    def test_browse_url_tool_exists(self):
        """Test that browse_url tool exists."""
        from tools.langgraph_tools import browse_url

        assert browse_url is not None

    def test_research_tools_list(self):
        """Test that RESEARCH_TOOLS list exists and has items."""
        from tools.langgraph_tools import RESEARCH_TOOLS

        assert isinstance(RESEARCH_TOOLS, list)
        assert len(RESEARCH_TOOLS) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
