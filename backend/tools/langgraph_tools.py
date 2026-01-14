"""LangGraph/LangChain tool wrappers for the research agent."""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_core.tools import tool

from tools.web_scraper_tool import WebScraperTool
from tools.statistical_analyzer_tool import StatisticalAnalyzerTool
from tools.chart_generator_tool import ChartGeneratorTool

logger = logging.getLogger(__name__)


class SearchResult:
    """Result from web search."""
    def __init__(self, url: str, title: str = "", snippet: str = ""):
        self.url = url
        self.title = title
        self.snippet = snippet

    def __repr__(self) -> str:
        return f"SearchResult(url={self.url!r}, title={self.title!r})"


# Initialize tool instances
_scraper_tool = WebScraperTool()
_stats_tool = StatisticalAnalyzerTool()
_chart_tool = ChartGeneratorTool(project_id="default", output_dir="/tmp/charts")


@tool
def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search the web and return top results as a list of dicts.

    Args:
        query: The search query
        max_results: Maximum number of results to return (default 5)

    Returns:
        List of dictionaries with 'url', 'title', and 'snippet' keys
    """
    logger.info(f"Searching web for: {query}")
    try:
        # Run synchronously by wrapping async call
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            urls = loop.run_until_complete(
                _scraper_tool.search_and_extract_urls(query, "research", max_results)
            )
        finally:
            loop.close()

        # Return as list of dicts
        return [{"url": url, "title": "", "snippet": ""} for url in urls]
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return [{"error": str(e)}]


@tool
def browse_url(url: str) -> str:
    """Browse a URL and return its content as markdown.

    Args:
        url: The URL to browse

    Returns:
        Markdown content of the page
    """
    logger.info(f"Browsing URL: {url}")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                _scraper_tool._scrape_single_url_to_markdown(url, "research")
            )
        finally:
            loop.close()

        return result if result else f"Failed to fetch content from {url}"
    except Exception as e:
        logger.error(f"Browse failed: {e}")
        return f"Error browsing {url}: {e}"


@tool
def browse_urls(urls: List[str]) -> Dict[str, str]:
    """Browse multiple URLs and return their content.

    Args:
        urls: List of URLs to browse

    Returns:
        Dictionary mapping URLs to their markdown content
    """
    logger.info(f"Browsing {len(urls)} URLs")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(
                _scraper_tool.ascrape_urls_to_markdown(urls, "research")
            )
        finally:
            loop.close()

        return results
    except Exception as e:
        logger.error(f"Batch browse failed: {e}")
        return {url: f"Error: {e}" for url in urls}


@tool
def analyze_quantitative_data(
    data: List[Dict[str, Any]],
    column_name: str = "value"
) -> Dict[str, Any]:
    """Analyze quantitative data and return statistics.

    Args:
        data: List of dictionaries containing numeric values
        column_name: The key containing the numeric value

    Returns:
        Dictionary with statistical analysis results
    """
    logger.info(f"Analyzing quantitative data: {len(data)} items")
    try:
        stats = _stats_tool.calculate_descriptive_stats(data, column_name, "research")
        return stats
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {"error": str(e)}


@tool
def calculate_correlation(
    data: List[Dict[str, Any]],
    columns: List[str]
) -> Dict[str, Any]:
    """Calculate correlation between columns in the data.

    Args:
        data: List of dictionaries
        columns: List of column names to correlate

    Returns:
        Dictionary with correlation matrix and insights
    """
    logger.info(f"Calculating correlation for columns: {columns}")
    try:
        result = _stats_tool.perform_correlation_analysis(data, columns, "research")
        return result
    except Exception as e:
        logger.error(f"Correlation calculation failed: {e}")
        return {"error": str(e)}


@tool
def generate_bar_chart(
    data: List[Dict[str, Any]],
    category_col: str,
    value_col: str,
    title: str = "Chart"
) -> str:
    """Generate a bar chart from data.

    Args:
        data: List of dictionaries
        category_col: Column for x-axis categories
        value_col: Column for y-axis values
        title: Chart title

    Returns:
        Path to the generated chart file
    """
    logger.info(f"Generating bar chart: {title}")
    try:
        result = _chart_tool.generate_bar_chart(
            data, category_col, value_col, title, "research"
        )
        return result
    except Exception as e:
        logger.error(f"Chart generation failed: {e}")
        return f"Error: {e}"


@tool
def generate_line_chart(
    data: List[Dict[str, Any]],
    x_col: str,
    y_col: str,
    title: str = "Line Chart"
) -> str:
    """Generate a line chart from data.

    Args:
        data: List of dictionaries
        x_col: Column for x-axis (typically time)
        y_col: Column for y-axis values
        title: Chart title

    Returns:
        Path to the generated chart file
    """
    logger.info(f"Generating line chart: {title}")
    try:
        result = _chart_tool.generate_line_chart(
            data, x_col, y_col, title, "research"
        )
        return result
    except Exception as e:
        logger.error(f"Chart generation failed: {e}")
        return f"Error: {e}"


# List of all tools for easy import
RESEARCH_TOOLS = [
    search_web,
    browse_url,
    browse_urls,
    analyze_quantitative_data,
    calculate_correlation,
    generate_bar_chart,
    generate_line_chart,
]
