"""Shared synthetic data generation for testing and fallback scenarios."""

import random
from typing import List, Dict, Any, Optional

# Seed for reproducibility in tests
_DEFAULT_SEED = 42


def generate_synthetic_quantitative_data(
    count: int = 15,
    include_time_series: bool = True,
    include_comparison: bool = True,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generate synthetic quantitative data for analysis.

    Args:
        count: Number of primary metrics to generate
        include_time_series: Include time series data
        include_comparison: Include comparison data across segments
        seed: Optional random seed for reproducibility

    Returns:
        List of dictionaries containing metric data
    """
    if seed is not None:
        random.seed(seed)

    data: List[Dict[str, Any]] = [
        {"metric_name": "Primary Metric 1", "value": 75.0, "unit": "%", "category": "Performance", "confidence": 3, "source_citation": "Synthetic data"},
        {"metric_name": "Primary Metric 2", "value": 42.0, "unit": "%", "category": "Adoption", "confidence": 3, "source_citation": "Synthetic data"},
        {"metric_name": "Primary Metric 3", "value": 125000.0, "unit": "USD", "category": "Financial", "confidence": 3, "source_citation": "Synthetic data"},
        {"metric_name": "Primary Metric 4", "value": 88.0, "unit": "%", "category": "Performance", "confidence": 3, "source_citation": "Synthetic data"},
        {"metric_name": "Primary Metric 5", "value": 67.0, "unit": "%", "category": "Adoption", "confidence": 3, "source_citation": "Synthetic data"},
        {"metric_name": "Primary Metric 6", "value": 89.0, "unit": "%", "category": "Performance", "confidence": 3, "source_citation": "Synthetic data"},
        {"metric_name": "Primary Metric 7", "value": 56.0, "unit": "%", "category": "Adoption", "confidence": 3, "source_citation": "Synthetic data"},
        {"metric_name": "Primary Metric 8", "value": 234000.0, "unit": "USD", "category": "Financial", "confidence": 3, "source_citation": "Synthetic data"},
        {"metric_name": "Primary Metric 9", "value": 45.0, "unit": "%", "category": "Adoption", "confidence": 3, "source_citation": "Synthetic data"},
        {"metric_name": "Primary Metric 10", "value": 92.0, "unit": "%", "category": "Performance", "confidence": 3, "source_citation": "Synthetic data"},
        {"metric_name": "Primary Metric 11", "value": 78.0, "unit": "%", "category": "Performance", "confidence": 3, "source_citation": "Synthetic data"},
        {"metric_name": "Primary Metric 12", "value": 34.0, "unit": "%", "category": "Adoption", "confidence": 3, "source_citation": "Synthetic data"},
        {"metric_name": "Primary Metric 13", "value": 156000.0, "unit": "USD", "category": "Financial", "confidence": 3, "source_citation": "Synthetic data"},
        {"metric_name": "Primary Metric 14", "value": 61.0, "unit": "%", "category": "Adoption", "confidence": 3, "source_citation": "Synthetic data"},
        {"metric_name": "Primary Metric 15", "value": 83.0, "unit": "%", "category": "Performance", "confidence": 3, "source_citation": "Synthetic data"},
    ]

    if include_time_series:
        categories = ["Performance", "Adoption", "Financial"]
        years = [2019, 2020, 2021, 2022, 2023]
        for category in categories:
            base_value = random.randint(20, 50)
            for year_idx, year in enumerate(years):
                growth = random.uniform(0.05, 0.25)
                value = base_value * (1 + growth * year_idx)
                data.append({
                    "metric_name": f"{category} Metric",
                    "value": round(value, 1),
                    "unit": "%",
                    "year": year,
                    "category": category,
                    "confidence": 3,
                    "source_citation": "Synthetic time series data"
                })

    if include_comparison:
        segments = ["Segment A", "Segment B", "Segment C", "Segment D"]
        metrics = ["Market Share", "Growth Rate", "Customer Satisfaction"]
        for metric in metrics:
            for segment in segments:
                data.append({
                    "metric_name": metric,
                    "segment": segment,
                    "value": random.uniform(10, 90),
                    "unit": "%",
                    "category": "Comparison",
                    "confidence": 3,
                    "source_citation": "Synthetic comparison data"
                })

    return data


def generate_synthetic_research_data(
    num_sources: int = 5,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generate synthetic research source data.

    Args:
        num_sources: Number of sources to generate
        seed: Optional random seed for reproducibility

    Returns:
        List of dictionaries containing source data
    """
    if seed is not None:
        random.seed(seed)

    sources = []
    for i in range(num_sources):
        sources.append({
            "url": f"https://example{i+1}.com/article",
            "title": f"Research Article {i+1}: Important Findings",
            "authors": [f"Author {j+1}" for j in range(random.randint(1, 3))],
            "publish_date": f"{2020 + random.randint(0, 4)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            "content": f"This is the content of research article {i+1}. " * 10,
            "domain": f"example{i+1}.com",
        })

    return sources


# Backwards compatibility alias
generate_synthetic_data = generate_synthetic_quantitative_data
