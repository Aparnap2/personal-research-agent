# research_agent/tools/statistical_analyzer_tool.py
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class StatisticalAnalyzerTool:
    def calculate_descriptive_stats(self, data: List[Dict[str, Any]], column_name: str, project_id: str) -> Optional[Dict[str, Any]]:
        logger.info(f"[{project_id}] StatisticalAnalyzerTool: Calculating descriptive stats for column '{column_name}' with {len(data)} items.")
        try:
            values = []
            for item in data:
                if column_name in item:
                    val = item[column_name]
                    # Attempt to convert to numeric, handling various "N/A" or empty strings
                    if isinstance(val, (int, float)):
                        values.append(float(val))
                    elif isinstance(val, str):
                        try:
                            cleaned_val = val.replace(',', '').replace('$', '').replace('%','').strip()
                            if cleaned_val.lower() not in ['n/a', 'na', '', '-']:
                                values.append(float(cleaned_val))
                        except ValueError:
                            logger.debug(f"[{project_id}] StatTool: Could not convert string '{val}' to float for column '{column_name}'.")
                            pass # Skip non-convertible strings
            
            if not values:
                logger.warning(f"[{project_id}] StatisticalAnalyzerTool: No valid numerical data found for column '{column_name}' after cleaning.")
                return {"error": f"No valid numerical data for column '{column_name}'."}
            
            df = pd.DataFrame(values, columns=['value'])
            stats = {
                "column": column_name,
                "count": len(values),
                "mean": float(df['value'].mean()),
                "median": float(df['value'].median()),
                "std_dev": float(df['value'].std()),
                "min": float(df['value'].min()),
                "max": float(df['value'].max()),
                "25th_percentile": float(df['value'].quantile(0.25)),
                "75th_percentile": float(df['value'].quantile(0.75))
            }
            logger.info(f"[{project_id}] StatisticalAnalyzerTool: Stats for '{column_name}': Count={stats['count']}, Mean={stats['mean']:.2f}")
            return stats
        except Exception as e:
            logger.error(f"[{project_id}] StatisticalAnalyzerTool: Error calculating stats for '{column_name}': {e}", exc_info=True)
            return {"error": str(e)}

    def format_stats_as_text(self, stats_dict: Dict[str, Any], project_id: str) -> str:
        logger.info(f"[{project_id}] StatisticalAnalyzerTool: Formatting stats: {stats_dict.get('column', 'N/A')}")
        if not stats_dict or "error" in stats_dict:
            return f"Could not calculate statistics for '{stats_dict.get('column', 'N/A')}': {stats_dict.get('error', 'Unknown reason')}"
        
        column = stats_dict.get("column", "N/A")
        # Ensure all expected keys are present, provide 'N/A' if not
        return (
            f"Descriptive Statistics for '{column}':\n"
            f"  Count: {stats_dict.get('count', 'N/A')}\n"
            f"  Mean: {stats_dict.get('mean', 'N/A'):.2f}\n"
            f"  Median: {stats_dict.get('median', 'N/A'):.2f}\n"
            f"  Standard Deviation: {stats_dict.get('std_dev', 'N/A'):.2f}\n"
            f"  Min: {stats_dict.get('min', 'N/A'):.2f}\n"
            f"  Max: {stats_dict.get('max', 'N/A'):.2f}\n"
            f"  25th Percentile: {stats_dict.get('25th_percentile', 'N/A'):.2f}\n"
            f"  75th Percentile: {stats_dict.get('75th_percentile', 'N/A'):.2f}\n"
        )