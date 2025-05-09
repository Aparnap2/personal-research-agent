# research_agent/tools/chart_generator_tool.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import logging
from typing import List, Dict, Any, Optional
import time # For unique filenames

logger = logging.getLogger(__name__)

class ChartGeneratorTool:
    def __init__(self, output_dir: str, project_id: str): # Added project_id for logging
        self.output_dir = output_dir
        self.project_id = project_id # Store for logging
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"[{self.project_id}] ChartGeneratorTool: Output directory set to {self.output_dir}")

    def _save_chart(self, filename_base: str) -> str:
        # Use a more unique filename to prevent clashes if called rapidly
        chart_filename = f"{filename_base}_{int(time.time() * 1000)}.png"
        filepath = os.path.join(self.output_dir, chart_filename)
        try:
            plt.savefig(filepath, bbox_inches='tight', dpi=100) # Added dpi
            logger.info(f"[{self.project_id}] ChartGeneratorTool: Saved chart to {filepath}")
        except Exception as e:
            logger.error(f"[{self.project_id}] ChartGeneratorTool: Failed to save chart {filepath}: {e}", exc_info=True)
            raise
        finally:
            plt.close() # Ensure figure is closed to free memory
        return filepath

    def generate_bar_chart(self, data: List[Dict[str, Any]], category_col: str, value_col: str, title: str,
                           xlabel: Optional[str] = None, ylabel: Optional[str] = None) -> Optional[str]:
        logger.info(f"[{self.project_id}] ChartGeneratorTool: Generating bar chart for '{value_col}' by '{category_col}'. Title: '{title}'")
        if not data:
            logger.warning(f"[{self.project_id}] ChartGeneratorTool: No data provided for bar chart '{title}'.")
            return None
        try:
            df = pd.DataFrame(data)
            if category_col not in df.columns or value_col not in df.columns:
                logger.error(f"[{self.project_id}] ChartGeneratorTool: Columns '{category_col}' or '{value_col}' not in data for bar chart '{title}'. Available: {df.columns.tolist()}")
                return None
            
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce').fillna(0)
            df[category_col] = df[category_col].astype(str) # Ensure category is string

            if df[value_col].isnull().all(): # Check if all values became NaN/0
                 logger.warning(f"[{self.project_id}] ChartGeneratorTool: All values in '{value_col}' are non-numeric for bar chart '{title}'.")
                 return None

            plt.figure(figsize=(min(10, len(df[category_col].unique()) * 0.8 + 4), 6)) # Dynamic width
            bars = plt.bar(df[category_col], df[value_col])
            plt.title(title, fontsize=14)
            plt.xlabel(xlabel if xlabel else category_col.replace('_', ' ').title(), fontsize=10)
            plt.ylabel(ylabel if ylabel else value_col.replace('_', ' ').title(), fontsize=10)
            plt.xticks(rotation=45, ha="right", fontsize=9)
            plt.yticks(fontsize=9)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout() # Adjust layout

            # Add data labels if not too many bars
            if len(bars) <= 15:
                for bar in bars:
                    yval = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * df[value_col].abs().max(), f'{yval:.2f}', ha='center', va='bottom', fontsize=8)
            
            return self._save_chart(f"bar_{value_col.replace(' ','_')}_by_{category_col.replace(' ','_')}")
        except Exception as e:
            logger.error(f"[{self.project_id}] ChartGeneratorTool: Error generating bar chart '{title}': {e}", exc_info=True)
            return None
    # ... (generate_line_chart and generate_pie_chart with similar logging and error handling improvements) ...
    def generate_line_chart(self, data: List[Dict[str, Any]], x_col: str, y_col: str, title: str,
                            xlabel: Optional[str] = None, ylabel: Optional[str] = None, sort_x: bool = True) -> Optional[str]:
        logger.info(f"[{self.project_id}] ChartGeneratorTool: Generating line chart for '{y_col}' over '{x_col}'. Title: '{title}'")
        if not data:
            logger.warning(f"[{self.project_id}] ChartGeneratorTool: No data provided for line chart '{title}'.")
            return None
        try:
            df = pd.DataFrame(data)
            if x_col not in df.columns or y_col not in df.columns:
                logger.error(f"[{self.project_id}] ChartGeneratorTool: Columns '{x_col}' or '{y_col}' not in data for line chart '{title}'. Available: {df.columns.tolist()}")
                return None

            df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            df.dropna(subset=[x_col, y_col], inplace=True)

            if df.empty:
                logger.warning(f"[{self.project_id}] ChartGeneratorTool: No valid numeric data for line chart after conversion for '{x_col}', '{y_col}' in '{title}'.")
                return None

            if sort_x:
                df = df.sort_values(by=x_col)

            plt.figure(figsize=(10, 6))
            plt.plot(df[x_col], df[y_col], marker='o', linestyle='-')
            plt.title(title, fontsize=14)
            plt.xlabel(xlabel if xlabel else x_col.replace('_', ' ').title(), fontsize=10)
            plt.ylabel(ylabel if ylabel else y_col.replace('_', ' ').title(), fontsize=10)
            plt.xticks(rotation=45, ha="right", fontsize=9)
            plt.yticks(fontsize=9)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            return self._save_chart(f"line_{y_col.replace(' ','_')}_over_{x_col.replace(' ','_')}")
        except Exception as e:
            logger.error(f"[{self.project_id}] ChartGeneratorTool: Error generating line chart '{title}': {e}", exc_info=True)
            return None
            
    def generate_pie_chart(self, data: List[Dict[str, Any]], label_col: str, value_col: str, title: str) -> Optional[str]:
        logger.info(f"[{self.project_id}] ChartGeneratorTool: Generating pie chart for '{value_col}' by '{label_col}'. Title: '{title}'")
        if not data:
            logger.warning(f"[{self.project_id}] ChartGeneratorTool: No data provided for pie chart '{title}'.")
            return None
        try:
            df = pd.DataFrame(data)
            if label_col not in df.columns or value_col not in df.columns:
                logger.error(f"[{self.project_id}] ChartGeneratorTool: Columns '{label_col}' or '{value_col}' not in data for pie chart '{title}'. Available: {df.columns.tolist()}")
                return None
            
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce').fillna(0)
            df[label_col] = df[label_col].astype(str)
            pie_data = df.groupby(label_col)[value_col].sum()
            pie_data = pie_data[pie_data > 0] # Only positive values for pie chart

            if pie_data.empty:
                logger.warning(f"[{self.project_id}] ChartGeneratorTool: No valid positive data for pie chart of '{value_col}' by '{label_col}' in '{title}'.")
                return None

            plt.figure(figsize=(8, 8))
            wedges, texts, autotexts = plt.pie(pie_data, labels=None, autopct='%1.1f%%', startangle=90, counterclock=False, pctdistance=0.85)
            plt.title(title, fontsize=14)
            plt.axis('equal') 
            # Create a legend if many slices, or labels are too long
            if len(pie_data) > 5:
                plt.legend(wedges, pie_data.index, title=label_col.replace('_',' ').title(), loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            else:
                plt.gca().set_labels(pie_data.index) # Use direct labels if few slices

            plt.tight_layout()
            return self._save_chart(f"pie_{value_col.replace(' ','_')}_by_{label_col.replace(' ','_')}")
        except Exception as e:
            logger.error(f"[{self.project_id}] ChartGeneratorTool: Error generating pie chart '{title}': {e}", exc_info=True)
            return None

    def generate_markdown_table(self, data: List[Dict[str, Any]], title: Optional[str] = None, project_id: Optional[str] = None) -> str:
        # Use self.project_id if project_id arg is not passed
        pid = project_id or self.project_id
        logger.info(f"[{pid}] ChartGeneratorTool: Generating markdown table titled '{title if title else 'Untitled'}'.")
        if not data:
            logger.warning(f"[{pid}] ChartGeneratorTool: No data provided for markdown table '{title}'.")
            return f"### {title}\n\nNo data available to display.\n" if title else "No data available to display.\n"
        try:
            df = pd.DataFrame(data)
            # Sanitize column names for Markdown (optional, but good practice)
            # df.columns = [str(col).replace('_', ' ').title() for col in df.columns]
            md_table = df.to_markdown(index=False)
            if title:
                return f"### {title}\n\n{md_table}\n"
            return md_table
        except Exception as e:
            logger.error(f"[{pid}] ChartGeneratorTool: Error generating markdown table '{title}': {e}", exc_info=True)
            return f"### {title}\n\nError generating table: {e}\n" if title else f"Error generating table: {e}\n"