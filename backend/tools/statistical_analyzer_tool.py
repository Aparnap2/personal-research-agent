# research_agent/tools/statistical_analyzer_tool.py
import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import List, Dict, Any, Optional, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import math
import json

logger = logging.getLogger(__name__)

class StatisticalAnalyzerTool:
    def __init__(self):
        """Initialize the StatisticalAnalyzerTool with default settings."""
        self.confidence_level = 0.95  # Default confidence level for intervals
        self.significance_level = 0.05  # Default significance level for hypothesis tests
    
    def extract_numeric_values(self, data: List[Dict[str, Any]], column_name: str, project_id: str) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Extract numeric values from a column in the data, with their original context."""
        values = []
        context_items = []
        
        for item in data:
            if column_name in item:
                val = item[column_name]
                # Attempt to convert to numeric, handling various formats
                if isinstance(val, (int, float)):
                    values.append(float(val))
                    context_items.append(item)
                elif isinstance(val, str):
                    try:
                        cleaned_val = val.replace(',', '').replace('$', '').replace('%','').strip()
                        if cleaned_val.lower() not in ['n/a', 'na', '', '-']:
                            values.append(float(cleaned_val))
                            context_items.append(item)
                    except ValueError:
                        logger.debug(f"[{project_id}] StatTool: Could not convert string '{val}' to float for column '{column_name}'.")
                        pass  # Skip non-convertible strings
        
        return values, context_items
    
    def calculate_descriptive_stats(self, data: List[Dict[str, Any]], column_name: str, project_id: str) -> Optional[Dict[str, Any]]:
        """Calculate comprehensive descriptive statistics for a column."""
        logger.info(f"[{project_id}] StatisticalAnalyzerTool: Calculating descriptive stats for column '{column_name}' with {len(data)} items.")
        try:
            values, context_items = self.extract_numeric_values(data, column_name, project_id)
            
            if not values:
                logger.warning(f"[{project_id}] StatisticalAnalyzerTool: No valid numerical data found for column '{column_name}' after cleaning.")
                return {"error": f"No valid numerical data for column '{column_name}'."}
            
            df = pd.DataFrame(values, columns=['value'])
            
            # Basic descriptive statistics
            basic_stats = {
                "column": column_name,
                "count": len(values),
                "mean": float(df['value'].mean()),
                "median": float(df['value'].median()),
                "mode": float(df['value'].mode().iloc[0]) if not df['value'].mode().empty else None,
                "std_dev": float(df['value'].std()),
                "variance": float(df['value'].var()),
                "min": float(df['value'].min()),
                "max": float(df['value'].max()),
                "range": float(df['value'].max() - df['value'].min()),
                "sum": float(df['value'].sum()),
                "25th_percentile": float(df['value'].quantile(0.25)),
                "75th_percentile": float(df['value'].quantile(0.75)),
                "iqr": float(df['value'].quantile(0.75) - df['value'].quantile(0.25)),
                "skewness": float(df['value'].skew()),
                "kurtosis": float(df['value'].kurtosis())
            }
            
            # Add confidence interval for the mean (95% by default)
            n = len(values)
            if n > 1:  # Need at least 2 values for confidence interval
                t_critical = stats.t.ppf((1 + self.confidence_level) / 2, n - 1)
                margin_of_error = t_critical * (basic_stats["std_dev"] / math.sqrt(n))
                basic_stats["confidence_interval_lower"] = basic_stats["mean"] - margin_of_error
                basic_stats["confidence_interval_upper"] = basic_stats["mean"] + margin_of_error
                basic_stats["confidence_level"] = self.confidence_level
            
            # Add normality test (Shapiro-Wilk)
            if 3 <= n <= 5000:  # Shapiro-Wilk test has sample size limitations
                shapiro_test = stats.shapiro(values)
                basic_stats["normality_test"] = {
                    "test_name": "Shapiro-Wilk",
                    "statistic": float(shapiro_test[0]),
                    "p_value": float(shapiro_test[1]),
                    "is_normal": shapiro_test[1] > self.significance_level
                }
            
            # Add outlier detection
            q1 = basic_stats["25th_percentile"]
            q3 = basic_stats["75th_percentile"]
            iqr = basic_stats["iqr"]
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = [v for v in values if v < lower_bound or v > upper_bound]
            basic_stats["outliers"] = {
                "count": len(outliers),
                "values": outliers[:10] if outliers else [],  # Limit to first 10 outliers
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            }
            
            # Add distribution characteristics
            if n >= 5:  # Need reasonable sample size for distribution characteristics
                percentiles = [5, 10, 25, 50, 75, 90, 95]
                basic_stats["percentiles"] = {
                    f"p{p}": float(df['value'].quantile(p/100)) for p in percentiles
                }
            
            # Add coefficient of variation
            if basic_stats["mean"] != 0:
                basic_stats["coefficient_of_variation"] = basic_stats["std_dev"] / abs(basic_stats["mean"])
            
            logger.info(f"[{project_id}] StatisticalAnalyzerTool: Comprehensive stats calculated for '{column_name}'")
            return basic_stats
        except Exception as e:
            logger.error(f"[{project_id}] StatisticalAnalyzerTool: Error calculating stats for '{column_name}': {e}", exc_info=True)
            return {"error": str(e)}
    
    def perform_correlation_analysis(self, data: List[Dict[str, Any]], columns: List[str], project_id: str) -> Dict[str, Any]:
        """Perform correlation analysis between multiple numeric columns."""
        logger.info(f"[{project_id}] StatisticalAnalyzerTool: Performing correlation analysis for {len(columns)} columns.")
        try:
            # Create a DataFrame with only the numeric values from specified columns
            df_data = {}
            
            for col in columns:
                values, _ = self.extract_numeric_values(data, col, project_id)
                if values:
                    df_data[col] = values
            
            # Ensure we have at least 2 columns with data
            if len(df_data) < 2:
                return {"error": "Need at least 2 columns with numeric data for correlation analysis."}
            
            # Create DataFrame with equal length columns (pad shorter ones with NaN)
            max_length = max(len(values) for values in df_data.values())
            for col, values in df_data.items():
                if len(values) < max_length:
                    df_data[col] = values + [np.nan] * (max_length - len(values))
            
            df = pd.DataFrame(df_data)
            
            # Calculate correlation matrix
            correlation_matrix = df.corr(method='pearson').to_dict()
            
            # Calculate p-values for correlations
            p_values = {}
            for col1 in df.columns:
                p_values[col1] = {}
                for col2 in df.columns:
                    if col1 != col2:
                        # Drop rows with NaN in either column
                        valid_data = df[[col1, col2]].dropna()
                        if len(valid_data) > 1:  # Need at least 2 points for correlation
                            r, p = stats.pearsonr(valid_data[col1], valid_data[col2])
                            p_values[col1][col2] = float(p)
                        else:
                            p_values[col1][col2] = None
                    else:
                        p_values[col1][col2] = 0.0  # p-value for self-correlation is 0
            
            # Identify significant correlations
            significant_correlations = []
            for col1 in df.columns:
                for col2 in df.columns:
                    if col1 < col2:  # Avoid duplicates and self-correlations
                        corr_value = correlation_matrix[col1][col2]
                        p_value = p_values[col1][col2]
                        
                        if p_value is not None and p_value < self.significance_level:
                            significant_correlations.append({
                                "variable1": col1,
                                "variable2": col2,
                                "correlation": corr_value,
                                "p_value": p_value,
                                "strength": self._interpret_correlation_strength(corr_value)
                            })
            
            # Sort by absolute correlation value
            significant_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            
            return {
                "correlation_matrix": correlation_matrix,
                "p_values": p_values,
                "significant_correlations": significant_correlations
            }
        except Exception as e:
            logger.error(f"[{project_id}] StatisticalAnalyzerTool: Error in correlation analysis: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret the strength of a correlation coefficient."""
        abs_corr = abs(correlation)
        if abs_corr < 0.1:
            return "Negligible"
        elif abs_corr < 0.3:
            return "Weak"
        elif abs_corr < 0.5:
            return "Moderate"
        elif abs_corr < 0.7:
            return "Strong"
        else:
            return "Very Strong"
    
    def perform_trend_analysis(self, data: List[Dict[str, Any]], value_column: str, time_column: str, project_id: str) -> Dict[str, Any]:
        """Perform trend analysis on time series data."""
        logger.info(f"[{project_id}] StatisticalAnalyzerTool: Performing trend analysis for {value_column} over {time_column}.")
        try:
            # Extract time and value data
            time_values = []
            numeric_values = []
            
            for item in data:
                if value_column in item and time_column in item:
                    time_val = item[time_column]
                    numeric_val = item[value_column]
                    
                    # Convert numeric value
                    if isinstance(numeric_val, (int, float)):
                        numeric_val = float(numeric_val)
                    elif isinstance(numeric_val, str):
                        try:
                            numeric_val = float(numeric_val.replace(',', '').replace('$', '').replace('%','').strip())
                        except ValueError:
                            continue
                    else:
                        continue
                    
                    # Add to our lists
                    time_values.append(time_val)
                    numeric_values.append(numeric_val)
            
            if len(time_values) < 3:
                return {"error": f"Need at least 3 time points for trend analysis, found {len(time_values)}."}
            
            # Create DataFrame
            df = pd.DataFrame({
                'time': time_values,
                'value': numeric_values
            })
            
            # Sort by time
            try:
                # Try to convert to datetime if it's a date/time column
                df['time'] = pd.to_datetime(df['time'])
                is_datetime = True
            except:
                # If not datetime, try to convert to numeric
                try:
                    df['time'] = pd.to_numeric(df['time'])
                    is_datetime = False
                except:
                    # If neither works, just use the original values and sort them
                    is_datetime = False
            
            df = df.sort_values('time')
            
            # Calculate basic trend statistics
            first_value = df['value'].iloc[0]
            last_value = df['value'].iloc[-1]
            total_change = last_value - first_value
            percent_change = (total_change / first_value) * 100 if first_value != 0 else float('inf')
            
            # Calculate moving averages
            if len(df) >= 3:
                df['MA_3'] = df['value'].rolling(window=3).mean()
            if len(df) >= 5:
                df['MA_5'] = df['value'].rolling(window=5).mean()
            
            # Linear regression for trend line
            x = np.arange(len(df))
            y = df['value'].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            trend_direction = "Increasing" if slope > 0 else "Decreasing" if slope < 0 else "Stable"
            trend_significance = "Significant" if p_value < self.significance_level else "Not Significant"
            
            # Calculate seasonality if enough data points
            seasonality_detected = False
            seasonality_period = None
            
            if len(df) >= 12:  # Need reasonable number of points for seasonality
                # Simple autocorrelation check for seasonality
                autocorr = pd.Series(df['value']).autocorr(lag=1)
                if autocorr > 0.7:  # High autocorrelation might indicate seasonality
                    seasonality_detected = True
            
            return {
                "data_points": len(df),
                "first_value": first_value,
                "last_value": last_value,
                "total_change": total_change,
                "percent_change": percent_change,
                "trend": {
                    "direction": trend_direction,
                    "slope": slope,
                    "intercept": intercept,
                    "r_squared": r_value**2,
                    "p_value": p_value,
                    "significance": trend_significance
                },
                "seasonality": {
                    "detected": seasonality_detected,
                    "period": seasonality_period
                },
                "volatility": float(df['value'].std())
            }
        except Exception as e:
            logger.error(f"[{project_id}] StatisticalAnalyzerTool: Error in trend analysis: {e}", exc_info=True)
            return {"error": str(e)}
    
    def format_stats_as_text(self, stats_dict: Dict[str, Any], project_id: str) -> str:
        """Format comprehensive statistics as readable text."""
        logger.info(f"[{project_id}] StatisticalAnalyzerTool: Formatting stats: {stats_dict.get('column', 'N/A')}")
        if not stats_dict or "error" in stats_dict:
            return f"Could not calculate statistics for '{stats_dict.get('column', 'N/A')}': {stats_dict.get('error', 'Unknown reason')}"
        
        column = stats_dict.get("column", "N/A")
        
        # Format basic statistics
        basic_stats = (
            f"## Descriptive Statistics for '{column}'\n\n"
            f"### Basic Measures\n"
            f"- **Count:** {stats_dict.get('count', 'N/A')}\n"
            f"- **Sum:** {stats_dict.get('sum', 'N/A'):.2f}\n"
            f"- **Mean:** {stats_dict.get('mean', 'N/A'):.2f}\n"
            f"- **Median:** {stats_dict.get('median', 'N/A'):.2f}\n"
            f"- **Mode:** {stats_dict.get('mode', 'N/A'):.2f if stats_dict.get('mode') is not None else 'N/A'}\n\n"
            
            f"### Dispersion Measures\n"
            f"- **Standard Deviation:** {stats_dict.get('std_dev', 'N/A'):.2f}\n"
            f"- **Variance:** {stats_dict.get('variance', 'N/A'):.2f}\n"
            f"- **Range:** {stats_dict.get('range', 'N/A'):.2f}\n"
            f"- **Minimum:** {stats_dict.get('min', 'N/A'):.2f}\n"
            f"- **Maximum:** {stats_dict.get('max', 'N/A'):.2f}\n"
            f"- **Interquartile Range (IQR):** {stats_dict.get('iqr', 'N/A'):.2f}\n\n"
            
            f"### Percentiles\n"
            f"- **25th Percentile (Q1):** {stats_dict.get('25th_percentile', 'N/A'):.2f}\n"
            f"- **50th Percentile (Median):** {stats_dict.get('median', 'N/A'):.2f}\n"
            f"- **75th Percentile (Q3):** {stats_dict.get('75th_percentile', 'N/A'):.2f}\n"
        )
        
        # Add distribution characteristics if available
        distribution_stats = ""
        if "skewness" in stats_dict and "kurtosis" in stats_dict:
            skew_interpretation = "symmetric"
            if stats_dict["skewness"] > 0.5:
                skew_interpretation = "positively skewed (right-tailed)"
            elif stats_dict["skewness"] < -0.5:
                skew_interpretation = "negatively skewed (left-tailed)"
            
            kurtosis_interpretation = "mesokurtic (normal distribution-like)"
            if stats_dict["kurtosis"] > 0.5:
                kurtosis_interpretation = "leptokurtic (heavy-tailed)"
            elif stats_dict["kurtosis"] < -0.5:
                kurtosis_interpretation = "platykurtic (light-tailed)"
            
            distribution_stats = (
                f"\n### Distribution Characteristics\n"
                f"- **Skewness:** {stats_dict.get('skewness', 'N/A'):.3f} ({skew_interpretation})\n"
                f"- **Kurtosis:** {stats_dict.get('kurtosis', 'N/A'):.3f} ({kurtosis_interpretation})\n"
                f"- **Coefficient of Variation:** {stats_dict.get('coefficient_of_variation', 'N/A'):.3f}\n"
            )
        
        # Add confidence interval if available
        confidence_interval = ""
        if "confidence_interval_lower" in stats_dict and "confidence_interval_upper" in stats_dict:
            confidence_level = stats_dict.get("confidence_level", 0.95) * 100
            confidence_interval = (
                f"\n### Confidence Interval ({confidence_level:.0f}%)\n"
                f"- **Lower Bound:** {stats_dict.get('confidence_interval_lower', 'N/A'):.2f}\n"
                f"- **Upper Bound:** {stats_dict.get('confidence_interval_upper', 'N/A'):.2f}\n"
                f"- **Interpretation:** We are {confidence_level:.0f}% confident that the true mean lies between "
                f"{stats_dict.get('confidence_interval_lower', 'N/A'):.2f} and {stats_dict.get('confidence_interval_upper', 'N/A'):.2f}.\n"
            )
        
        # Add normality test if available
        normality_test = ""
        if "normality_test" in stats_dict:
            test_info = stats_dict["normality_test"]
            normality_test = (
                f"\n### Normality Test ({test_info.get('test_name', 'N/A')})\n"
                f"- **Test Statistic:** {test_info.get('statistic', 'N/A'):.3f}\n"
                f"- **P-value:** {test_info.get('p_value', 'N/A'):.4f}\n"
                f"- **Interpretation:** The data {'appears to be normally distributed' if test_info.get('is_normal', False) else 'does not appear to be normally distributed'}.\n"
            )
        
        # Add outlier information if available
        outlier_info = ""
        if "outliers" in stats_dict:
            outlier_data = stats_dict["outliers"]
            outlier_info = (
                f"\n### Outlier Analysis\n"
                f"- **Number of Outliers:** {outlier_data.get('count', 0)}\n"
                f"- **Outlier Boundaries:** Values below {outlier_data.get('lower_bound', 'N/A'):.2f} or above {outlier_data.get('upper_bound', 'N/A'):.2f}\n"
            )
            
            if outlier_data.get('count', 0) > 0 and outlier_data.get('values', []):
                outlier_values = ", ".join([f"{v:.2f}" for v in outlier_data.get('values', [])[:5]])
                if len(outlier_data.get('values', [])) > 5:
                    outlier_values += ", ..."
                outlier_info += f"- **Example Outliers:** {outlier_values}\n"
        
        # Combine all sections
        full_text = basic_stats + distribution_stats + confidence_interval + normality_test + outlier_info
        
        return full_text
    
    def format_correlation_as_text(self, correlation_results: Dict[str, Any], project_id: str) -> str:
        """Format correlation analysis results as readable text."""
        logger.info(f"[{project_id}] StatisticalAnalyzerTool: Formatting correlation results")
        if not correlation_results or "error" in correlation_results:
            return f"Could not perform correlation analysis: {correlation_results.get('error', 'Unknown reason')}"
        
        # Format significant correlations
        significant_correlations = correlation_results.get("significant_correlations", [])
        
        if not significant_correlations:
            return "## Correlation Analysis\n\nNo significant correlations were found between the analyzed variables."
        
        # Create a markdown table of significant correlations
        correlation_table = "## Correlation Analysis\n\n### Significant Correlations\n\n"
        correlation_table += "| Variable 1 | Variable 2 | Correlation | Strength | P-value |\n"
        correlation_table += "|------------|------------|-------------|----------|--------|\n"
        
        for corr in significant_correlations:
            correlation_table += f"| {corr['variable1']} | {corr['variable2']} | {corr['correlation']:.3f} | {corr['strength']} | {corr['p_value']:.4f} |\n"
        
        # Add interpretation
        interpretation = "\n### Interpretation\n\n"
        
        # Highlight the strongest correlation
        if significant_correlations:
            strongest = significant_correlations[0]  # Already sorted by absolute correlation
            direction = "positive" if strongest["correlation"] > 0 else "negative"
            interpretation += (
                f"- The strongest correlation is a {direction} relationship between **{strongest['variable1']}** and "
                f"**{strongest['variable2']}** (r = {strongest['correlation']:.3f}).\n"
            )
        
        # Add general notes about correlation
        interpretation += (
            "\n**Note on Correlation:**\n"
            "- Correlation measures the strength and direction of a linear relationship between two variables.\n"
            "- Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation).\n"
            "- A value of 0 indicates no linear relationship.\n"
            "- Correlation does not imply causation.\n"
        )
        
        return correlation_table + interpretation
    
    def format_trend_analysis_as_text(self, trend_results: Dict[str, Any], value_column: str, time_column: str, project_id: str) -> str:
        """Format trend analysis results as readable text."""
        logger.info(f"[{project_id}] StatisticalAnalyzerTool: Formatting trend analysis results")
        if not trend_results or "error" in trend_results:
            return f"Could not perform trend analysis: {trend_results.get('error', 'Unknown reason')}"
        
        # Basic trend information
        trend_text = (
            f"## Trend Analysis: {value_column} over {time_column}\n\n"
            f"### Overview\n"
            f"- **Data Points:** {trend_results.get('data_points', 'N/A')}\n"
            f"- **First Value:** {trend_results.get('first_value', 'N/A'):.2f}\n"
            f"- **Last Value:** {trend_results.get('last_value', 'N/A'):.2f}\n"
            f"- **Total Change:** {trend_results.get('total_change', 'N/A'):.2f}\n"
            f"- **Percent Change:** {trend_results.get('percent_change', 'N/A'):.2f}%\n"
        )
        
        # Trend details
        trend_info = trend_results.get("trend", {})
        if trend_info:
            trend_text += (
                f"\n### Trend Details\n"
                f"- **Direction:** {trend_info.get('direction', 'N/A')}\n"
                f"- **Slope:** {trend_info.get('slope', 'N/A'):.4f} (change in {value_column} per unit of {time_column})\n"
                f"- **R-squared:** {trend_info.get('r_squared', 'N/A'):.4f} (goodness of fit)\n"
                f"- **Statistical Significance:** {trend_info.get('significance', 'N/A')} (p-value: {trend_info.get('p_value', 'N/A'):.4f})\n"
            )
        
        # Seasonality information
        seasonality_info = trend_results.get("seasonality", {})
        if seasonality_info:
            seasonality_text = "Detected" if seasonality_info.get("detected", False) else "Not detected"
            period_text = f", Period: {seasonality_info.get('period')}" if seasonality_info.get("period") else ""
            
            trend_text += (
                f"\n### Seasonality\n"
                f"- **Seasonality:** {seasonality_text}{period_text}\n"
                f"- **Volatility:** {trend_results.get('volatility', 'N/A'):.4f}\n"
            )
        
        # Interpretation
        trend_text += (
            f"\n### Interpretation\n"
            f"The data shows a {trend_info.get('direction', 'N/A').lower()} trend "
            f"that is {trend_info.get('significance', 'N/A').lower()}. "
        )
        
        if trend_info.get('r_squared', 0) > 0.7:
            trend_text += f"The trend explains a large portion of the variation in the data (R² = {trend_info.get('r_squared', 'N/A'):.2f})."
        elif trend_info.get('r_squared', 0) > 0.3:
            trend_text += f"The trend explains a moderate portion of the variation in the data (R² = {trend_info.get('r_squared', 'N/A'):.2f})."
        else:
            trend_text += f"The trend explains only a small portion of the variation in the data (R² = {trend_info.get('r_squared', 'N/A'):.2f})."
        
        if seasonality_info.get("detected", False):
            trend_text += " The data also shows evidence of seasonality, which should be considered when interpreting the trend."
        
        return trend_text