"""
Code Execution Service
This service handles the execution of Python code for data analysis and visualization
using a Docker container with the Qwen-2.5-Coder model for code generation.
"""

import os
import json
import logging
import time
import uuid
import subprocess
import requests
from typing import Dict, List, Any, Optional, Tuple
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DOCKER_INPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "docker/input"))
DOCKER_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "docker/output"))
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Ensure directories exist
os.makedirs(DOCKER_INPUT_DIR, exist_ok=True)
os.makedirs(DOCKER_OUTPUT_DIR, exist_ok=True)

class CodeExecutionService:
    """Service for generating and executing Python code for data analysis and visualization."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the service with optional API key for Hugging Face."""
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
    
    def generate_code(self, task_description: str, data: Dict[str, Any]) -> str:
        """Generate Python code using the Qwen-2.5-Coder model."""
        if not self.api_key:
            logger.warning("No Hugging Face API key provided. Using fallback code generation.")
            return self._generate_fallback_code(task_description, data)
        
        # Create a prompt for the model
        prompt = self._create_code_generation_prompt(task_description, data)
        
        # Call the Hugging Face API
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    HUGGINGFACE_API_URL,
                    headers=self.headers,
                    json={"inputs": prompt, "parameters": {"max_new_tokens": 2048, "temperature": 0.2}}
                )
                
                if response.status_code == 200:
                    # Extract the generated code
                    generated_text = response.json()[0]["generated_text"]
                    code = self._extract_code_from_response(generated_text)
                    return code
                else:
                    logger.error(f"API request failed with status {response.status_code}: {response.text}")
                    time.sleep(RETRY_DELAY)
            except Exception as e:
                logger.error(f"Error calling Hugging Face API: {str(e)}")
                time.sleep(RETRY_DELAY)
        
        # If all attempts fail, use fallback
        logger.warning("All API attempts failed. Using fallback code generation.")
        return self._generate_fallback_code(task_description, data)
    
    def _create_code_generation_prompt(self, task_description: str, data: Dict[str, Any]) -> str:
        """Create a prompt for code generation."""
        # Create a sample of the data for the prompt
        data_sample = self._create_data_sample(data)
        
        prompt = f"""You are a Python data scientist expert. Write Python code to analyze data and create visualizations based on the following task:

Task: {task_description}

The data is available as a Python dictionary named 'data' with the following structure:
{data_sample}

Your code should:
1. Use pandas, numpy, matplotlib, seaborn, and other data science libraries
2. Create clear, informative visualizations with proper titles, labels, and legends
3. Calculate relevant statistics and metrics
4. Store results in the 'results' dictionary with these keys:
   - 'metrics': Dictionary of calculated metrics
   - 'tables': List of pandas DataFrames or tables
5. Save figures using plt.savefig() or add them to the results

Only return the Python code without any explanations or markdown. The code should be ready to execute.

```python
# Your code here
"""
        return prompt
    
    def _extract_code_from_response(self, response_text: str) -> str:
        """Extract code from the model's response."""
        # Look for code between triple backticks
        if "```python" in response_text and "```" in response_text:
            start_idx = response_text.find("```python") + len("```python")
            end_idx = response_text.rfind("```")
            code = response_text[start_idx:end_idx].strip()
            return code
        
        # If no code block markers, try to extract just the code part
        if "import " in response_text:
            # Find the first import statement
            import_idx = response_text.find("import ")
            code = response_text[import_idx:].strip()
            return code
        
        # Return the whole response if we can't identify code blocks
        return response_text
    
    def _create_data_sample(self, data: Dict[str, Any]) -> str:
        """Create a sample representation of the data for the prompt."""
        sample = "{\n"
        
        for key, value in data.items():
            if isinstance(value, list):
                if len(value) > 0:
                    if isinstance(value[0], dict):
                        # For list of dictionaries, show first item and length
                        sample += f"    '{key}': List of {len(value)} dictionaries. First item: {value[0]},\n"
                    else:
                        # For list of values, show first few items
                        sample_items = value[:3]
                        sample += f"    '{key}': {sample_items} (List of {len(value)} items),\n"
                else:
                    sample += f"    '{key}': [] (empty list),\n"
            elif isinstance(value, dict):
                # For nested dictionaries, show keys
                sample += f"    '{key}': Dictionary with keys {list(value.keys())},\n"
            else:
                # For simple values, show the value
                sample += f"    '{key}': {value},\n"
        
        sample += "}"
        return sample
    
    def _generate_fallback_code(self, task_description: str, data: Dict[str, Any]) -> str:
        """Generate fallback code when API is not available."""
        # Determine what kind of analysis to do based on the task description
        is_time_series = any(keyword in task_description.lower() 
                            for keyword in ["time series", "trend", "over time", "historical"])
        
        is_comparison = any(keyword in task_description.lower() 
                           for keyword in ["compare", "comparison", "versus", "vs", "difference"])
        
        is_distribution = any(keyword in task_description.lower() 
                             for keyword in ["distribution", "histogram", "frequency", "spread"])
        
        is_correlation = any(keyword in task_description.lower() 
                            for keyword in ["correlation", "relationship", "association", "connect"])
        
        # Generate appropriate code based on the type of analysis
        if is_time_series:
            return self._generate_time_series_code(data)
        elif is_comparison:
            return self._generate_comparison_code(data)
        elif is_distribution:
            return self._generate_distribution_code(data)
        elif is_correlation:
            return self._generate_correlation_code(data)
        else:
            return self._generate_general_analysis_code(data)
    
    def _generate_time_series_code(self, data: Dict[str, Any]) -> str:
        """Generate code for time series analysis."""
        return """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Process data
if isinstance(data, list):
    df = pd.DataFrame(data)
elif isinstance(data, dict) and 'items' in data:
    df = pd.DataFrame(data['items'])
else:
    # Create a DataFrame from the data dictionary
    df = pd.DataFrame(data)

# Try to find date/time columns
date_cols = [col for col in df.columns if any(date_term in col.lower() 
                                             for date_term in ['date', 'time', 'year', 'month', 'day'])]

if date_cols:
    date_col = date_cols[0]
    # Try to convert to datetime
    try:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)
    except:
        pass
else:
    # If no date column, create a sequential one
    df['date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
    date_col = 'date'

# Find numeric columns for analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if date_col in numeric_cols:
    numeric_cols.remove(date_col)

# Create time series plot
plt.figure(figsize=(12, 6))

if numeric_cols:
    for i, col in enumerate(numeric_cols[:3]):  # Limit to first 3 numeric columns
        plt.plot(df[date_col], df[col], marker='o', linestyle='-', label=col)

plt.title('Time Series Analysis', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Format x-axis dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate()

# Calculate metrics
metrics = {}
for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
    metrics[f"{col}_mean"] = df[col].mean()
    metrics[f"{col}_median"] = df[col].median()
    metrics[f"{col}_std"] = df[col].std()
    metrics[f"{col}_min"] = df[col].min()
    metrics[f"{col}_max"] = df[col].max()
    
    # Calculate growth rate if time series
    if len(df) > 1:
        first_val = df[col].iloc[0]
        last_val = df[col].iloc[-1]
        if first_val != 0:
            growth_rate = ((last_val - first_val) / first_val) * 100
            metrics[f"{col}_growth_rate"] = f"{growth_rate:.2f}%"

# Create a summary table
summary_df = pd.DataFrame({
    'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
})

for col in numeric_cols[:3]:
    summary_df[col] = [
        f"{df[col].mean():.2f}",
        f"{df[col].median():.2f}",
        f"{df[col].std():.2f}",
        f"{df[col].min():.2f}",
        f"{df[col].max():.2f}"
    ]

# Add results to the results dictionary
results['metrics'] = metrics
results['tables'] = [summary_df.to_dict()]
"""
    
    def _generate_comparison_code(self, data: Dict[str, Any]) -> str:
        """Generate code for comparison analysis."""
        return """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Process data
if isinstance(data, list):
    df = pd.DataFrame(data)
elif isinstance(data, dict) and 'items' in data:
    df = pd.DataFrame(data['items'])
else:
    # Create a DataFrame from the data dictionary
    df = pd.DataFrame(data)

# Find categorical and numeric columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Create comparison visualizations
if categorical_cols and numeric_cols:
    # Bar chart for categorical vs numeric
    plt.figure(figsize=(12, 6))
    
    cat_col = categorical_cols[0]  # Use first categorical column
    num_col = numeric_cols[0]      # Use first numeric column
    
    # Group by categorical column and calculate mean of numeric column
    grouped_data = df.groupby(cat_col)[num_col].mean().sort_values(ascending=False).head(10)
    
    # Create bar chart
    ax = grouped_data.plot(kind='bar', color='skyblue')
    plt.title(f'Average {num_col} by {cat_col}', fontsize=16)
    plt.xlabel(cat_col, fontsize=12)
    plt.ylabel(f'Average {num_col}', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for i, v in enumerate(grouped_data):
        ax.text(i, v + (v * 0.02), f'{v:.2f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Create a second visualization - boxplot
    plt.figure(figsize=(12, 6))
    
    if len(categorical_cols) > 0 and len(df[categorical_cols[0]].unique()) <= 10:
        # Boxplot for distribution comparison
        sns.boxplot(x=categorical_cols[0], y=numeric_cols[0], data=df)
        plt.title(f'Distribution of {numeric_cols[0]} by {categorical_cols[0]}', fontsize=16)
        plt.xlabel(categorical_cols[0], fontsize=12)
        plt.ylabel(numeric_cols[0], fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
else:
    # If no categorical columns, compare numeric columns
    plt.figure(figsize=(12, 6))
    
    # Select up to 5 numeric columns
    cols_to_plot = numeric_cols[:5]
    
    # Normalize the data for comparison
    df_norm = df[cols_to_plot].copy()
    for col in cols_to_plot:
        df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    
    # Plot normalized values
    df_norm[cols_to_plot].mean().plot(kind='bar', color='skyblue')
    plt.title('Comparison of Normalized Average Values', fontsize=16)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Normalized Average Value', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

# Calculate comparison metrics
metrics = {}

if categorical_cols:
    cat_col = categorical_cols[0]
    for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
        # Group by categorical column
        group_stats = df.groupby(cat_col)[num_col].agg(['mean', 'median', 'std', 'min', 'max'])
        
        # Find category with highest and lowest mean
        max_cat = group_stats['mean'].idxmax()
        min_cat = group_stats['mean'].idxmin()
        
        metrics[f"{num_col}_highest_avg"] = f"{max_cat} ({group_stats.loc[max_cat, 'mean']:.2f})"
        metrics[f"{num_col}_lowest_avg"] = f"{min_cat} ({group_stats.loc[min_cat, 'mean']:.2f})"
        
        # Calculate the ratio between highest and lowest
        if group_stats.loc[min_cat, 'mean'] != 0:
            ratio = group_stats.loc[max_cat, 'mean'] / group_stats.loc[min_cat, 'mean']
            metrics[f"{num_col}_high_low_ratio"] = f"{ratio:.2f}x"

# Create comparison table
if categorical_cols:
    cat_col = categorical_cols[0]
    comparison_df = df.groupby(cat_col)[numeric_cols[:3]].agg(['mean', 'count']).round(2)
    comparison_df.columns = [f"{col}_{stat}" for col, stat in comparison_df.columns]
    
    # Add results to the results dictionary
    results['tables'] = [comparison_df.reset_index().to_dict()]
else:
    # Create a summary comparison table for numeric columns
    comparison_df = pd.DataFrame({
        'Metric': numeric_cols[:5],
        'Mean': [df[col].mean() for col in numeric_cols[:5]],
        'Median': [df[col].median() for col in numeric_cols[:5]],
        'Std Dev': [df[col].std() for col in numeric_cols[:5]],
        'Min': [df[col].min() for col in numeric_cols[:5]],
        'Max': [df[col].max() for col in numeric_cols[:5]]
    })
    
    # Add results to the results dictionary
    results['tables'] = [comparison_df.to_dict()]

results['metrics'] = metrics
"""
    
    def _generate_distribution_code(self, data: Dict[str, Any]) -> str:
        """Generate code for distribution analysis."""
        return """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Process data
if isinstance(data, list):
    df = pd.DataFrame(data)
elif isinstance(data, dict) and 'items' in data:
    df = pd.DataFrame(data['items'])
else:
    # Create a DataFrame from the data dictionary
    df = pd.DataFrame(data)

# Find numeric columns for analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if numeric_cols:
    # Create distribution plots for up to 3 numeric columns
    for i, col in enumerate(numeric_cols[:3]):
        plt.figure(figsize=(12, 6))
        
        # Create a subplot grid
        gs = plt.GridSpec(2, 2, height_ratios=[3, 1])
        
        # Histogram with KDE
        ax0 = plt.subplot(gs[0, :])
        sns.histplot(df[col], kde=True, ax=ax0, color='skyblue')
        ax0.set_title(f'Distribution of {col}', fontsize=16)
        ax0.set_xlabel(col, fontsize=12)
        ax0.set_ylabel('Frequency', fontsize=12)
        
        # Add mean and median lines
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax0.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax0.axvline(median_val, color='green', linestyle='-.', label=f'Median: {median_val:.2f}')
        ax0.legend()
        
        # Boxplot
        ax1 = plt.subplot(gs[1, 0])
        sns.boxplot(x=df[col], ax=ax1, color='skyblue')
        ax1.set_title('Boxplot', fontsize=12)
        ax1.set_xlabel(col, fontsize=10)
        
        # QQ plot
        ax2 = plt.subplot(gs[1, 1])
        stats.probplot(df[col].dropna(), plot=ax2)
        ax2.set_title('Q-Q Plot', fontsize=12)
        
        plt.tight_layout()
    
    # Calculate distribution metrics
    metrics = {}
    for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
        metrics[f"{col}_mean"] = df[col].mean()
        metrics[f"{col}_median"] = df[col].median()
        metrics[f"{col}_std"] = df[col].std()
        metrics[f"{col}_skewness"] = df[col].skew()
        metrics[f"{col}_kurtosis"] = df[col].kurtosis()
        
        # Calculate percentiles
        metrics[f"{col}_25th_percentile"] = df[col].quantile(0.25)
        metrics[f"{col}_75th_percentile"] = df[col].quantile(0.75)
        metrics[f"{col}_iqr"] = metrics[f"{col}_75th_percentile"] - metrics[f"{col}_25th_percentile"]
        
        # Test for normality
        if len(df) > 3:  # Need at least 3 data points for normality test
            _, p_value = stats.shapiro(df[col].dropna())
            metrics[f"{col}_normality_p_value"] = p_value
            metrics[f"{col}_is_normal"] = "Yes" if p_value > 0.05 else "No"
    
    # Create a summary table
    summary_df = pd.DataFrame({
        'Metric': ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis', '25th Percentile', '75th Percentile', 'IQR'],
    })
    
    for col in numeric_cols[:3]:
        summary_df[col] = [
            f"{df[col].mean():.2f}",
            f"{df[col].median():.2f}",
            f"{df[col].std():.2f}",
            f"{df[col].skew():.2f}",
            f"{df[col].kurtosis():.2f}",
            f"{df[col].quantile(0.25):.2f}",
            f"{df[col].quantile(0.75):.2f}",
            f"{df[col].quantile(0.75) - df[col].quantile(0.25):.2f}"
        ]
    
    # Add results to the results dictionary
    results['metrics'] = metrics
    results['tables'] = [summary_df.to_dict()]
else:
    # If no numeric columns, create a frequency distribution of categorical data
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_cols:
        plt.figure(figsize=(12, 6))
        
        cat_col = categorical_cols[0]  # Use first categorical column
        value_counts = df[cat_col].value_counts().head(10)  # Top 10 categories
        
        # Create bar chart
        ax = value_counts.plot(kind='bar', color='skyblue')
        plt.title(f'Frequency Distribution of {cat_col}', fontsize=16)
        plt.xlabel(cat_col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on top of bars
        for i, v in enumerate(value_counts):
            ax.text(i, v + (v * 0.02), str(v), ha='center', fontsize=9)
        
        plt.tight_layout()
        
        # Calculate metrics
        metrics = {
            f"{cat_col}_most_common": f"{value_counts.index[0]} ({value_counts.iloc[0]} occurrences)",
            f"{cat_col}_unique_values": len(df[cat_col].unique()),
            f"{cat_col}_missing_values": df[cat_col].isna().sum()
        }
        
        # Create a frequency table
        freq_df = pd.DataFrame({
            'Category': value_counts.index,
            'Count': value_counts.values,
            'Percentage': (value_counts.values / len(df) * 100).round(2)
        })
        
        # Add results to the results dictionary
        results['metrics'] = metrics
        results['tables'] = [freq_df.to_dict()]
"""
    
    def _generate_correlation_code(self, data: Dict[str, Any]) -> str:
        """Generate code for correlation analysis."""
        return """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Process data
if isinstance(data, list):
    df = pd.DataFrame(data)
elif isinstance(data, dict) and 'items' in data:
    df = pd.DataFrame(data['items'])
else:
    # Create a DataFrame from the data dictionary
    df = pd.DataFrame(data)

# Find numeric columns for correlation analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) >= 2:
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask for upper triangle
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    plt.title('Correlation Matrix', fontsize=16)
    plt.tight_layout()
    
    # Create scatter plots for top correlations
    # Get the top 3 correlations (excluding self-correlations)
    corr_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            corr_pairs.append((numeric_cols[i], numeric_cols[j], abs(corr_matrix.iloc[i, j])))
    
    # Sort by correlation strength
    corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Plot top 3 correlations (or fewer if there aren't 3)
    for i, (col1, col2, corr) in enumerate(corr_pairs[:3]):
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot with regression line
        sns.regplot(x=col1, y=col2, data=df, scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
        
        # Calculate Pearson correlation and p-value
        pearson_r, p_value = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
        
        plt.title(f'Correlation between {col1} and {col2}\nPearson r: {pearson_r:.3f}, p-value: {p_value:.3e}', 
                 fontsize=14)
        plt.xlabel(col1, fontsize=12)
        plt.ylabel(col2, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    # Calculate correlation metrics
    metrics = {}
    
    # Store top correlations
    for i, (col1, col2, corr) in enumerate(corr_pairs[:5]):
        metrics[f"top_corr_{i+1}_variables"] = f"{col1} & {col2}"
        metrics[f"top_corr_{i+1}_value"] = corr
        
        # Calculate Pearson correlation and p-value
        pearson_r, p_value = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
        metrics[f"top_corr_{i+1}_p_value"] = p_value
        metrics[f"top_corr_{i+1}_significant"] = "Yes" if p_value < 0.05 else "No"
    
    # Create a correlation table
    corr_df = corr_matrix.round(3).reset_index()
    corr_df.columns = ['Variable'] + list(corr_df.columns[1:])
    
    # Add results to the results dictionary
    results['metrics'] = metrics
    results['tables'] = [corr_df.to_dict()]
else:
    # If not enough numeric columns, analyze categorical associations
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(categorical_cols) >= 2 and len(df) > 10:
        plt.figure(figsize=(12, 8))
        
        # Create a contingency table
        cat1, cat2 = categorical_cols[0], categorical_cols[1]
        contingency = pd.crosstab(df[cat1], df[cat2])
        
        # Create a heatmap
        sns.heatmap(contingency, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5)
        plt.title(f'Contingency Table: {cat1} vs {cat2}', fontsize=16)
        plt.xlabel(cat2, fontsize=12)
        plt.ylabel(cat1, fontsize=12)
        plt.tight_layout()
        
        # Calculate chi-square test
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        
        # Calculate metrics
        metrics = {
            "chi2_statistic": chi2,
            "chi2_p_value": p,
            "chi2_degrees_of_freedom": dof,
            "association_significant": "Yes" if p < 0.05 else "No"
        }
        
        # Add results to the results dictionary
        results['metrics'] = metrics
        results['tables'] = [contingency.reset_index().to_dict()]
"""
    
    def _generate_general_analysis_code(self, data: Dict[str, Any]) -> str:
        """Generate general analysis code when no specific type is detected."""
        return """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Process data
if isinstance(data, list):
    df = pd.DataFrame(data)
elif isinstance(data, dict) and 'items' in data:
    df = pd.DataFrame(data['items'])
else:
    # Create a DataFrame from the data dictionary
    df = pd.DataFrame(data)

# Find categorical and numeric columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Create a general overview visualization
plt.figure(figsize=(12, 6))

# If we have numeric columns, create a summary visualization
if numeric_cols:
    # Create a bar chart of means for numeric columns
    means = df[numeric_cols[:5]].mean().sort_values(ascending=False)  # Top 5 columns
    
    ax = means.plot(kind='bar', color='skyblue')
    plt.title('Average Values by Metric', fontsize=16)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Average Value', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for i, v in enumerate(means):
        ax.text(i, v + (v * 0.02), f'{v:.2f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Create a second visualization - boxplot for distribution
    plt.figure(figsize=(12, 6))
    
    # Melt the DataFrame for easier plotting
    melted_df = df[numeric_cols[:5]].melt(var_name='Metric', value_name='Value')
    
    # Create boxplot
    sns.boxplot(x='Metric', y='Value', data=melted_df)
    plt.title('Distribution of Values by Metric', fontsize=16)
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

# If we have categorical columns, create a frequency visualization
if categorical_cols:
    plt.figure(figsize=(12, 6))
    
    cat_col = categorical_cols[0]  # Use first categorical column
    value_counts = df[cat_col].value_counts().head(10)  # Top 10 categories
    
    # Create bar chart
    ax = value_counts.plot(kind='bar', color='lightgreen')
    plt.title(f'Frequency Distribution of {cat_col}', fontsize=16)
    plt.xlabel(cat_col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for i, v in enumerate(value_counts):
        ax.text(i, v + (v * 0.02), str(v), ha='center', fontsize=9)
    
    plt.tight_layout()

# Calculate general metrics
metrics = {
    "row_count": len(df),
    "column_count": len(df.columns),
    "missing_values_count": df.isna().sum().sum(),
    "missing_values_percentage": (df.isna().sum().sum() / (len(df) * len(df.columns)) * 100)
}

# Add metrics for numeric columns
if numeric_cols:
    for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
        metrics[f"{col}_mean"] = df[col].mean()
        metrics[f"{col}_median"] = df[col].median()
        metrics[f"{col}_std"] = df[col].std()
        metrics[f"{col}_min"] = df[col].min()
        metrics[f"{col}_max"] = df[col].max()

# Create a summary table
summary_df = pd.DataFrame({
    'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
})

for col in numeric_cols[:5]:
    summary_df[col] = [
        len(df[col].dropna()),
        f"{df[col].mean():.2f}",
        f"{df[col].median():.2f}",
        f"{df[col].std():.2f}",
        f"{df[col].min():.2f}",
        f"{df[col].max():.2f}"
    ]

# Add results to the results dictionary
results['metrics'] = metrics
results['tables'] = [summary_df.to_dict()]
"""
    
    def execute_code(self, code: str, data: Dict[str, Any], execution_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute the generated code in the Docker container."""
        if not execution_id:
            execution_id = str(uuid.uuid4())
        
        # Create input directory for this execution
        input_dir = os.path.join(DOCKER_INPUT_DIR, execution_id)
        output_dir = os.path.join(DOCKER_OUTPUT_DIR, execution_id)
        
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Write code to file
            with open(os.path.join(input_dir, "code.py"), "w") as f:
                f.write(code)
            
            # Write data to file
            with open(os.path.join(input_dir, "data.json"), "w") as f:
                json.dump(data, f)
            
            # Execute code in Docker container
            logger.info(f"Executing code in Docker container with ID: {execution_id}")
            
            # Check if Docker container is running
            docker_ps = subprocess.run(
                ["docker", "ps", "--filter", "name=code-executor", "--format", "{{.Names}}"],
                capture_output=True,
                text=True
            )
            
            if "code-executor" not in docker_ps.stdout:
                logger.warning("Docker container not running. Starting container...")
                subprocess.run(
                    ["docker-compose", "up", "-d", "code-executor"],
                    cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                )
                time.sleep(5)  # Wait for container to start
            
            # Execute code in container
            subprocess.run(
                ["docker", "exec", "personal-research-agent-code-executor-1", "python", "code_executor.py"],
                capture_output=True,
                text=True
            )
            
            # Wait for results
            max_wait = 60  # seconds
            wait_time = 0
            results_file = os.path.join(output_dir, "results_complete.json")
            
            while not os.path.exists(results_file) and wait_time < max_wait:
                time.sleep(1)
                wait_time += 1
            
            if not os.path.exists(results_file):
                logger.error(f"Execution timed out after {max_wait} seconds")
                return {
                    "error": f"Execution timed out after {max_wait} seconds",
                    "figures": [],
                    "tables": [],
                    "metrics": {}
                }
            
            # Read results
            with open(results_file, "r") as f:
                results = json.load(f)
            
            return results
        
        except Exception as e:
            logger.error(f"Error executing code: {str(e)}")
            return {
                "error": f"Error executing code: {str(e)}",
                "figures": [],
                "tables": [],
                "metrics": {}
            }
        finally:
            # Clean up (optional - you might want to keep files for debugging)
            # shutil.rmtree(input_dir, ignore_errors=True)
            # shutil.rmtree(output_dir, ignore_errors=True)
            pass
    
    def analyze_data(self, task_description: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code and execute it to analyze data based on the task description."""
        # Generate code
        code = self.generate_code(task_description, data)
        
        # Execute code
        results = self.execute_code(code, data)
        
        # Add the generated code to the results
        results["generated_code"] = code
        
        return results