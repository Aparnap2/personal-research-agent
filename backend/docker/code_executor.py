#!/usr/bin/env python3
"""
Code Executor for Research Agent
This script runs inside the Docker container and executes Python code for data analysis and visualization.
"""

import os
import sys
import json
import traceback
import io
import base64
import time
import logging
from contextlib import redirect_stdout, redirect_stderr
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("code_executor")

# Constants
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"
MAX_EXECUTION_TIME = 60  # Maximum execution time in seconds

def setup_environment():
    """Set up the execution environment."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info("Environment setup complete")

def read_input():
    """Read input data and code from the input directory."""
    try:
        with open(os.path.join(INPUT_DIR, "code.py"), "r") as f:
            code = f.read()
        
        # Check if data file exists
        data_path = os.path.join(INPUT_DIR, "data.json")
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                data = json.load(f)
        else:
            data = {}
        
        return code, data
    except Exception as e:
        logger.error(f"Error reading input: {str(e)}")
        return None, None

def execute_code(code, data):
    """Execute the provided code with the given data."""
    # Create a dictionary for local variables
    local_vars = {
        "data": data,
        "pd": pd,
        "np": np,
        "plt": plt,
        "output_dir": OUTPUT_DIR,
        "results": {"figures": [], "tables": [], "metrics": {}, "error": None}
    }
    
    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    start_time = time.time()
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            # Execute the code with a timeout
            exec(code, globals(), local_vars)
            
            # Save all open figures
            for i, fig in enumerate(plt.get_fignums()):
                figure = plt.figure(fig)
                fig_path = os.path.join(OUTPUT_DIR, f"figure_{i}.png")
                figure.savefig(fig_path, bbox_inches='tight', dpi=300)
                
                # Convert figure to base64 for JSON response
                buffer = io.BytesIO()
                figure.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
                buffer.seek(0)
                img_str = base64.b64encode(buffer.read()).decode('utf-8')
                
                local_vars["results"]["figures"].append({
                    "path": fig_path,
                    "base64": img_str,
                    "index": i
                })
                
                plt.close(figure)
        
        # Check execution time
        execution_time = time.time() - start_time
        if execution_time > MAX_EXECUTION_TIME:
            logger.warning(f"Code execution took {execution_time:.2f} seconds, which exceeds the recommended limit of {MAX_EXECUTION_TIME} seconds")
        
        # Get stdout and stderr
        stdout = stdout_capture.getvalue()
        stderr = stderr_capture.getvalue()
        
        # Add to results
        local_vars["results"]["stdout"] = stdout
        local_vars["results"]["stderr"] = stderr
        local_vars["results"]["execution_time"] = execution_time
        
        return local_vars["results"]
    
    except Exception as e:
        error_msg = f"Error executing code: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return {
            "figures": [],
            "tables": [],
            "metrics": {},
            "error": error_msg,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "execution_time": time.time() - start_time
        }

def write_output(results):
    """Write execution results to the output directory."""
    try:
        # Write results to JSON file
        with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
            # Remove base64 data from JSON file to keep it smaller
            results_copy = results.copy()
            for fig in results_copy.get("figures", []):
                if "base64" in fig:
                    del fig["base64"]
            json.dump(results_copy, f, indent=2)
        
        # Write complete results (including base64) to another file
        with open(os.path.join(OUTPUT_DIR, "results_complete.json"), "w") as f:
            json.dump(results, f)
        
        logger.info("Results written to output directory")
        return True
    except Exception as e:
        logger.error(f"Error writing output: {str(e)}")
        return False

def main():
    """Main execution function."""
    logger.info("Starting code executor")
    setup_environment()
    
    # Check for input file
    if not os.path.exists(os.path.join(INPUT_DIR, "code.py")):
        logger.info("Waiting for input file...")
        return
    
    logger.info("Reading input files")
    code, data = read_input()
    
    if code is None:
        logger.error("Failed to read input files")
        return
    
    logger.info("Executing code")
    results = execute_code(code, data)
    
    logger.info("Writing output")
    write_output(results)
    
    logger.info("Execution complete")

if __name__ == "__main__":
    main()