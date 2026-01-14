FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for data analysis and visualization
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scipy \
    scikit-learn \
    statsmodels \
    plotly \
    nbformat \
    nbconvert \
    ipython

# Create a non-root user to run the code
RUN useradd -m -u 1000 codeuser
USER codeuser

# Create directories for code execution
RUN mkdir -p /app/input /app/output
WORKDIR /app

# Copy the code execution script
COPY --chown=codeuser:codeuser code_executor.py /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MPLCONFIGDIR=/tmp/matplotlib

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import os; exit(0 if os.path.exists('/app/healthy') else 1)"

# Resource limits via Python (soft limits, docker-compose can enforce hard limits)
CMD ["python", "-c", \
     "import os; " \
     "os.environ['OPENBLAS_NUM_THREADS'] = '1'; " \
     "os.environ['MKL_NUM_THREADS'] = '1'; " \
     "with open('/app/healthy', 'w') as f: f.write('ready'); " \
     "import code_executor; code_executor.main()"]
