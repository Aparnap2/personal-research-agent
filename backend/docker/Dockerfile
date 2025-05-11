FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for data analysis and visualization
RUN pip instal \
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

# Command to run when container starts
CMD ["python", "-u", "code_executor.py"]