FROM continuumio/miniconda3:23.10.0-1

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    gcc \
    g++

# Create a Python 3.11 environment
RUN conda create -n myenv python=3.11 -y
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Install conda packages
RUN conda install -c menpo opencv && \
    conda install numpy && \
    conda install fastapi uvicorn

# Install pip packages
RUN pip install insightface && \
    pip install onnxruntime && \
    pip install faiss-cpu && \
    pip install python-multipart

# Copy the application code
COPY . .

EXPOSE 8000

# Start the application with the conda environment
CMD ["conda", "run", "-n", "myenv", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]