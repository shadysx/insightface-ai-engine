# Facematch AI Engine V2

## Prerequisites

1. Create a new conda environment:
```bash
conda create -n insightface-ai-engine python=3.8
```

2. Activate the environment:
```bash
conda activate insightface-ai-engine
```

3. Install required packages:
```bash
conda install -c conda-forge insightface
```

4. Install OpenCV:
```bash
conda install -c menpo opencv
```

```bash
conda install numpy
```

```bash
pip install onnxruntime-gpu
```

```bash
pip install faiss-gpu
```

```bash
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.9
```

```bash
conda install pytorch torchvision torchaudio -c pytorch
```

```bash
conda install fastapi uvicorn
```

```bash
pip install python-multipart
```

# Test the API
```bash
curl -X POST "http://127.0.0.1:8000/ai/match?threshold=0.65" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@face5.jpg"

curl -X POST "http://127.0.0.1:8000/ai/brains" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\": \"user1\", \"brain_id\": \"brain1\"}"
```


