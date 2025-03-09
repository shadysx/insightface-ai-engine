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
conda pip install insightface
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
# Match a face in the default brain
curl -X POST "http://127.0.0.1:8080/ai/match?threshold=0.65" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@face5.jpg"

# Create a new brain
curl -X POST "http://127.0.0.1:8080/ai/brains/?user_id=user1&brain_id=brain1" \
  -H "accept: application/json"

# Delete a brain
curl -X DELETE "http://127.0.0.1:8080/ai/brains" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\": \"user1\", \"brain_id\": \"brain1\"}"

# Upload files to a brain
curl -X POST "http://127.0.0.1:8080/ai/users/Q0OHvIjD4MA8qOkNR6kSthAcoKoBtmS6/brains/cm7qxtuc10005jxdscc6wxopz/files" \
  -F "files=@face1.jpg" \
  -F "files=@face2.jpg" \
  -F "files=@face3.png"

# Get a file from a brain
curl "http://127.0.0.1:8080/ai/users/Q0OHvIjD4MA8qOkNR6kSthAcoKoBtmS6/brains/cm7rgyiq00001jxaso5pw4nn3/files/face1.jpg" \
  --output test_image.jpg
```
