from fastapi import FastAPI, UploadFile, File
import cv2
import torch
import insightface
import numpy as np
from insightface.app import FaceAnalysis
import os
import faiss
import pickle
import shutil
from typing import Optional, List
from fastapi import Header
from fastapi.responses import JSONResponse

app = FastAPI()

def create_brain_structure(user_id: str, brain_id: str):
    try:
        print(f"Creating brain for user: {user_id} and brain: {brain_id}")
        base_path = "brain-containers"
        user_path = os.path.join(base_path, user_id)
        brain_path = os.path.join(user_path, brain_id)
        
        if not os.path.exists(base_path):
            print(f"Creating base directory: {base_path}")
            os.makedirs(base_path)
            
        if not os.path.exists(user_path):
            print(f"Creating user directory: {user_path}")
            os.makedirs(user_path)
            
        if not os.path.exists(brain_path):
            print(f"Creating brain directory: {brain_path}")
            os.makedirs(brain_path)
            
        subdirs = ['dataset', 'index']
        created_paths = {}
        
        for subdir in subdirs:
            subdir_path = os.path.join(brain_path, subdir)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path)
                print(f"Created subdirectory: {subdir_path}")
            created_paths[subdir] = subdir_path
            
        return {
            "status": "success",
            "message": "Brain directory structure created successfully",
        }
    except Exception as e:
        raise Exception(f"Error creating brain directory structure: {str(e)}")

def delete_brain_structure(user_id: str, brain_id: str):
    try:
        print(f"Deleting brain: {brain_id}")
        base_path = "brain-containers"
        user_path = os.path.join(base_path, user_id)
        brain_path = os.path.join(user_path, brain_id)
        
        if os.path.exists(brain_path):
            shutil.rmtree(brain_path)
            print(f"Brain directory deleted: {brain_path}")
            return {"status": "success", "message": "Brain deleted successfully"}
    except Exception as e:
        raise Exception(f"Error deleting brain directory structure: {str(e)}")

async def upload_files_to_brain(user_id: str, brain_id: str, files: List[UploadFile]):
    try:
        results = []
        for file in files:
            print(f"Uploading file: {file.filename}")
            file_path = f"brain-containers/{user_id}/{brain_id}/{file.filename}"
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            content = await file.read()
            with open(file_path, "wb") as buffer:
                buffer.write(content)
                
            results.append({
                "filename": file.filename,
                "path": file_path
            })
        
        return {
            "status": "success",
            "files": results
        }
    except Exception as e:
        raise Exception(f"Error uploading files: {str(e)}")


async def get_best_match_from_upload(upload_file, threshold = 0.65):
    face_recognition = FaceRecognition(threshold)
    face_recognition.initialize_model()
    face_recognition.load_index()
    img_rgb = await face_recognition.convert_upload_to_img_rgb(upload_file)
    test_embedding = await face_recognition.get_face_embedding_from_img_rgb(img_rgb)
    best_match, best_similarity = face_recognition.search_faces(test_embedding)
    return best_match, best_similarity

def get_best_match_from_path(image_path, threshold = 0.65):
    face_recognition = FaceRecognition(threshold)
    face_recognition.initialize_model()
    face_recognition.load_index()
    test_embedding = face_recognition.get_face_embedding_from_path(image_path)
    best_match, best_similarity = face_recognition.search_faces(test_embedding)
    return best_match, best_similarity

class FaceRecognition:
    def __init__(self, threshold=0.65):
        self.app = None
        self.index = None
        self.threshold  = threshold
        self.index_file = "index/face_index.faiss"
        self.paths_file = "index/file_paths.pkl"
        self.dataset_image_path = "dataset/original_images"

    def check_gpu_support(self):
        """Vérifie la disponibilité du support GPU"""
        try:
            # Check PyTorch CUDA support
            torch_cuda_available = torch.cuda.is_available()
            print(f"\nPyTorch CUDA support: {'Available' if torch_cuda_available else 'Not available'}")
            if torch_cuda_available:
                print(f"PyTorch CUDA version: {torch.version.cuda}")
                print(f"GPU Device: {torch.cuda.get_device_name(0)}")
            
            # Check FAISS GPU support
            faiss_gpu_available = hasattr(faiss, 'GpuIndexFlatIP')
            print(f"\nFAISS GPU support: {'Available' if faiss_gpu_available else 'Not available'}")
            
            # Check OpenCV CUDA support
            cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
            print(f"\nOpenCV CUDA support: {'Available' if cuda_available else 'Not available'}")
            
            if cuda_available:
                print("\nGPU Info:")
                os.system('nvidia-smi')
                
        except Exception as e:
            print(f"Error checking GPU support: {e}")


    def initialize_model(self):
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
        self.app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GPU

    def get_face_embedding_from_path(self, image_path):
        """Extract face embedding from an image"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        faces = self.app.get(img)
        
        if len(faces) < 1:
            raise ValueError("No faces detected in the image")
        if len(faces) > 1:
            print("Warning: Multiple faces detected. Using first detected face")
        
        return faces[0].embedding

    async def get_face_embedding_from_img_rgb(self, img_rgb):
        faces = self.app.get(img_rgb)
        if len(faces) < 1:
            raise ValueError("No faces detected in the image")
        if len(faces) > 1:
            print("Warning: Multiple faces detected. Using first detected face")
        return faces[0].embedding
    
    async def convert_upload_to_img_rgb(self, upload_file):
        contents = await upload_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img_rgb

   # TODO: Check if this can improve the accuracy of the model
    def compare_faces(self, emb1, emb2, threshold=0.65): 
        """Compare two embeddings using cosine similarity"""
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"Similarity: {similarity}")
        return similarity, similarity > threshold

    def build_face_database(self):
        """Build a FAISS index of face embeddings from the dataset"""
        embeddings = []
        file_paths = []
        
        for root, dirs, files in os.walk(self.dataset_image_path):
            for filename in files:
                print(f"Processing {filename}")
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, filename)
                    try:
                        embedding = self.get_face_embedding_from_path(image_path)
                        embeddings.append(embedding)
                        file_paths.append(os.path.relpath(image_path, self.dataset_image_path))
                    except Exception as e:
                        print(f"Skipping {image_path}: {str(e)}")
        
        # Convert embeddings to numpy array
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings.shape[1]  # Should be 512 for InsightFace
        index = faiss.IndexFlatIP(dimension)  # Using inner product (cosine similarity)
        
        # Normalize vectors to use inner product as cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        return index, file_paths

    def search_faces(self, query_embedding, top_k=1):
        """Search for similar faces using FAISS"""
        # Normalize query vector
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        similarities, indices = self.index.search(query_embedding, top_k)
        
        # Get best match
        best_similarity = float(similarities[0][0])
        best_match = self.file_paths[indices[0][0]] if best_similarity > self.threshold else None
        
        return best_match, best_similarity

    def load_index(self):
        if os.path.exists(self.index_file) and os.path.exists(self.paths_file):
            print("Loading existing index...")
            self.index = faiss.read_index(self.index_file)
            with open(self.paths_file, 'rb') as f:
                self.file_paths = pickle.load(f)
        else:
            print("Building new index...")
            index, file_paths = self.build_face_database()
            # Save index and paths for future use
            faiss.write_index(index, self.index_file)
            with open(self.paths_file, 'wb') as f:
                pickle.dump(file_paths, f)

@app.post("/ai/matchs")
async def get_best_match(file: UploadFile, threshold: float = 0.65):
    best_match, best_similarity = await get_best_match_from_upload(file, threshold)
    return {"match": best_match, "similarity": best_similarity}

@app.post("/ai/brains/")
async def create_brain(user_id: str, brain_id: str):
    return create_brain_structure(user_id, brain_id)

@app.delete("/ai/brains/")
async def delete_brain(user_id: str, brain_id: str):
    return delete_brain_structure(user_id, brain_id)


@app.post("/ai/users/{user_id}/brains/{brain_id}/files")
async def upload_files(
    user_id: str,
    brain_id: str,
    files: List[UploadFile] = File(...)
):
    return await upload_files_to_brain(user_id, brain_id, files)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
