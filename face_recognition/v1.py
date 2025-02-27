import cv2
import torch
import insightface
import numpy as np
from insightface.app import FaceAnalysis
import os
import faiss
import pickle

def check_gpu_support():
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
        print(f"Erreur lors de la vérification: {e}")


# Initialize face analysis model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
app.prepare(ctx_id=-1)  # ctx_id=-1 for CPU, 0 for GPU

def get_face_embedding(image_path):
    """Extract face embedding from an image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    faces = app.get(img)
    
    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")
    
    return faces[0].embedding

def compare_faces(emb1, emb2, threshold=0.65): # Adjust this threshold according to your usecase.
    """Compare two embeddings using cosine similarity"""
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    print(f"Similarity: {similarity}")
    return similarity, similarity > threshold

def build_face_database(dataset_path):
    """Build a FAISS index of face embeddings from the dataset"""
    embeddings = []
    file_paths = []
    
    for root, dirs, files in os.walk(dataset_path):
        for filename in files:
            print(f"Processing {filename}")
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, filename)
                try:
                    embedding = get_face_embedding(image_path)
                    embeddings.append(embedding)
                    file_paths.append(os.path.relpath(image_path, dataset_path))
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

def search_faces(index, file_paths, query_embedding, top_k=1, threshold=0.65):
    """Search for similar faces using FAISS"""
    # Normalize query vector
    query_embedding = query_embedding.astype('float32').reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    
    # Search
    similarities, indices = index.search(query_embedding, top_k)
    
    # Get best match
    best_similarity = similarities[0][0]
    best_match = file_paths[indices[0][0]] if best_similarity > threshold else None
    
    return best_match, best_similarity



try:
    # check_gpu_support()

    index_file = "../index/face_index.faiss"
    paths_file = "../index/file_paths.pkl"
    test_image_path = "../dataset/test_faces/face1.jpg"
    dataset_image_path = "../dataset/original_images"
    
    if os.path.exists(index_file) and os.path.exists(paths_file):
        print("Loading existing index...")
        index = faiss.read_index(index_file)
        with open(paths_file, 'rb') as f:
            file_paths = pickle.load(f)
    else:
        print("Building new index...")
        index, file_paths = build_face_database(dataset_image_path)
        # Save index and paths for future use
        faiss.write_index(index, index_file)
        with open(paths_file, 'wb') as f:
            pickle.dump(file_paths, f)
    
    # Get test image embedding
    test_embedding = get_face_embedding(test_image_path)
    
    # Search for matches
    best_match, best_similarity = search_faces(index, file_paths, test_embedding)
    
    if best_match:
        print(f"\nBest match: {best_match}")
        print(f"Best similarity score: {best_similarity:.4f}")
    else:
        print("No matches found in the database")
        
except Exception as e:
    print(f"Error: {str(e)}")