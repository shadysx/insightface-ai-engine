from deepface import DeepFace
import json
from pprint import pprint

models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
  "Buffalo_L" 
]

def verify_faces(img1_path, img2_path):
    result = DeepFace.verify(
        img1_path = img1_path,
        img2_path = img2_path,
        model_name = models[0]
    )
    print("\n=== JSON Formatted Results ===")
    print(json.dumps(result, indent=2))
    return result


def analyze_faces(img_path):
    result = DeepFace.analyze(
        img_path = img_path,
        actions = ['age', 'gender', 'race', 'emotion']
    )
    
    print(f"""
        === Face Analysis Results ===
        👤 Age: {result[0]['age']} years old
        🎯 Face Confidence: {result[0]['face_confidence']:.2f}%

        👫 Gender Analysis:
        • Dominant: {result[0]['dominant_gender']}
        • Confidence: {result[0]['gender']['Man']:.2f}%

        🌍 Ethnicity Analysis:
        • Dominant: {result[0]['dominant_race']}
        • Distribution:
            - Latino/Hispanic: {result[0]['race']['latino hispanic']:.1f}%
            - White: {result[0]['race']['white']:.1f}%
            - Middle Eastern: {result[0]['race']['middle eastern']:.1f}%
            - Indian: {result[0]['race']['indian']:.1f}%
            - Asian: {result[0]['race']['asian']:.1f}%
            - Black: {result[0]['race']['black']:.1f}%

        😊 Emotion Analysis:
        • Dominant: {result[0]['dominant_emotion']}
        • Distribution:
            - Happy: {result[0]['emotion']['happy']:.1f}%
            - Neutral: {result[0]['emotion']['neutral']:.1f}%
            - Sad: {result[0]['emotion']['sad']:.1f}%
            - Fear: {result[0]['emotion']['fear']:.1f}%
            - Angry: {result[0]['emotion']['angry']:.1f}%
            - Surprise: {result[0]['emotion']['surprise']:.1f}%
            - Disgust: {result[0]['emotion']['disgust']:.1f}%

        📐 Face Region:
        • Position: x={result[0]['region']['x']}, y={result[0]['region']['y']}
        • Size: {result[0]['region']['w']}x{result[0]['region']['h']}
        • Eyes: Left={result[0]['region']['left_eye']}, Right={result[0]['region']['right_eye']}
    """)

    return result

if __name__ == "__main__":
    # verify_faces("test-files/brad-pitt-1.jpg", "test-files/brad-pitt-2.jpg")
    analyze_faces("test-files/will-smith-1.jpg")