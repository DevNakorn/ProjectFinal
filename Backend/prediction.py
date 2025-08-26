import sys
import cv2
import numpy as np
import joblib

def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print("Error: Image not found or cannot be read.")
        return None
    image = cv2.resize(image, (224, 224))
    brightness = np.mean(image)
    contrast = np.std(image)
    flat_features = image.flatten()[:5000]
    return [brightness, contrast] + list(flat_features)

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        return

    image_path = sys.argv[1]

    try:
        model = joblib.load("D:\\Project Final\\Backend\\models\\exposure_prediction_model.pkl")
    except Exception as e:
        print("Error loading model:", e)
        return

    features = extract_features(image_path)
    if features is None:
        return

    prediction = model.predict([features])[0]
    print(f"Predicted Exposure value for '{image_path}': {prediction:.2f}")

if __name__ == "__main__":
    main()
