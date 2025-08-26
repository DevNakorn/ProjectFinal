import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xml.etree.ElementTree as ET

# Define paths based on the extracted structure
RAW_IMAGES_PATH = "D:\my-project\Data\Before"  # Raw images from the "Before" folder
XMP_FILES_PATH = "D:\my-project\Data\Xmp"  # XMP files are in subfolders named numerically

def extract_features(image_path):
    """
    Extract features from an image (e.g., brightness, contrast, etc.).
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    # Resize image to a fixed size for consistency
    image = cv2.resize(image, (224, 224))
    # Calculate brightness (mean pixel intensity)
    brightness = np.mean(image)
    # Calculate standard deviation of pixel intensity
    contrast = np.std(image)
    # Flatten the image as a feature vector (optional for other ML models)
    flat_features = image.flatten()[:5000]  # Use only first 5000 features for efficiency
    return [brightness, contrast] + list(flat_features)

def parse_xmp(file_path):
    """
    Parses the .xmp file as plain text to extract the 'Exposure2012' value.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if "Exposure2012" in line:
                    # Extract the value using string manipulation
                    start = line.find('Exposure2012="') + len('Exposure2012="')
                    end = line.find('"', start)
                    return float(line[start:end])
        print(f"'Exposure2012' not found in: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


# Load and process data
def load_data():
    features = []
    labels = []
    count = 0
    for raw_file in os.listdir(RAW_IMAGES_PATH):
        
        if raw_file.endswith(".jpg"):  # Adjust if using other raw formats
            raw_path = os.path.join(RAW_IMAGES_PATH, raw_file)
            print("Processing Image:", raw_path)
            feature = extract_features(raw_path)
            if feature is None:
                print("Failed to extract features from:", raw_path)
            # Extract subfolder name from the file name (e.g., Raw-1.jpg -> 1)
            subfile_name = raw_file.split("-")[1].split(".")[0]
            xmp_file = f"{"D:\my-project\Data\Xmp\Raw-"}{subfile_name}.xmp"
            
            xmp_path = os.path.join(XMP_FILES_PATH, xmp_file)
            if os.path.exists(xmp_path):
                # Extract features and labels
                feature = extract_features(raw_path)
                if feature is not None:
                    print("Parsing XMP File:", xmp_path)
                    count+=1
                    label = parse_xmp(xmp_path)
                    if label is None:
                        print("Failed to parse exposure from:", xmp_path)

                    if label is not None:
                        features.append(feature)
                        labels.append(label)
    print("-"*50)
    print(f"Total xmp files: {count}")   
    return np.array(features), np.array(labels)

print("Raw Images Path:", RAW_IMAGES_PATH)
print("XMP Files Path:", XMP_FILES_PATH)
print("Raw Files:", os.listdir(RAW_IMAGES_PATH))


# Load data
features, labels = load_data()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Save the model
import joblib
joblib.dump(model, "exposure_prediction_model.pkl")
print("Model saved as exposure_prediction_model.pkl")
