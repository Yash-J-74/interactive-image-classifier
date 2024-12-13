import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import cv2
import os

# Function to compute HOG features
def compute_hog_features(patches):
    hog_features = []
    for patch in patches:
        if patch.shape[0] < 32 or patch.shape[1] < 32:
            # Skip patches that are too small due to cropping near edges
            continue
        patch = cv2.resize(patch, (32, 32))  # Ensure all patches are 32x32
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        features = hog(
            gray_patch,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            orientations=9,
            block_norm='L2-Hys',
            visualize=False
        )
        hog_features.append(features)
    return np.array(hog_features)

# Function to load cropped patches from folders
def load_cropped_patches(folder_path, category):
    patches = []
    folder = os.path.join(folder_path, category)
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        patch = cv2.imread(file_path)
        if patch is not None:
            patches.append(patch)
    return patches

if __name__ == "__main__":
    patches_folder = "cropped"

    # Load cropped patches
    print("Loading cropped patches...")
    foreground_patches = load_cropped_patches(patches_folder, "foreground")
    background_patches = load_cropped_patches(patches_folder, "background")

    # Compute HOG features
    print("Computing HOG features...")
    foreground_features = compute_hog_features(foreground_patches)
    background_features = compute_hog_features(background_patches)

    # Prepare dataset
    X = np.vstack((foreground_features, background_features))
    y = np.hstack((np.ones(len(foreground_features)), np.zeros(len(background_features))))  # 1: Foreground, 0: Background

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Training
    print("Training SVM classifier...")
    classifier = SVC(kernel='linear', probability=True, random_state=42)
    classifier.fit(X_train, y_train)

    # Evaluate
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Save Model
    joblib.dump(classifier, "classifier_model.pkl")
    print("Classifier saved as classifier_model.pkl")
