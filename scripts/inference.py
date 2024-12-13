import cv2
import numpy as np
from skimage.feature import hog
import joblib

# Compute HOG features for a single patch
def compute_hog_feature(patch):
    patch = cv2.resize(patch, (32, 32))  # Ensure patch is 32x32
    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray_patch,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        orientations=9,
        block_norm='L2-Hys',
        visualize=False
    )
    return features.reshape(1, -1)

# Callback function for mouse events
def classify_pixel(event, x, y, flags, param):
    global model, image, crop_size

    if event == cv2.EVENT_LBUTTONDOWN:  # Left-click to classify
        # Extract a patch centered at the clicked location
        x_start = max(0, x - crop_size // 2)
        y_start = max(0, y - crop_size // 2)
        x_end = min(image.shape[1], x + crop_size // 2)
        y_end = min(image.shape[0], y + crop_size // 2)
        patch = image[y_start:y_end, x_start:x_end]

        if patch.shape[0] == crop_size and patch.shape[1] == crop_size:
            # Compute HOG features
            features = compute_hog_feature(patch)
            # Classify using the trained model
            prediction = model.predict(features)[0]
            label = "Foreground (Bird)" if prediction == 1 else "Background"
            print(f"Location ({x}, {y}) classified as: {label}")

            # Display the classification result as a circle
            color = (0, 255, 0) if prediction == 1 else (0, 0, 255)  # Green for bird, Red for background
            cv2.circle(image, (x, y), 5, color, -1)
            cv2.imshow("Classify Pixels", cv2.resize(image, (800,600), interpolation=cv2.INTER_AREA))

if __name__ == "__main__":
    # Load the trained model
    model_path = "classifier_model.pkl"
    try:
        model = joblib.load(model_path)
        print("Trained model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        exit()

    # Load the image for classification
    file_path = input("Enter the image file path: ")
    image = cv2.imread(file_path)
    if image is None:
        print("Error: Could not load image.")
        exit()

    # Set the crop size for patches
    crop_size = int(input("Enter patch crop size (default 32): ") or 32)

    # Display the image and set up the mouse callback for classification
    cv2.imshow("Classify Pixels", cv2.resize(image, (800,600), interpolation=cv2.INTER_AREA))
    cv2.setMouseCallback("Classify Pixels", classify_pixel)

    print("Left-click on the image to classify pixels.")
    print("Press 'q' to exit.")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
