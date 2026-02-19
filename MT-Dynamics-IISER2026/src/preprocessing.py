import numpy as np
import cv2
import skimage.restoration
import os

def load_image(path):
    # Read image from disk
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    # Convert to grayscale if RGB
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def background_subtraction(image):
    # Use rolling ball algorithm (skimage.restoration.rolling_ball)
    # Radius of 50 is a reasonable default for background
    background = skimage.restoration.rolling_ball(image, radius=50)
    # Return background corrected image
    return cv2.subtract(image, background)

def contrast_enhancement(image):
    # Use CLAHE (cv2.createCLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def gaussian_denoise(image):
    # Apply Gaussian blur (sigma=1)
    # Using kernel size (0,0) lets cv2 calculate it from sigma
    return cv2.GaussianBlur(image, (0, 0), sigmaX=1, sigmaY=1)

def preprocess_image(path):
    # Combine all steps
    img = load_image(path)
    img = background_subtraction(img)
    img = contrast_enhancement(img)
    img = gaussian_denoise(img)
    return img

if __name__ == "__main__":
    # Test block
    raw_dir = os.path.join("data", "raw")
    processed_dir = os.path.join("data", "processed")
    
    # Ensure processed directory exists
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    # Load one image from data/raw/
    if os.path.exists(raw_dir):
        files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.tif', '.jpg', '.jpeg'))]
        if files:
            sample_file = files[0]
            input_path = os.path.join(raw_dir, sample_file)
            print(f"Processing {input_path}...")
            
            result = preprocess_image(input_path)
            
            output_path = os.path.join(processed_dir, "test_preprocessed.png")
            cv2.imwrite(output_path, result)
            print(f"Saved result to {output_path}")
        else:
            print(f"No images found in {raw_dir}")
    else:
        print(f"Directory {raw_dir} does not exist.")
