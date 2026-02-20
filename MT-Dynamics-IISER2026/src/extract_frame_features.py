import cv2
import numpy as np
from skimage.measure import label, regionprops
import os
import csv

def extract_features():
    # Paths
    input_dir = "../data/real_skeletons/"
    output_path = "../results/frame_features.csv"
    
    # Ensure results directory exists
    os.makedirs("../results/", exist_ok=True)
    
    # Get all skeleton frames
    if not os.path.exists(input_dir):
        print(f"Error: Directory {input_dir} not found.")
        return
        
    frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.tif'))])
    
    results = []
    print(f"Processing {len(frame_files)} frames...")
    
    for i, filename in enumerate(frame_files):
        # Load frame (assume grayscale binary skeleton)
        frame_path = os.path.join(input_dir, filename)
        skeleton = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        
        if skeleton is None:
            continue
            
        # Label connected components
        labeled_mask = label(skeleton > 0)
        props = regionprops(labeled_mask)
        
        for prop in props:
            # frame_index, component_id, length, centroid_x, centroid_y
            results.append([
                i, 
                prop.label, 
                prop.area, 
                prop.centroid[1], # x (column)
                prop.centroid[0]  # y (row)
            ])
            
    # Save results to CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "component_id", "length", "centroid_x", "centroid_y"])
        writer.writerows(results)
        
    print(f"Successfully saved results to {output_path}")

if __name__ == "__main__":
    extract_features()
