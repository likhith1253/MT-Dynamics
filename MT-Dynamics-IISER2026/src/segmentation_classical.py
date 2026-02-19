import cv2
import numpy as np
import os
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops
from skimage.filters import frangi
from skimage import img_as_ubyte
from preprocessing import preprocess_image

def segment_image(image):
    # Apply Frangi vesselness filter to enhance filaments
    vesselness = frangi(image)
    
    # Normalize to 0-255
    normalized = cv2.normalize(vesselness, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype(np.uint8)
    
    # Otsu thresholding on frangi output
    _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Remove small objects (< 30 pixels)
    bool_mask = binary > 0
    clean_mask = remove_small_objects(bool_mask, min_size=30)
    
    return img_as_ubyte(clean_mask)

def skeletonize_mask(binary_mask):
    bool_mask = binary_mask > 0
    skeleton = skeletonize(bool_mask)
    return img_as_ubyte(skeleton)

def label_instances(binary_mask):
    labeled_mask = label(binary_mask > 0)
    return labeled_mask


def measure_lengths(labeled_mask):
    return {p.label: p.area for p in regionprops(labeled_mask)}

def compute_iou(pred_mask, gt_mask):
    pred = pred_mask > 0
    gt = gt_mask > 0
    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)
    return intersection / union if union > 0 else 0.0

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    raw_dir = os.path.join(project_root, "data", "raw")
    gt_dir = os.path.join(project_root, "data", "ground_truth")
    results_dir = os.path.join(project_root, "results")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    iou_scores = []
    
    if os.path.exists(raw_dir) and os.path.exists(gt_dir):
        files = sorted([f for f in os.listdir(raw_dir) if f.lower().endswith('.png')])
        # Evaluate over all 50 saved samples
        files = files[:50]
        
        print(f"dict_keys(['processing', 'segmentation', 'evaluation'])")
        print(f"Processing {len(files)} images...")
        
        for filename in files:
            img_path = os.path.join(raw_dir, filename)
            
            # Preprocess & Segment
            processed = preprocess_image(img_path)
            binary_mask = segment_image(processed)
            
            # Ground Truth
            gt_filename = filename.replace("sample", "mask")
            gt_path = os.path.join(gt_dir, gt_filename)
            
            if os.path.exists(gt_path):
                gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt_mask is not None:
                    score = compute_iou(binary_mask, gt_mask)
                    iou_scores.append(score)
        
        if iou_scores:
            mean_iou = np.mean(iou_scores)
            std_iou = np.std(iou_scores)
            
            print(f"Mean IoU: {mean_iou:.4f}")
            print(f"Std IoU: {std_iou:.4f}")
            
            with open(os.path.join(results_dir, "batch_iou.txt"), "w") as f:
                for s in iou_scores:
                    f.write(f"{s:.4f}\n")
        else:
            print("No valid IoU scores computed.")
    else:
        print("Data directories not found.")
