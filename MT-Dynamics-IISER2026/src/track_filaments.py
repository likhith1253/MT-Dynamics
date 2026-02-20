import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import os

def track_filaments():
    input_path = "../results/frame_features.csv"
    output_path = "../results/filament_tracks.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # 1. Load data
    df = pd.read_csv(input_path)
    
    # 2. Sort by frame_index
    df = df.sort_values("frame_index").reset_index(drop=True)
    
    # Max distance threshold
    max_dist = 20.0
    
    # Add track_id column, initialize with -1
    df["track_id"] = -1
    
    frames = sorted(df["frame_index"].unique())
    next_track_id = 0
    
    # 3. Tracking logic
    for i in range(len(frames)):
        current_frame = frames[i]
        curr_mask = df["frame_index"] == current_frame
        curr_data = df[curr_mask]
        
        # Assign new track IDs to unmatched components in current frame (initially frame 0)
        for idx in curr_data.index:
            if df.at[idx, "track_id"] == -1:
                df.at[idx, "track_id"] = next_track_id
                next_track_id += 1
        
        # If not the last frame, match to next
        if i < len(frames) - 1:
            next_frame = frames[i+1]
            next_mask = df["frame_index"] == next_frame
            next_data = df[next_mask]
            
            if len(curr_data) > 0 and len(next_data) > 0:
                # Compute distance matrix
                curr_coords = curr_data[["centroid_x", "centroid_y"]].values
                next_coords = next_data[["centroid_x", "centroid_y"]].values
                
                dists = cdist(curr_coords, next_coords)
                
                # Simple greedy matching
                # Find smallest distance, match, then zero out row/col
                temp_dists = dists.copy()
                for _ in range(min(len(curr_data), len(next_data))):
                    min_idx = np.unravel_index(np.argmin(temp_dists), temp_dists.shape)
                    if temp_dists[min_idx] > max_dist:
                        break
                    
                    # Match found
                    curr_row_idx = curr_data.index[min_idx[0]]
                    next_row_idx = next_data.index[min_idx[1]]
                    
                    df.at[next_row_idx, "track_id"] = df.at[curr_row_idx, "track_id"]
                    
                    # Remove from further matching
                    temp_dists[min_idx[0], :] = np.inf
                    temp_dists[:, min_idx[1]] = np.inf
                    
    # 4. Save results
    tracked_df = df[["track_id", "frame_index", "length", "centroid_x", "centroid_y"]]
    tracked_df = tracked_df.sort_values(["track_id", "frame_index"])
    tracked_df.to_csv(output_path, index=False)
    
    print(f"Total tracks created: {next_track_id}")
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    track_filaments()
