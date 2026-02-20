import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_dynamics():
    tracks_path = "../results/filament_tracks.csv"
    summary_path = "../results/dynamics_summary.csv"
    output_dir = "../results/"
    fps = 8
    
    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load filament_tracks.csv
    if not os.path.exists(tracks_path):
        print(f"Error: {tracks_path} not found.")
        return
        
    df_tracks = pd.read_csv(tracks_path)
    
    # 2. Select track 0
    track0 = df_tracks[df_tracks["track_id"] == 0].sort_values("frame_index")
    
    if len(track0) == 0:
        print("Error: Track 0 not found.")
        return

    # 3. Compute time
    time = track0["frame_index"] / fps
    length = track0["length"]
    
    # 4. Plot Length vs Time
    plt.figure(figsize=(10, 6))
    plt.plot(time, length, marker='o', linestyle='-', color='b')
    plt.title("Track 0: Length vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Length (pixels)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "track0_length_time.png"))
    plt.close()
    print(f"Saved {output_dir}track0_length_time.png")

    # Smoothed velocity calculation
    length_smooth = length.rolling(window=3, center=True).mean()
    # Filling NaN values caused by rolling mean at the edges to allow gradient calculation
    length_smooth = length_smooth.ffill().bfill()
    velocity = np.gradient(length_smooth, time)
    
    # Plot Velocity vs Time
    plt.figure(figsize=(10, 6))
    plt.plot(time, velocity, marker='x', linestyle='--', color='r')
    plt.axhline(0, color='k', linewidth=0.5)
    plt.title("Track 0: Velocity vs Time (Smoothed)")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (pixels/s)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "track0_velocity_time.png"))
    plt.close()
    print(f"Saved {output_dir}track0_velocity_time.png")

    # 6. Load dynamics_summary.csv and plot histogram
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found.")
        return
        
    df_summary = pd.read_csv(summary_path)
    
    # Filter tracks with > 20 frames
    long_tracks = df_summary[df_summary["total_frames"] > 20]
    
    if len(long_tracks) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(long_tracks["mean_growth_velocity"], bins=10, color='green', alpha=0.7, edgecolor='black')
        plt.title("Histogram of Mean Growth Velocity (Tracks > 20 frames)")
        plt.xlabel("Mean Growth Velocity (pixels/s)")
        plt.ylabel("Count")
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(output_dir, "growth_velocity_hist.png"))
        plt.close()
        print(f"Saved {output_dir}growth_velocity_hist.png")
    else:
        print("No tracks with > 20 frames found for histogram.")

if __name__ == "__main__":
    plot_dynamics()
