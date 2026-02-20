import pandas as pd
import numpy as np
import os

def compute_dynamics():
    input_path = "../results/filament_tracks.csv"
    output_path = "../results/dynamics_summary.csv"
    fps = 8
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # 1. Load data
    df = pd.read_csv(input_path)
    
    summary_list = []
    track_ids = sorted(df["track_id"].unique())
    
    # 3. For each track_id:
    for tid in track_ids:
        track = df[df["track_id"] == tid].sort_values("frame_index").copy()
        
        if len(track) < 2:
            continue
            
        # Compute time and velocity
        track["time"] = track["frame_index"] / fps
        dt = track["time"].diff().values[1:]
        dl = track["length"].diff().values[1:]
        velocities = dl / dt
        
        # Classify states
        # growth: v > 5, shrink: v < -5, else pause
        states = []
        for v in velocities:
            if v > 5:
                states.append("growth")
            elif v < -5:
                states.append("shrink")
            else:
                states.append("pause")
        
        # Count catastrophe (growth -> shrink) and rescue (shrink -> growth)
        catastrophes = 0
        rescues = 0
        for i in range(len(states) - 1):
            if states[i] == "growth" and states[i+1] == "shrink":
                catastrophes += 1
            if states[i] == "shrink" and states[i+1] == "growth":
                rescues += 1
                
        # Compute mean velocities
        growth_vs = velocities[np.array(states) == "growth"]
        shrink_vs = velocities[np.array(states) == "shrink"]
        
        mean_growth = np.mean(growth_vs) if len(growth_vs) > 0 else 0
        mean_shrink = np.mean(shrink_vs) if len(shrink_vs) > 0 else 0
        
        # Compute dynamicity
        total_time = track["time"].iloc[-1] - track["time"].iloc[0]
        if total_time > 0:
            dynamicity = np.sum(np.abs(dl)) / total_time
        else:
            dynamicity = 0
            
        summary_list.append({
            "track_id": tid,
            "mean_growth_velocity": mean_growth,
            "mean_shrink_velocity": mean_shrink,
            "catastrophes": catastrophes,
            "rescues": rescues,
            "dynamicity": dynamicity,
            "total_frames": len(track)
        })
        
    # 4. Save summary
    summary_df = pd.DataFrame(summary_list)
    summary_df.to_csv(output_path, index=False)
    
    # 5. Print summary table
    print("\n--- Dynamics Summary Table ---")
    pd.set_option('display.max_rows', 20)
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to {output_path}")

if __name__ == "__main__":
    compute_dynamics()
