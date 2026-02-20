import pandas as pd
import os

def convert_to_physical():
    input_path = "../results/dynamics_summary.csv"
    output_path = "../results/dynamics_summary_physical_units.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    # 1. Ask user to input (using placeholders if running in non-interactive environment)
    print("Welcome to Physical Units Converter.")
    try:
        scale_bar_pixels = float(input("Enter scale bar length in pixels: "))
        scale_bar_um = float(input("Enter scale bar length in micrometers: "))
    except ValueError:
        print("Invalid input. Please enter numbers.")
        return

    # 2. Compute conversion factor
    um_per_pixel = scale_bar_um / scale_bar_pixels
    print(f"Conversion factor: {um_per_pixel:.4f} um/px")

    # 3. Load results
    df = pd.read_csv(input_path)

    # 4. Convert units
    # mean_growth_velocity, mean_shrink_velocity, dynamicity
    df["mean_growth_velocity_um_s"] = df["mean_growth_velocity"] * um_per_pixel
    df["mean_shrink_velocity_um_s"] = df["mean_shrink_velocity"] * um_per_pixel
    df["dynamicity_um_s"] = df["dynamicity"] * um_per_pixel

    # 5. Save as
    df.to_csv(output_path, index=False)
    
    print("\n--- Converted Summary (Physical Units: um/s) ---")
    cols = ["track_id", "mean_growth_velocity_um_s", "mean_shrink_velocity_um_s", "dynamicity_um_s"]
    print(df[cols].to_string(index=False))
    print(f"\nSaved physical units summary to {output_path}")

if __name__ == "__main__":
    convert_to_physical()
