import polars as pl
import pyvista as pv
import numpy as np
import sys
import time

# --- CONFIGURATION ---
# We use the PROCESSED file because it has cleaned coordinates
DATA_FILE_PATH = 'processed_place_data.parquet' 
RPLACE_WIDTH = 2000
RPLACE_HEIGHT = 2000

# Color Palette Mapping (Hex -> RGB Tuple 0-255)
# This converts color names from the processed parquet back to RGB for display
COLOR_NAME_TO_RGB = {
    "dark red": (109, 0, 26), "red": (190, 0, 57), "orange": (255, 69, 0), "yellow": (255, 168, 0),
    "pale yellow": (255, 214, 53), "ivory": (255, 248, 184), "dark green": (0, 163, 104), "green": (0, 204, 120),
    "light green": (126, 237, 86), "dark teal": (0, 117, 111), "teal": (0, 158, 170), "light teal": (0, 204, 192),
    "dark blue": (36, 80, 164), "blue": (54, 144, 234), "light blue": (81, 233, 244), "indigo": (73, 58, 193),
    "periwinkle": (106, 92, 255), "lavender": (148, 179, 255), "dark purple": (129, 30, 159), "purple": (180, 74, 192),
    "pale purple": (228, 171, 255), "magenta": (222, 16, 127), "pink": (255, 56, 129), "light pink": (255, 153, 170),
    "dark brown": (109, 72, 47), "brown": (156, 105, 38), "beige": (255, 180, 112), "black": (0, 0, 0),
    "dark gray": (81, 82, 82), "gray": (137, 141, 144), "light gray": (212, 215, 217), "white": (255, 255, 255),
    "unknown": (0, 0, 0)
}

def visualize_linear_bots():
    print(f"--- Visualizing Linear Users from: {DATA_FILE_PATH} ---")
    start_t = time.perf_counter()

    # 1. Load Data
    lf = pl.scan_parquet(DATA_FILE_PATH)
    
    # Filter out users with very low activity 
    # We need a window context, so we filter basic count first
    lf = lf.filter(pl.len().over("user_id_int") > 10)

    # 2. Identify Linear Users
    print("  > Calculating linearity scores (dx/dy)...")
    
    linear_users = (
        lf.sort(["user_id_int", "seconds_since_start"])
        .with_columns([
            (pl.col("x") - pl.col("x").shift(1)).over("user_id_int").alias("dx"),
            (pl.col("y") - pl.col("y").shift(1)).over("user_id_int").alias("dy")
        ])
        .with_columns(
            ((pl.col("dx").abs() == 1) & (pl.col("dy") == 0) | 
             (pl.col("dx") == 0) & (pl.col("dy").abs() == 1))
            .cast(pl.Int32).alias("is_linear")
        )
        .group_by("user_id_int")
        .agg(pl.col("is_linear").mean().alias("linearity_score"))
        .filter(pl.col("linearity_score") > 0.95)
        .select("user_id_int") # Just keep the IDs
    )
    
    # 3. Filter Main Data for ONLY Linear Users
    print("  > Filtering dataset for identified linear bots...")
    
    # Join back to get all pixels placed by these users
    target_pixels = (
        lf.join(linear_users, on="user_id_int", how="inner")
        .select(["x", "y", "color_name"])
    )

    # 4. Aggregation: Get HEIGHT and FINAL COLOR per pixel
    print("  > Aggregating pixel history...")
    
    final_state = (
        target_pixels
        .group_by(["x", "y"])
        .agg([
            pl.len().alias("height"),
            pl.col("color_name").last().alias("final_color")
        ])
        .collect()
    )

    user_count = linear_users.collect().height
    pixel_count = final_state.height
    print(f"  > Found {user_count} linear users responsible for {pixel_count} unique pixels.")
    print(f"  > Data processed in {time.perf_counter() - start_t:.2f}s.")

    # --- 5. Constructing the Visualization ---
    print("  > Building 3D Geometry...")

    # Initialize Grids
    height_grid = np.zeros((RPLACE_WIDTH, RPLACE_HEIGHT))
    # Default background to white so colors can pop
    color_grid = np.full((RPLACE_WIDTH, RPLACE_HEIGHT, 3), 255, dtype=np.uint8)

    # Filter Valid Bounds
    valid = final_state.filter(
        (pl.col("x") < RPLACE_WIDTH) & (pl.col("y") < RPLACE_HEIGHT)
    )

    # Map Data to Arrays
    xs = valid["x"].to_numpy()
    ys = valid["y"].to_numpy()
    zs = valid["height"].to_numpy()
    colors = valid["final_color"].to_list()

    # Set Heights
    height_grid[xs, ys] = zs

    # Set Colors
    rgb_values = np.array([COLOR_NAME_TO_RGB.get(c, (255,255,255)) for c in colors], dtype=np.uint8)
    color_grid[xs, ys] = rgb_values

    # Create PyVista Mesh
    grid = pv.ImageData(dimensions=(RPLACE_WIDTH, RPLACE_HEIGHT, 1))
    grid.point_data["RGB"] = color_grid.swapaxes(0, 1).reshape(-1, 3)
    grid.point_data["Elevation"] = height_grid.T.flatten()

    # Warp by Scalar (Height represents traffic intensity)
    # Using a higher factor (0.5) here because linear bots often overlap heavily
    warped = grid.warp_by_scalar("Elevation", factor=0.5)

    # --- 6. Rendering ---
    plotter = pv.Plotter()
    plotter.add_mesh(warped, rgb=True, show_scalar_bar=False)
    
    plotter.add_scalar_bar(title="Linear Bot Overlaps", mapper=None)
    plotter.show_grid()
    plotter.add_text(f"Linear 'Printer' Bots ({user_count} Users)", position='upper_left', font_size=12)
    
    # Set camera to isometric view
    plotter.camera_position = 'iso'
    
    print("  > Opening Interactive Window...")
    plotter.show()

if __name__ == "__main__":
    visualize_linear_bots()