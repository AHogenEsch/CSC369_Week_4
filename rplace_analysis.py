import polars as pl
import time

DATA_FILE_PATH = '2022_place_canvas_history.csv'

def analyze_moderation_actions(file_path=DATA_FILE_PATH):
    print(f"Scanning {file_path} for moderation actions...")
    start_time = time.perf_counter()

    try:
        # Scan the CSV. Note: Polars handles quoted fields with commas 
        # automatically, but we want to look inside that specific string.
        lf = pl.scan_csv(file_path)

        # 1. Identify Moderation Actions
        # Logic: Count commas in the 'coordinate' string. 
        # A normal pixel is "x,y" (1 comma). 
        # A moderator rectangle is "x1,y1,x2,y2" (3 commas).
        analysis = (
            lf.select([
                pl.all(),
                pl.col("coordinate").str.count_matches(",").alias("comma_count")
            ])
            .with_columns(
                pl.col("comma_count").ge(3).alias("is_mod_action")
            )
        )

        # 2. Collect Totals
        # We execute two paths: one for the full count, one for the mod subset
        results = analysis.select([
            pl.len().alias("total_rows"),
            pl.col("is_mod_action").sum().alias("mod_count"),
            pl.col("user_id").filter(pl.col("is_mod_action")).n_unique().alias("unique_mod_users")
        ]).collect()

        total_rows = results["total_rows"][0]
        mod_count = results["mod_count"][0]
        unique_mod_users = results["unique_mod_users"][0]

        # 3. Get 10 examples of moderation coordinates
        examples = (
            analysis.filter(pl.col("is_mod_action"))
            .select("coordinate")
            .limit(10)
            .collect()
        )

        # 4. Calculations
        percentage = (mod_count / total_rows * 100) if total_rows > 0 else 0
        execution_time = time.perf_counter() - start_time

        # --- Output Results ---
        print("\n" + "="*40)
        print("MODERATION ACTION REPORT")
        print("="*40)
        print(f"Total Rows Processed:      {total_rows:,}")
        print(f"Total Moderation Actions:  {mod_count:,}")
        print(f"Unique Admins/Mods:        {unique_mod_users:,}") # Added this line
        print(f"Percentage of Total:       {percentage:.6f}%")
        print(f"Execution Time:            {execution_time:.2f} seconds")
        
        print("\nExample Moderation Coordinates:")
        if examples.height > 0:
            for i, coord in enumerate(examples["coordinate"], 1):
                print(f"{i}. {coord}")
        else:
            print("No moderation actions found.")
        print("="*40)

    except Exception as e:
        print(f"An error occurred: {e}")

def analyze_coordinate_commas(file_path=DATA_FILE_PATH):
    print(f"Executing strict comma-count analysis on {file_path}...")
    start_time = time.perf_counter()

    try:
        lf = pl.scan_csv(file_path)

        # Create a temporary column that counts the commas for every single row
        analysis = lf.select([
            pl.len().alias("total_rows"),
            # Count occurrences of 0, 1, 2, and 3 commas strictly
            pl.col("coordinate").str.count_matches(",").eq(0).sum().alias("zero_commas"),
            pl.col("coordinate").str.count_matches(",").eq(1).sum().alias("one_comma"),
            pl.col("coordinate").str.count_matches(",").eq(2).sum().alias("two_commas"),
            pl.col("coordinate").str.count_matches(",").eq(3).sum().alias("three_commas")
        ]).collect()

        total = analysis["total_rows"][0]
        
        # Helper to format output
        def print_stat(label, count):
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{label:15} {count:12,} rows ({percentage:6.2f}%)")

        # Get one real example for each category to help  visualize what Excel is hiding
        examples = {}
        for i in range(4):
            ex = lf.filter(pl.col("coordinate").str.count_matches(",").eq(i)).select("coordinate").limit(1).collect()
            examples[i] = ex["coordinate"][0] if ex.height > 0 else "N/A"

        execution_time = time.perf_counter() - start_time

        print("\n" + "="*55)
        print("STRICT COORDINATE COMMA-COUNT REPORT")
        print("="*55)
        print(f"Total Rows Processed: {total:,}")
        print("-" * 55)
        
        print_stat("0 Commas:", analysis["zero_commas"][0])
        print(f"   Example: {examples[0]}")
        
        print_stat("1 Comma:", analysis["one_comma"][0])
        print(f"   Example: {examples[1]}")
        
        print_stat("2 Commas:", analysis["two_commas"][0])
        print(f"   Example: {examples[2]}")
        
        print_stat("3 Commas:", analysis["three_commas"][0])
        print(f"   Example: {examples[3]}")
        
        print("-" * 55)
        print(f"Execution Time: {execution_time:.2f} seconds")
        print("="*55)

    except Exception as e:
        print(f"An error occurred during analysis: {e}")


if __name__ == "__main__":
    analyze_moderation_actions()
    analyze_coordinate_commas()