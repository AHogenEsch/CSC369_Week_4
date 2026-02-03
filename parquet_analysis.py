import polars as pl
import time

DATA_FILE_PATH = 'processed_place_data.parquet'

def analyze_parquet_data():
    start_time = time.perf_counter()

    # Scan the parquet file lazily
    lf = pl.scan_parquet(DATA_FILE_PATH)
    
    # 1. Row Count (Metadata operation)
    row_count = lf.select(pl.len()).collect().item()

    # 2. Average Standard Deviation Query
    # Logic: 
    # a. Group by user
    # b. Calculate the Standard Deviation of their placement times
    # c. Calculate the Mean of those Standard Deviations
    avg_std_dev = (
        lf.group_by("user_id_int")
        .agg(
            pl.col("seconds_since_start").std().alias("user_std")
        )
        .select(
            pl.col("user_std").mean()
        )
        .collect()
        .item()
    )

    unique_user_count = (
        lf.select(pl.col("user_id_int").n_unique())
        .collect()
        .item()
    )

    end_time = time.perf_counter()
    
    print(f"Total rows: {row_count:,}")
    print(f"Average Standard Deviation across all users: {avg_std_dev:.2f} seconds")
    print(f"Unique users: {unique_user_count:,}")
    print(f"Query took: {end_time - start_time:.4f} seconds")
    

if __name__ == "__main__":
    analyze_parquet_data()