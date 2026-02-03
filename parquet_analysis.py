import polars as pl
import time

DATA_FILE_PATH = 'processed_place_data.parquet'

def count_parquet_rows():
    start_time = time.perf_counter()

    # Scan the metadata without reading the actual pixel data
    lf = pl.scan_parquet(DATA_FILE_PATH)
    
    # Select the length and collect the result
    row_count = lf.select(pl.len()).collect().item()

    end_time = time.perf_counter()
    
    print(f"Total rows: {row_count:,}")
    print(f"Query took: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    count_parquet_rows()