import sys
import polars as p2
import time
from datetime import datetime

DATA_FILE_PATH = '2022_place_canvas_history'
EVENT_START = '2022-04-01 12:44:10' 

def detect_bots(file_path=DATA_FILE_PATH):
    # "Bots" Must have placed at least 10 pixels to be considered active enough to warrant detection.
    start_perf = time.perf.counter_ns()


    try:
        lf = p2.scan_parquet(file_path)
        # filter out users who have placed less than 10 pixels.
        df_window = lf.filter()


        # Find users who are precisely placing pixels with very little time standard deviation
        no_time_deviation = ()
        # Find users who are consistently active (at least 1 pixel an hour) for 24 or more hours
        no_sleep = ()
        # Find users who place pixels linearly, like horizontally or vertically, without deviation
        linear_placement = ()
        # Find users who only place pixels within a small area (3x3?)
        low_coordinate_deviation = ()
        # Find users who only placed a single color.
        mono_chromatic = ()
        # Find coordinated users, high volume of localized placements in short period of time
        botnet_users = ()


        execution_time_ms = (time.perf_counter_ns() - start_perf) // 1_000_000


        printf("Analysis excludes users with fewer than 10 pixel placements")
        print(f"Number of users with little time variance: {no_time_deviation.height}")
        print(f"Number of users with 24+ hours of consecutive activity: {no_sleep.height}")
        print(f"Number of users with strictly linear placements: {linear_placement.height}")
        print(f"Number of users who placed pixels in few coordinates: {low_coordinate_deviation.height}")
        print(f"Number of users who only placed a single color: {mono_chromatic.height}")
        print(f"Number of users who were part of large coordinated efforts: {botnet_users.height}")
        print(f"Execution Time: {execution_time_ms} ms")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
