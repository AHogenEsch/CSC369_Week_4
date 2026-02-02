import sys
import polars as pl
import time
from datetime import datetime

DATA_FILE_PATH = 'processed_place_data.parquet'
EVENT_START = '2022-04-01 12:44:10' 

def analyze_rplace(start_str, end_str, file_path=DATA_FILE_PATH):
    try:
        base_time = datetime.strptime(EVENT_START, "%Y-%m-%d %H:%M:%S")
        start_dt = datetime.strptime(start_str, "%Y-%m-%d %H")
        end_dt = datetime.strptime(end_str, "%Y-%m-%d %H")
        
        if start_dt < base_time:
            print("Choose a start date after the event start (2022-04-01 12:44:10)")
            return
        if end_dt < start_dt:
            print("Usage: python PolarsPlace.py START_DATE START_HOUR END_DATE END_HOUR")
            return
        
        # converting from timestamp to seconds after event start
        start_sec = int((start_dt - base_time).total_seconds())
        end_sec = int((end_dt - base_time).total_seconds())

    except ValueError as e:
        print(f"Error parsing input dates: {e}")
        return

    start_perf = time.perf_counter_ns()

    try:
        lf = pl.scan_parquet(file_path)
        df_window = lf.filter((pl.col("seconds_since_start") >= start_sec) & 
                              (pl.col("seconds_since_start") < end_sec))

        # ---  Color Ranking by Distinct Users  ---
        # .group_by splits the data in buckets, grouped by color
        # n_unique() ignores duplicate users
        # .sort() with descending puts the highest count at [0]
        color_ranking = (
            df_window.group_by("color_name")
            .agg(pl.col("user_id_int").n_unique().alias("user_count"))
            .sort("user_count", descending=True)
            .limit(3)
            .collect()
        )

        # If this dataframe is empty then there is no point continuing
        if color_ranking.height == 0:
            print("No Data found in given timeframe")
            return
        # ---   Average Session Length  ---
        # Gap > 15 mins starts new session
        # ASSUMES TIMESTAMPS ARE CONVERTED TO SECONDS

        # unclear expected behavior if the given time window 'slices' a session, where the first part of what could be considered 
        # a session in the context of the whole data will not look like a session after filtering for the given timeframe.

        # In the case of a sliced session, if there are two or more pixels within the latter part of the session (that is within our timeframe)
        # then they will be counted as a session together, and the duration will exclude the <15 minute slice at the beginning of the timeframe.
        # If only the last pixel placement of a session is included in the timeframe, then it will not be counted as a session

        # This behaviour means that calculated session lengths within the first 15m of the timeframe 
        # may be slightly lower than reality, and overall accuracy increases with a larger timeframe.
    
        session_data = (
            # sorts first by users, then by timestamp, to group their timelines in chronological order
            df_window.sort(["user_id_int", "seconds_since_start"])
            .with_columns([
                # using shift(1) to access the previous row, find the difference, which is in seconds
                # it better be in seconds
                (pl.col("seconds_since_start") - pl.col("seconds_since_start").shift(1))
                # only compare within users. Not worried about setting null to 0 since in polars {anything - NULL} = NULL WHICH IS CONVENIENT
                # then it will fill null with 0, resulting in a diff of 0 for the first pixel of each user
                .over("user_id_int").fill_null(0).alias("diff")
            ])
            .with_columns([
                # If the difference between users' activity is larger than 15 minutes, or 900 seconds, then surely a session has just ended.
                # cast it to an int; true = 1, false = 0; then create session id by summing the column, which adds 1 for each session
                # session ids start at 0. Messages within the same session have the same ID, since the ID only changes when a session ends.
                # using .over("user_id_int") to reset the session id for each user to prevent large integers
                (pl.col("diff") > 900).cast(pl.Int32).cum_sum().over("user_id_int").alias("session_id")
            ])
            # now sort by user and their session ID
            .group_by(["user_id_int", "session_id"])
            # Find the difference of the min and max seconds for each session to find duration in seconds, 
            # and the amount of pixels (length of the array of pixels in one session)
            .agg([
                (pl.col("seconds_since_start").max() - pl.col("seconds_since_start").min()).alias("dur"),
                pl.len().alias("p_count") 
            ])
            # 'Only include cases where a user had more than one pixel placement during the time period in the average.'
            # Do not count sessions where the user placed only 1 pixel. 
            # Assume a user places two pixels
            # if the second is 900 seconds after the first, then they are in the same session, which has a duration of 900 seconds.
            # if the second is 901 seconds after  the first, then they are two different sessions with one pixel each and are not counted 
            .filter(pl.col("p_count") > 1)
            # get the average session duration
            .select(pl.col("dur").mean())
            .collect()
        )
        avg_session = session_data.item() if session_data.height > 0 else 0

        # --- Percentiles ---
        pixel_counts = (
            # Group the data by user so that all of a user's pixel placements are in the same group
            df_window.group_by("user_id_int")
            # count how many pixels they placed
            .agg(pl.len().alias("count")) 
            .collect()
        )
        quants = [0.5, 0.75, 0.9, 0.99]
        # Use quantile to find quantiles
        p_vals = [pixel_counts["count"].quantile(q) if pixel_counts.height > 0 else 0 for q in quants]

        # --- First-time Users ---
        first_time_count = (
            # group by user
            lf.group_by("user_id_int")
            # find the first pixel for each user
            .agg(pl.col("seconds_since_start").min().alias("first"))
            # filter to only give the first pixels that fall in our time window
            .filter((pl.col("first") >= start_sec) & (pl.col("first") < end_sec))
            .select(pl.len()) 
            .collect()
        ).item()

        execution_time_ms = (time.perf_counter_ns() - start_perf) // 1_000_000

        print(f"\n--- Final Results (Polars) ---")
        print(f"Timeframe: {start_str}:00 to {end_str}:00")
        print(f"Execution Time: {execution_time_ms} ms")
        print("\nRanking of Colors by distinct Users:")
        # Print the 3 top colors by distinct users withiin timeframe
        for i, row in enumerate(color_ranking.rows(), 1):
            print(f"{i}. {row[0]}: {row[1]} users")
        
        print(f"\nAverage Session Length: {avg_session:.2f} seconds")
        print(f"\nPercentiles of Pixels Placed:")
        print(f"50th Percentile: {p_vals[0]} pixels")
        print(f"75th Percentile: {p_vals[1]} pixels")
        print(f"90th Percentile: {p_vals[2]} pixels")
        print(f"99th Percentile: {p_vals[3]} pixels")
        print(f"\nCount of First-time Users: {first_time_count}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python PolarsPlace.py 2022-04-01 14 2022-04-01 17")
    else:
        analyze_rplace(f"{sys.argv[1]} {sys.argv[2]}", f"{sys.argv[3]} {sys.argv[4]}")