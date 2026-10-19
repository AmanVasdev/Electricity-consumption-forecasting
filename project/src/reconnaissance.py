"""
Dataset Reconnaissance Script for acad_build_mains.csv
Uses chunked reading to efficiently inspect large CSV datasets without loading the full file into memory.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

RAW_FILE = r"energy_dataset\energy_dataset\acad_build_mains.csv"
SAMPLE_FILE = r"project\data\processed\acad_sample.csv"
CHUNK_SIZE = 100000

def run_reconnaissance():
    if not os.path.exists(RAW_FILE):
        print(f"Error: File not found: {RAW_FILE}")
        sys.exit(1)

    # 1. File size
    file_size_bytes = os.path.getsize(RAW_FILE)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"1. File size: {file_size_bytes:,} bytes ({file_size_mb:.2f} MB)")

    # Read first 5 rows and headers
    df_head = pd.read_csv(RAW_FILE, nrows=5)
    columns = list(df_head.columns)
    print(f"2. Column names: {columns}")
    print("3. Inferred data types (first 5 rows):")
    for col, dtype in df_head.dtypes.items():
        print(f"   - {col}: {dtype}")
    print("\n4. First 5 rows:")
    print(df_head.to_string())

    # We will also create the sample of 10,000 rows
    print(f"\nCreating sample file ({SAMPLE_FILE}) with 10,000 rows...")
    df_sample = pd.read_csv(RAW_FILE, nrows=10000)
    df_sample.to_csv(SAMPLE_FILE, index=False)
    print(f"Sample file created successfully at {SAMPLE_FILE} ({len(df_sample)} rows).")

    # Metrics to accumulate across chunks
    total_rows = 0
    missing_counts = {col: 0 for col in columns}
    
    # Running stats for numerical columns
    # We will track count, sum, sum of squares, min, max
    num_cols = [c for c in columns if c != "timestamp"]
    stats = {
        c: {
            "count": 0,
            "sum": 0.0,
            "sum_sq": 0.0,
            "min": float("inf"),
            "max": float("-inf")
        } for c in num_cols
    }

    min_timestamp = None
    max_timestamp = None
    prev_last_timestamp = None
    
    # Track timestamp delta distribution
    delta_counts = {}
    timestamp_non_monotonic_count = 0
    duplicate_timestamps_count = 0
    exact_duplicate_rows_count = 0

    print("\nProcessing file in chunks of", CHUNK_SIZE, "rows...")
    
    # We will read chunks
    # Note: NA values like 'NA' in csv
    chunk_idx = 0
    for chunk in pd.read_csv(RAW_FILE, chunksize=CHUNK_SIZE, na_values=["NA", "null", "NaN", ""]):
        chunk_idx += 1
        n_rows = len(chunk)
        total_rows += n_rows

        # Missing counts
        for col in columns:
            missing_counts[col] += int(chunk[col].isna().sum())

        # Check duplicate rows within chunk
        exact_duplicate_rows_count += int(chunk.duplicated().sum())

        # Timestamp tracking
        ts_series = chunk["timestamp"]
        chunk_min_ts = ts_series.min()
        chunk_max_ts = ts_series.max()

        if min_timestamp is None or chunk_min_ts < min_timestamp:
            min_timestamp = chunk_min_ts
        if max_timestamp is None or chunk_max_ts > max_timestamp:
            max_timestamp = chunk_max_ts

        # Timestamp monotonicity and delta
        # Check boundary from previous chunk
        if prev_last_timestamp is not None:
            boundary_delta = ts_series.iloc[0] - prev_last_timestamp
            delta_counts[boundary_delta] = delta_counts.get(boundary_delta, 0) + 1
            if boundary_delta < 0:
                timestamp_non_monotonic_count += 1
            elif boundary_delta == 0:
                duplicate_timestamps_count += 1

        prev_last_timestamp = ts_series.iloc[-1]

        # In-chunk deltas
        deltas = ts_series.diff().dropna().astype(int)
        val_counts = deltas.value_counts()
        for d, count in val_counts.items():
            delta_counts[d] = delta_counts.get(d, 0) + count
            if d < 0:
                timestamp_non_monotonic_count += count
            elif d == 0:
                duplicate_timestamps_count += count

        # Numerical stats
        for c in num_cols:
            s = chunk[c].dropna()
            if len(s) > 0:
                s_count = len(s)
                s_sum = float(s.sum())
                s_sum_sq = float((s ** 2).sum())
                s_min = float(s.min())
                s_max = float(s.max())

                st = stats[c]
                st["count"] += s_count
                st["sum"] += s_sum
                st["sum_sq"] += s_sum_sq
                if s_min < st["min"]:
                    st["min"] = s_min
                if s_max > st["max"]:
                    st["max"] = s_max

        if chunk_idx % 5 == 0:
            print(f"Processed {total_rows:,} rows...")

    print(f"Finished processing all {total_rows:,} rows across {chunk_idx} chunks.\n")

    # Timezone conversion
    tz_kolkata = timezone(timedelta(hours=5, minutes=30))
    min_dt_utc = datetime.fromtimestamp(min_timestamp, tz=timezone.utc)
    min_dt_kolkata = datetime.fromtimestamp(min_timestamp, tz=tz_kolkata)
    max_dt_utc = datetime.fromtimestamp(max_timestamp, tz=timezone.utc)
    max_dt_kolkata = datetime.fromtimestamp(max_timestamp, tz=tz_kolkata)

    print("=== SUMMARY FINDINGS ===")
    print(f"Total Rows: {total_rows:,}")
    print(f"File Size: {file_size_bytes:,} bytes ({file_size_mb:.2f} MB)")
    print(f"Timestamp range:")
    print(f"  Min timestamp: {min_timestamp} -> UTC: {min_dt_utc.isoformat()} | Kolkata: {min_dt_kolkata.isoformat()}")
    print(f"  Max timestamp: {max_timestamp} -> UTC: {max_dt_utc.isoformat()} | Kolkata: {max_dt_kolkata.isoformat()}")
    
    timespan = max_dt_utc - min_dt_utc
    print(f"  Total timespan: {timespan.days} days, {timespan.seconds // 3600} hours")

    print("\nSampling Delays / Frequency (Top 10):")
    sorted_deltas = sorted(delta_counts.items(), key=lambda x: x[1], reverse=True)
    for delta, count in sorted_deltas[:10]:
        pct = (count / (total_rows - 1)) * 100
        print(f"  Delta = {delta}s ({delta/60:.2f} min): {count:,} times ({pct:.2f}%)")

    print(f"\nMonotonicity & Duplicates:")
    print(f"  Duplicate Timestamps: {duplicate_timestamps_count:,}")
    print(f"  Non-monotonic timestamp steps: {timestamp_non_monotonic_count:,}")
    print(f"  Exact duplicate rows (within chunks): {exact_duplicate_rows_count:,}")

    print("\nMissing values per column:")
    for col, cnt in missing_counts.items():
        pct = (cnt / total_rows) * 100
        print(f"  {col}: {cnt:,} missing ({pct:.2f}%)")

    print("\nNumerical Statistics:")
    for col, st in stats.items():
        cnt = st["count"]
        if cnt > 0:
            mean = st["sum"] / cnt
            variance = (st["sum_sq"] / cnt) - (mean ** 2)
            std = np.sqrt(max(0.0, variance))
            print(f"  {col}:")
            print(f"    Count: {cnt:,}")
            print(f"    Mean: {mean:.4f}")
            print(f"    Std:  {std:.4f}")
            print(f"    Min:  {st['min']:.4f}")
            print(f"    Max:  {st['max']:.4f}")
        else:
            print(f"  {col}: All values are missing/NA (Count: 0)")

    # Save summary as JSON for reporting
    results = {
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_mb,
        "total_rows": total_rows,
        "columns": columns,
        "inferred_dtypes": {k: str(v) for k, v in df_head.dtypes.items()},
        "head_5": df_head.to_dict(orient="records"),
        "min_timestamp": int(min_timestamp),
        "min_datetime_utc": min_dt_utc.isoformat(),
        "min_datetime_kolkata": min_dt_kolkata.isoformat(),
        "max_timestamp": int(max_timestamp),
        "max_datetime_utc": max_dt_utc.isoformat(),
        "max_datetime_kolkata": max_dt_kolkata.isoformat(),
        "timespan_days": timespan.days,
        "missing_counts": missing_counts,
        "missing_percentages": {k: (v / total_rows) * 100 for k, v in missing_counts.items()},
        "duplicate_timestamps": duplicate_timestamps_count,
        "non_monotonic_steps": timestamp_non_monotonic_count,
        "exact_duplicate_rows_in_chunks": exact_duplicate_rows_count,
        "top_deltas": [{"delta_seconds": int(k), "count": int(v), "percentage": (v / (total_rows - 1)) * 100} for k, v in sorted_deltas[:10]],
        "numerical_stats": {
            k: {
                "count": st["count"],
                "mean": (st["sum"] / st["count"]) if st["count"] > 0 else None,
                "std": float(np.sqrt(max(0.0, (st["sum_sq"] / st["count"]) - ((st["sum"] / st["count"]) ** 2)))) if st["count"] > 0 else None,
                "min": st["min"] if st["count"] > 0 else None,
                "max": st["max"] if st["count"] > 0 else None
            } for k, st in stats.items()
        }
    }

    with open(r"project\reports\recon_raw_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRaw JSON summary saved to project/reports/recon_raw_summary.json")

if __name__ == "__main__":
    run_reconnaissance()
