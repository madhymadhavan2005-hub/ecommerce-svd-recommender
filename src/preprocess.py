"""
Data Preprocessing Pipeline for E-Commerce Collaborative Filtering
Cleans raw transaction logs, aggregates purchase counts, scales implicit feedback
to a [1.0, 5.0] rating interval, and exports user-item matrices.
"""

import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "transactions.csv")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
USER_ITEM_MATRIX_PATH = os.path.join(PROCESSED_DATA_DIR, "user_item_matrix.csv")
SURPRISE_RATINGS_PATH = os.path.join(PROCESSED_DATA_DIR, "surprise_ratings.csv")

def preprocess_pipeline(raw_csv_path=RAW_DATA_PATH, output_dir=PROCESSED_DATA_DIR):
    """
    Executes the full data preparation pipeline:
    1. Null value removal on CustomerID and ProductID.
    2. Deduplication across [InvoiceNo, CustomerID, ProductID].
    3. Aggregation of Quantity into purchase_count.
    4. Implicit rating scaling into [1.0, 5.0].
    5. Export of user_item_matrix.csv and surprise_ratings.csv.
    """
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw data file not found at '{raw_csv_path}'. Run generate_mock_data.py first.")

    os.makedirs(output_dir, exist_ok=True)
    matrix_path = os.path.join(output_dir, "user_item_matrix.csv")
    ratings_path = os.path.join(output_dir, "surprise_ratings.csv")

    raw_df = pd.read_csv(raw_csv_path)
    initial_count = len(raw_df)

    # 1. Null Handling: Drop any row where CustomerID or ProductID is NaN or empty string
    clean_df = raw_df.dropna(subset=["CustomerID", "ProductID"]).copy()
    clean_df["CustomerID"] = clean_df["CustomerID"].astype(str).str.strip()
    clean_df["ProductID"] = clean_df["ProductID"].astype(str).str.strip()
    clean_df = clean_df[(clean_df["CustomerID"] != "") & (clean_df["ProductID"] != "")]
    after_null_drop = len(clean_df)
    null_dropped = initial_count - after_null_drop

    # 2. Deduplication: Drop exact duplicate rows across ['InvoiceNo', 'CustomerID', 'ProductID']
    clean_df = clean_df.drop_duplicates(subset=["InvoiceNo", "CustomerID", "ProductID"])
    after_dedup = len(clean_df)
    dedup_dropped = after_null_drop - after_dedup

    # 3. Aggregation: Group by CustomerID and ProductID, summing Quantity
    clean_df["Quantity"] = pd.to_numeric(clean_df["Quantity"], errors="coerce").fillna(1).astype(int)
    agg_df = (
        clean_df.groupby(["CustomerID", "ProductID"], as_index=False)["Quantity"]
        .sum()
        .rename(columns={"Quantity": "purchase_count"})
    )

    # 4. Implicit Rating Normalization:
    # Scale linearly to [1.0, 5.0]:
    # r_ui = 1 + 4 * (count_ui - 1) / (max(count) - 1)
    max_count = agg_df["purchase_count"].max()
    min_count = agg_df["purchase_count"].min()

    if max_count > min_count:
        agg_df["rating"] = 1.0 + 4.0 * ((agg_df["purchase_count"] - min_count) / (max_count - min_count))
    else:
        agg_df["rating"] = 3.0  # Default neutral score if uniform

    agg_df["rating"] = agg_df["rating"].round(2)

    # 5. Export surprise_ratings.csv (triples: CustomerID, ProductID, rating)
    surprise_df = agg_df[["CustomerID", "ProductID", "rating"]]
    surprise_df.to_csv(ratings_path, index=False)

    # 6. Pivot into dense User-Item Interaction Matrix
    user_item_matrix = agg_df.pivot(
        index="CustomerID",
        columns="ProductID",
        values="rating"
    ).fillna(0.0)

    # Sort indices for deterministic output
    user_item_matrix = user_item_matrix.sort_index(axis=0).sort_index(axis=1)
    user_item_matrix.to_csv(matrix_path)

    # Compute Matrix Sparsity
    total_cells = user_item_matrix.shape[0] * user_item_matrix.shape[1]
    non_zero_cells = (user_item_matrix > 0.0).values.sum()
    sparsity_pct = (1.0 - (non_zero_cells / total_cells)) * 100.0 if total_cells > 0 else 0.0

    print("==================================================")
    print("      DATA PREPROCESSING PIPELINE SUMMARY         ")
    print("==================================================")
    print(f"Initial raw transactions:     {initial_count}")
    print(f"Dropped Null ID rows:         {null_dropped}")
    print(f"Dropped Duplicate rows:       {dedup_dropped}")
    print(f"Cleaned valid transactions:   {after_dedup}")
    print(f"Aggregated User-Item pairs:   {len(agg_df)}")
    print(f"Unique Customers:             {user_item_matrix.shape[0]}")
    print(f"Unique Products:              {user_item_matrix.shape[1]}")
    print(f"Interaction Matrix Shape:     {user_item_matrix.shape}")
    print(f"Matrix Sparsity:              {sparsity_pct:.2f}%")
    print(f"Rating Range:                 [{agg_df['rating'].min()} - {agg_df['rating'].max()}]")
    print(f"Surprise Triples exported to: {ratings_path}")
    print(f"User-Item Matrix exported to: {matrix_path}")
    print("==================================================")

    return {
        "raw_count": initial_count,
        "clean_count": after_dedup,
        "user_count": user_item_matrix.shape[0],
        "item_count": user_item_matrix.shape[1],
        "sparsity_pct": sparsity_pct,
        "ratings_path": ratings_path,
        "matrix_path": matrix_path,
    }

if __name__ == "__main__":
    preprocess_pipeline()
