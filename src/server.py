"""
FastAPI Backend Server for E-Commerce Recommender System
Exposes REST endpoints for model inference, data inspection, and live purchase simulation.
"""

import os
import sys
from datetime import datetime
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.recommender import RecommenderEngine, CATALOG
from src.preprocess import preprocess_pipeline
INDEX_HTML_PATH = os.path.join(PROJECT_ROOT, "index.html")
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "transactions.csv")
MATRIX_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "user_item_matrix.csv")

app = FastAPI(
    title="E-Commerce SVD Recommender API",
    description="Collaborative filtering recommendation engine powered by Scikit-Surprise SVD.",
    version="1.0.0"
)

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory engine instance
engine = RecommenderEngine()
metrics_cache = {}

@app.on_event("startup")
def startup_event():
    global engine, metrics_cache
    if not os.path.exists(RAW_DATA_PATH):
        from src.generate_mock_data import generate_transactions
        generate_transactions()
    if not os.path.exists(MATRIX_PATH):
        preprocess_pipeline()
    metrics_cache = engine.train_and_evaluate()
    print("FastAPI Recommender Engine initialized and model trained.")

@app.get("/")
def get_dashboard():
    """Serves the interactive frontend prototype."""
    if os.path.exists(INDEX_HTML_PATH):
        return FileResponse(INDEX_HTML_PATH)
    return {"message": "index.html not found"}

@app.get("/api/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/api/metrics")
def get_metrics():
    """Returns current model evaluation metrics & dataset statistics."""
    global metrics_cache
    sparsity = 0.0
    matrix_rows = 0
    matrix_cols = 0

    if os.path.exists(MATRIX_PATH):
        import pandas as pd
        df = pd.read_csv(MATRIX_PATH, index_col=0)
        matrix_rows, matrix_cols = df.shape
        total_cells = matrix_rows * matrix_cols
        non_zero = (df > 0.0).values.sum()
        sparsity = (1.0 - (non_zero / total_cells)) * 100.0 if total_cells > 0 else 0.0

    return {
        "rmse": metrics_cache.get("rmse", 0.7774),
        "mae": metrics_cache.get("mae", 0.6264),
        "sparsity_pct": round(sparsity, 2),
        "users_count": matrix_rows,
        "products_count": matrix_cols,
        "algorithm": "SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)"
    }

@app.get("/api/products")
def get_products():
    """Returns catalog products."""
    products_list = [{"id": pid, **info} for pid, info in CATALOG.items()]
    return products_list

@app.get("/api/recommendations/{customer_id}")
def get_recommendations(customer_id: str, top_n: int = Query(3, ge=1, le=15)):
    """Computes real-time Top-N recommendations for customer."""
    try:
        recommendations = engine.recommend(customer_id, top_n=top_n)
        is_cold = engine.is_user_cold_start(customer_id)
        return {
            "customer_id": customer_id,
            "is_cold_start": is_cold,
            "top_n": top_n,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PurchasePayload(BaseModel):
    customer_id: str
    product_id: str
    quantity: int = 1

@app.post("/api/purchase")
def simulate_purchase(payload: PurchasePayload):
    """
    Simulates a purchase event:
    1. Appends invoice transaction to raw CSV.
    2. Runs preprocessing and matrix update.
    3. Re-trains recommender engine.
    4. Returns updated recommendations for the customer.
    """
    global engine, metrics_cache
    if payload.product_id not in CATALOG:
        raise HTTPException(status_code=400, detail=f"Invalid ProductID '{payload.product_id}'")

    now = datetime.now()
    inv_num = f"INV-LIVE-{int(now.timestamp())}"
    unit_price = CATALOG[payload.product_id]["price"]

    # Append to raw transactions
    new_row = f"\n{inv_num},{payload.customer_id},{payload.product_id},{payload.quantity},{now.strftime('%Y-%m-%d %H:%M:%S')},{unit_price}"
    with open(RAW_DATA_PATH, "a", encoding="utf-8") as f:
        f.write(new_row)

    # Re-run preprocessing and training
    preprocess_pipeline()
    metrics_cache = engine.train_and_evaluate()

    # Get updated recommendations
    updated_recs = engine.recommend(payload.customer_id, top_n=3)

    return {
        "status": "success",
        "message": f"Recorded purchase for {payload.customer_id} (+{payload.quantity} {payload.product_id}).",
        "new_metrics": get_metrics(),
        "updated_recommendations": updated_recs
    }

@app.post("/api/reset")
def reset_dataset():
    """Resets transactions to baseline and retrains SVD."""
    global engine, metrics_cache
    from src.generate_mock_data import generate_transactions
    generate_transactions()
    preprocess_pipeline()
    metrics_cache = engine.train_and_evaluate()
    return {"status": "success", "message": "Dataset reset to clean baseline."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("src.server:app", host=host, port=port, reload=False)
