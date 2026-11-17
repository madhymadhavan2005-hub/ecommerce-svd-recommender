# Antigravity Multi-Agent Blueprint: E-Commerce Product Recommender

Drop this file into the root of your `ecommerce-recommender` workspace in **Google Antigravity**, open the **Agent Manager (Mission Control)**, and paste the **Master Multi-Agent Prompt** below to orchestrate the build.

---

## 1. Master Multi-Agent Prompt (Copy & Paste into Agent Manager)

> **Objective:** Build an end-to-end E-Commerce Customer Purchase Data Preparation & Collaborative Filtering Recommendation System following the architecture and specifications in `@ANTIGRAVITY_BLUEPRINT.md`.
>
> **Agent Orchestration Plan:**
> 1. **Workspace & Data Architect Agent:** Scaffold the project directory, create `requirements.txt`, and build `src/generate_mock_data.py` to generate realistic raw transaction logs (`data/raw/transactions.csv`) containing deliberate duplicates, null IDs, timestamps, and customer segments for testing.
> 2. **Data Pipeline Engineer Agent:** Implement `src/preprocess.py` to clean raw transactions, aggregate purchase frequencies, min-max scale implicit quantities into a `1.0–5.0` rating scale, and export `data/processed/user_item_matrix.csv` and `data/processed/surprise_ratings.csv`.
> 3. **ML Recommender Agent:** Implement `src/recommender.py` using `scikit-surprise` (`SVD`), train with an 80/20 split, log test `RMSE`, implement Top-N recommendation inference with a Cold Start popularity fallback, and expose a lightweight local JSON/API bridge (`src/app.py` via FastAPI or Flask) or export predictions JSON (`data/processed/recommendations.json`).
> 4. **Frontend Prototype & Browser Verification Agent:** Connect `index.html` (our interactive dashboard prototype) to real pipeline outputs (or live local API endpoints), launch a local server, and verify via the Antigravity Browser Surface that switching customer profiles, simulating purchases, and inspecting the interaction matrix work without console errors.
>
> **Verification Requirement:** Run all Python scripts sequentially in the integrated terminal, confirm zero errors, verify that `RMSE` is logged, and capture a browser verification check of the interactive dashboard.

---

## 2. Target Workspace Architecture

```text
ecommerce-recommender/
│
├── ANTIGRAVITY_BLUEPRINT.md      # This specification & prompt file
├── requirements.txt              # Python dependencies
├── README.md                     # Setup & execution instructions
├── index.html                    # Interactive SVD & Data Prep Prototype UI
│
├── data/
│   ├── raw/
│   │   └── transactions.csv      # Generated raw logs (with noise/duplicates)
│   └── processed/
│       ├── user_item_matrix.csv  # Pivoted Customers × Products matrix
│       ├── surprise_ratings.csv  # Normalized [1.0 - 5.0] interaction triples
│       └── recommendations.json  # Exported Top-N predictions for UI consumption
│
└── src/
    ├── __init__.py
    ├── generate_mock_data.py     # Synthetic e-commerce transaction generator
    ├── preprocess.py             # Cleaning, aggregation & scaling pipeline
    ├── recommender.py            # Scikit-Surprise SVD training & Top-N inference
    └── server.py                 # Optional FastAPI/Flask live backend for index.html
```

---

## 3. Technical Specifications & Acceptance Criteria

### A. Dependencies (`requirements.txt`)
```text
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scikit-surprise>=1.1.3
fastapi>=0.110.0
uvicorn>=0.27.0
```

### B. Data Preparation Rules (`src/preprocess.py`)
* **Null Handling:** Drop any row where `CustomerID` or `ProductID` is `NaN` or empty.
* **Deduplication:** Drop exact duplicate rows across `['InvoiceNo', 'CustomerID', 'ProductID']`.
* **Implicit Rating Normalization:** Group by `['CustomerID', 'ProductID']`, sum `Quantity` as `purchase_count`, and scale linearly to $[1.0, 5.0]$:
  $$r_{ui} = 1 + 4 \times \frac{\text{count}_{ui} - 1}{\max(\text{count}) - 1}$$
* **Matrix Export:** Pivot into a dense/sparse matrix (`index='CustomerID'`, `columns='ProductID'`, `values='rating'`, `fill_value=0.0`).

### C. Collaborative Filtering Model (`src/recommender.py`)
* **Algorithm:** `SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)`
* **Evaluation:** Compute and print test set `RMSE` and `MAE` on a 20% holdout set.
* **Cold Start Strategy:** If a queried `CustomerID` is absent from `trainset`, gracefully fall back to items ranked by highest global mean rating and purchase frequency rather than throwing a `ValueError`.

### D. Frontend Dashboard Integration (`index.html`)
* Keep the single-file interactive dashboard prototype responsive and self-contained.
* Ensure the **Simulate New Purchase** button updates the interaction matrix state, recalculates sparsity, and updates the Top-N SVD recommendations immediately.

---

## 4. Antigravity Agent Guardrails (Optional `.antigravity/rules.md`)

If you use workspace rules in Antigravity, include these constraints so agents stay aligned:
1. **Virtual Environment:** Always activate or run Python commands inside `./venv` if present.
2. **Data Integrity:** Never overwrite `data/raw/transactions.csv` inside `preprocess.py`; write all cleaned outputs strictly to `data/processed/`.
3. **Single-File UI:** Keep `index.html` self-contained with Tailwind CDN and inline JS/CSS so it can be opened directly or served via `uvicorn`/`python -m http.server`.