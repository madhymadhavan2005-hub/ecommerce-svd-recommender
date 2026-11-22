"""
Collaborative Filtering Recommender Engine using Scikit-Surprise SVD
Trains Matrix Factorization model, evaluates RMSE/MAE, implements Cold Start fallbacks,
and exports Top-N recommendations with Explainable AI rationales.
"""

import os
import json
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RATINGS_PATH = os.path.join(PROCESSED_DATA_DIR, "surprise_ratings.csv")
RECOMMENDATIONS_JSON_PATH = os.path.join(PROCESSED_DATA_DIR, "recommendations.json")

# Catalog Metadata for Rich Explanations
CATALOG = {
    "P101": {"name": "Wireless Mechanical Keyboard", "category": "Tech & Computing", "price": 129.0},
    "P102": {"name": "4K USB-C Docking Station", "category": "Tech & Computing", "price": 189.0},
    "P103": {"name": "Ergonomic Vertical Mouse", "category": "Tech & Computing", "price": 79.0},
    "P104": {"name": "Active Noise Canceling Headphones", "category": "Tech & Computing", "price": 249.0},
    "P105": {"name": "UltraWide 34-inch Monitor", "category": "Tech & Computing", "price": 499.0},
    "P201": {"name": "Precision Pour-Over Kettle", "category": "Home & Kitchen", "price": 95.0},
    "P202": {"name": "Burr Coffee Grinder Pro", "category": "Home & Kitchen", "price": 145.0},
    "P203": {"name": "Smart Air Fryer Oven", "category": "Home & Kitchen", "price": 160.0},
    "P204": {"name": "Cast Iron Dutch Oven", "category": "Home & Kitchen", "price": 115.0},
    "P205": {"name": "Dual-Boiler Espresso Machine", "category": "Home & Kitchen", "price": 599.0},
    "P301": {"name": "Adjustable Kettlebell 20kg", "category": "Fitness & Outdoor", "price": 110.0},
    "P302": {"name": "Insulated Trail Hydration Flask", "category": "Fitness & Outdoor", "price": 38.0},
    "P303": {"name": "High-Density Yoga Mat", "category": "Fitness & Outdoor", "price": 45.0},
    "P304": {"name": "Resistance Bands Pro Set", "category": "Fitness & Outdoor", "price": 29.0},
    "P305": {"name": "GPS Multisport Smartwatch", "category": "Fitness & Outdoor", "price": 299.0},
}

class RecommenderEngine:
    def __init__(self, ratings_file=RATINGS_PATH):
        self.ratings_file = ratings_file
        self.algo = SVD(
            n_factors=50,
            n_epochs=20,
            lr_all=0.005,
            reg_all=0.02,
            random_state=42
        )
        self.ratings_df = None
        self.full_trainset = None
        self.test_rmse = None
        self.test_mae = None
        self.all_products = list(CATALOG.keys())
        self.global_item_popularity = {}

    def load_data(self):
        """Loads normalized interaction triples."""
        if not os.path.exists(self.ratings_file):
            raise FileNotFoundError(f"Ratings file '{self.ratings_file}' not found. Run preprocess.py first.")
        self.ratings_df = pd.read_csv(self.ratings_file)
        self.ratings_df["CustomerID"] = self.ratings_df["CustomerID"].astype(str)
        self.ratings_df["ProductID"] = self.ratings_df["ProductID"].astype(str)
        self.ratings_df["rating"] = self.ratings_df["rating"].astype(float)

        # Compute global popularity baseline (average rating & transaction frequency)
        item_stats = (
            self.ratings_df.groupby("ProductID")
            .agg(avg_rating=("rating", "mean"), buyer_count=("CustomerID", "nunique"))
            .reset_index()
        )
        for _, row in item_stats.iterrows():
            self.global_item_popularity[row["ProductID"]] = {
                "avg_rating": round(float(row["avg_rating"]), 2),
                "buyer_count": int(row["buyer_count"])
            }

    def train_and_evaluate(self):
        """
        Trains SVD on an 80/20 train/test holdout split to compute RMSE/MAE,
        then refits on the full dataset for deployment inference.
        """
        self.load_data()

        reader = Reader(rating_scale=(1.0, 5.0))
        data = Dataset.load_from_df(self.ratings_df[["CustomerID", "ProductID", "rating"]], reader)

        # 80/20 Train/Test Split
        trainset, testset = train_test_split(data, test_size=0.20, random_state=42)

        # Fit model on training partition
        self.algo.fit(trainset)

        # Evaluate on holdout test partition
        predictions = self.algo.test(testset)
        self.test_rmse = accuracy.rmse(predictions, verbose=False)
        self.test_mae = accuracy.mae(predictions, verbose=False)

        # Fit on 100% data for maximum inference knowledge
        self.full_trainset = data.build_full_trainset()
        self.algo.fit(self.full_trainset)

        return {
            "rmse": round(self.test_rmse, 4),
            "mae": round(self.test_mae, 4),
            "total_ratings": len(self.ratings_df),
            "n_users": self.full_trainset.n_users,
            "n_items": self.full_trainset.n_items
        }

    def is_user_cold_start(self, customer_id: str) -> bool:
        """Determines if a customer has zero purchase history in the dataset."""
        if self.ratings_df is None:
            return True
        user_rows = self.ratings_df[self.ratings_df["CustomerID"] == str(customer_id)]
        return len(user_rows) == 0

    def recommend(self, customer_id: str, top_n: int = 3):
        """
        Generates Top-N recommendations for a customer.
        Gracefully handles Cold Start using popularity and category fallback.
        """
        customer_id = str(customer_id)

        # Determine already purchased items
        if self.ratings_df is not None:
            user_purchased = set(self.ratings_df[self.ratings_df["CustomerID"] == customer_id]["ProductID"].tolist())
        else:
            user_purchased = set()

        unpurchased = [pid for pid in self.all_products if pid not in user_purchased]
        if not unpurchased:
            unpurchased = self.all_products  # Fallback if user purchased all items

        # Check Cold Start
        if self.is_user_cold_start(customer_id):
            return self._cold_start_recommendations(customer_id, unpurchased, top_n)

        # Warm User SVD Prediction with Category Affinity Prior
        user_history = self.ratings_df[self.ratings_df["CustomerID"] == customer_id]
        top_purchased = user_history.sort_values(by="rating", ascending=False).iloc[0]
        top_sku = top_purchased["ProductID"]
        top_sku_meta = CATALOG.get(top_sku, {"name": top_sku, "category": "General"})

        # Calculate category preference distribution from user history
        category_weights = {}
        for _, row in user_history.iterrows():
            prod_info = CATALOG.get(str(row["ProductID"]))
            if prod_info:
                cat = prod_info["category"]
                category_weights[cat] = category_weights.get(cat, 0.0) + float(row["rating"])
        total_rating_mass = sum(category_weights.values()) or 1.0
        category_affinity = {cat: w / total_rating_mass for cat, w in category_weights.items()}

        scored_items = []
        for pid in unpurchased:
            pred = self.algo.predict(customer_id, pid)
            base_score = float(pred.est)
            prod_meta = CATALOG.get(pid, {"name": pid, "category": "General", "price": 0.0})
            cat = prod_meta["category"]

            # Hybrid score: Combine SVD latent score with user's category affinity boost
            cat_boost = category_affinity.get(cat, 0.0) * 1.50
            final_score = round(min(5.0, max(1.0, base_score + cat_boost)), 2)
            match_pct = int(round(max(0, min(100, (final_score - 1.0) / 4.0 * 100))))

            # Explainable AI rationale
            if cat == top_sku_meta["category"]:
                reason = f"High category affinity with your {cat} purchases (e.g. '{top_sku_meta['name']}')."
            elif cat_boost > 0:
                reason = f"Personalized category recommendation based on your interest in {cat}."
            else:
                reason = f"Collaborative filtering peer affinity: shoppers with similar purchase patterns also bought this item."

            scored_items.append({
                "product_id": pid,
                "name": prod_meta["name"],
                "category": prod_meta["category"],
                "price": prod_meta["price"],
                "est_score": final_score,
                "match_pct": match_pct,
                "is_cold_start": False,
                "reason": reason
            })

        scored_items.sort(key=lambda x: x["est_score"], reverse=True)
        return scored_items[:top_n]

    def _cold_start_recommendations(self, customer_id: str, candidates: list, top_n: int):
        """Fallback recommendation based on global catalog popularity."""
        cold_items = []
        for pid in candidates:
            stats = self.global_item_popularity.get(pid, {"avg_rating": 3.0, "buyer_count": 0})
            prod_meta = CATALOG.get(pid, {"name": pid, "category": "General", "price": 0.0})

            # Popularity-weighted cold-start score
            est_score = round(min(4.85, max(3.10, 2.5 + (stats["avg_rating"] * 0.45))), 2)
            match_pct = int(round(max(0, min(100, (est_score - 1.0) / 4.0 * 100))))

            if stats["buyer_count"] > 0:
                reason = f"Cold-start trending bestseller across {stats['buyer_count']} customer accounts with {stats['avg_rating']}/5.0 average satisfaction."
            else:
                reason = "Curated catalog baseline recommended for newly registered customer onboarding."

            cold_items.append({
                "product_id": pid,
                "name": prod_meta["name"],
                "category": prod_meta["category"],
                "price": prod_meta["price"],
                "est_score": est_score,
                "match_pct": match_pct,
                "is_cold_start": True,
                "reason": reason
            })

        cold_items.sort(key=lambda x: (x["est_score"], x["price"]), reverse=True)
        return cold_items[:top_n]

    def export_all_recommendations(self, output_path=RECOMMENDATIONS_JSON_PATH, top_n=5):
        """Exports recommendations for known profiles and cold start persona to JSON."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        all_customers = sorted(self.ratings_df["CustomerID"].unique().tolist())
        if "C004" not in all_customers:
            all_customers.append("C004")  # Explicit cold start profile

        export_data = {
            "metadata": {
                "algorithm": "SVD (Matrix Factorization)",
                "n_factors": 50,
                "n_epochs": 20,
                "test_rmse": round(self.test_rmse, 4) if self.test_rmse else None,
                "test_mae": round(self.test_mae, 4) if self.test_mae else None,
                "total_users": len(all_customers),
                "total_products": len(self.all_products)
            },
            "recommendations": {}
        }

        for cid in all_customers:
            export_data["recommendations"][cid] = self.recommend(cid, top_n=top_n)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

        print(f"Exported recommendations for {len(all_customers)} customers to '{output_path}'.")
        return export_data

def train_and_recommend():
    engine = RecommenderEngine()
    print("Loading data & training SVD model...")
    metrics = engine.train_and_evaluate()
    print("==================================================")
    print("        SVD RECOMMENDER EVALUATION RESULTS        ")
    print("==================================================")
    print(f"Model Algorithm:        SVD(n_factors=50, n_epochs=20)")
    print(f"Test Set RMSE:          {metrics['rmse']}")
    print(f"Test Set MAE:           {metrics['mae']}")
    print(f"Total Ratings Trained:  {metrics['total_ratings']}")
    print(f"Users Factorized:       {metrics['n_users']}")
    print(f"Items Factorized:       {metrics['n_items']}")
    print("==================================================")

    # Test sample inference across key UI personas
    test_personas = [
        ("C001", "Alex Rivera (Tech Enthusiast)"),
        ("C002", "Priya Sharma (Home & Kitchen)"),
        ("C003", "Marcus Vance (Fitness & Outdoor)"),
        ("C004", "Elena Rostova (Cold Start / New User)"),
    ]

    print("\nSample Top-3 Inference Results:")
    for cid, label in test_personas:
        recs = engine.recommend(cid, top_n=3)
        print(f"\n--- {cid}: {label} ---")
        for idx, r in enumerate(recs, 1):
            flag = "[COLD-START]" if r["is_cold_start"] else "[PERSONALIZED]"
            print(f"  {idx}. {flag} {r['product_id']} - {r['name']} (${r['price']})")
            print(f"     Est. Rating: {r['est_score']}/5.0 ({r['match_pct']}% match)")
            print(f"     Rationale:   {r['reason']}")

    engine.export_all_recommendations()
    return engine

if __name__ == "__main__":
    train_and_recommend()
