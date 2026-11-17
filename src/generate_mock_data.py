"""
Mock Data Generator for E-Commerce Recommendation Pipeline
Generates synthetic transaction logs with deliberate duplicates, nulls, and customer segments.
"""

import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# Directory configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUTPUT_FILE = os.path.join(RAW_DATA_DIR, "transactions.csv")

# Catalog definitions
PRODUCTS = [
    # Tech & Computing
    {"id": "P101", "name": "Wireless Mechanical Keyboard", "category": "Tech & Computing", "price": 129.0},
    {"id": "P102", "name": "4K USB-C Docking Station", "category": "Tech & Computing", "price": 189.0},
    {"id": "P103", "name": "Ergonomic Vertical Mouse", "category": "Tech & Computing", "price": 79.0},
    {"id": "P104", "name": "Active Noise Canceling Headphones", "category": "Tech & Computing", "price": 249.0},
    {"id": "P105", "name": "UltraWide 34-inch Monitor", "category": "Tech & Computing", "price": 499.0},
    # Home & Kitchen
    {"id": "P201", "name": "Precision Pour-Over Kettle", "category": "Home & Kitchen", "price": 95.0},
    {"id": "P202", "name": "Burr Coffee Grinder Pro", "category": "Home & Kitchen", "price": 145.0},
    {"id": "P203", "name": "Smart Air Fryer Oven", "category": "Home & Kitchen", "price": 160.0},
    {"id": "P204", "name": "Cast Iron Dutch Oven", "category": "Home & Kitchen", "price": 115.0},
    {"id": "P205", "name": "Dual-Boiler Espresso Machine", "category": "Home & Kitchen", "price": 599.0},
    # Fitness & Outdoor
    {"id": "P301", "name": "Adjustable Kettlebell 20kg", "category": "Fitness & Outdoor", "price": 110.0},
    {"id": "P302", "name": "Insulated Trail Hydration Flask", "category": "Fitness & Outdoor", "price": 38.0},
    {"id": "P303", "name": "High-Density Yoga Mat", "category": "Fitness & Outdoor", "price": 45.0},
    {"id": "P304", "name": "Resistance Bands Pro Set", "category": "Fitness & Outdoor", "price": 29.0},
    {"id": "P305", "name": "GPS Multisport Smartwatch", "category": "Fitness & Outdoor", "price": 299.0},
]

PRODUCT_LOOKUP = {p["id"]: p for p in PRODUCTS}

# Core archetypes matching UI prototype
CORE_PROFILES = [
    {"id": "C001", "persona": "Tech", "affinity": ["P101", "P102", "P103", "P104", "P105"]},
    {"id": "C002", "persona": "Kitchen", "affinity": ["P201", "P202", "P203", "P204", "P205"]},
    {"id": "C003", "persona": "Fitness", "affinity": ["P301", "P302", "P303", "P304", "P305"]},
    {"id": "C004", "persona": "ColdStart", "affinity": []},  # Explicitly zero purchases
]

def generate_transactions(num_records=1200):
    """Generate realistic e-commerce transactions with deliberate noise."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    records = []
    base_time = datetime(2026, 10, 1, 8, 30)
    invoice_seq = 1000

    # 1. Deterministic baseline matching prototype UI
    baseline_events = [
        ("C001", "P101", 2, timedelta(days=2, hours=1)),
        ("C001", "P102", 1, timedelta(days=5, hours=6)),
        ("C001", "P201", 1, timedelta(days=9, hours=10)),
        ("C002", "P201", 3, timedelta(days=11, hours=3)),
        ("C002", "P202", 2, timedelta(days=14, hours=8)),
        ("C002", "P302", 1, timedelta(days=18, hours=4)),
        ("C003", "P301", 2, timedelta(days=21, hours=9)),
        ("C003", "P302", 3, timedelta(days=24, hours=2)),
        ("C003", "P103", 1, timedelta(days=28, hours=12)),
    ]

    for cust_id, prod_id, qty, offset in baseline_events:
        invoice_seq += 1
        prod = PRODUCT_LOOKUP[prod_id]
        records.append({
            "InvoiceNo": f"INV-{invoice_seq}",
            "CustomerID": cust_id,
            "ProductID": prod_id,
            "Quantity": qty,
            "InvoiceDate": (base_time + offset).strftime("%Y-%m-%d %H:%M:%S"),
            "UnitPrice": prod["price"]
        })

    # 2. Broader customer population (C005 - C060)
    customer_personas = {}
    for c_num in range(5, 61):
        cid = f"C{c_num:03d}"
        segment = random.choices(["Tech", "Kitchen", "Fitness", "CrossShopper"], weights=[0.35, 0.35, 0.25, 0.05])[0]
        customer_personas[cid] = segment

    # Product category groupings
    tech_skus = [p["id"] for p in PRODUCTS if p["category"] == "Tech & Computing"]
    kitchen_skus = [p["id"] for p in PRODUCTS if p["category"] == "Home & Kitchen"]
    fitness_skus = [p["id"] for p in PRODUCTS if p["category"] == "Fitness & Outdoor"]
    all_skus = tech_skus + kitchen_skus + fitness_skus

    # Precompute probability distributions for each persona
    weights_tech = ([0.75 / len(tech_skus)] * len(tech_skus) +
                    [0.15 / len(kitchen_skus)] * len(kitchen_skus) +
                    [0.10 / len(fitness_skus)] * len(fitness_skus))

    weights_kitchen = ([0.15 / len(tech_skus)] * len(tech_skus) +
                       [0.75 / len(kitchen_skus)] * len(kitchen_skus) +
                       [0.10 / len(fitness_skus)] * len(fitness_skus))

    weights_fitness = ([0.10 / len(tech_skus)] * len(tech_skus) +
                       [0.15 / len(kitchen_skus)] * len(kitchen_skus) +
                       [0.75 / len(fitness_skus)] * len(fitness_skus))

    while len(records) < num_records:
        cid = random.choice(list(customer_personas.keys()))
        segment = customer_personas[cid]

        # Select SKU based on customer segment preference
        if segment == "Tech":
            sku = random.choices(all_skus, weights=weights_tech)[0]
        elif segment == "Kitchen":
            sku = random.choices(all_skus, weights=weights_kitchen)[0]
        elif segment == "Fitness":
            sku = random.choices(all_skus, weights=weights_fitness)[0]
        else:
            sku = random.choice(all_skus)

        # Quantity distribution (1 to 5 units)
        qty = int(np.random.choice([1, 2, 3, 4, 5], p=[0.55, 0.25, 0.12, 0.05, 0.03]))

        days_offset = random.randint(0, 45)
        minute_offset = random.randint(0, 1440)
        t_stamp = base_time + timedelta(days=days_offset, minutes=minute_offset)
        invoice_seq += 1

        records.append({
            "InvoiceNo": f"INV-{invoice_seq}",
            "CustomerID": cid,
            "ProductID": sku,
            "Quantity": qty,
            "InvoiceDate": t_stamp.strftime("%Y-%m-%d %H:%M:%S"),
            "UnitPrice": PRODUCT_LOOKUP[sku]["price"]
        })

    # 3. Deliberately inject noise: Exact duplicates (e.g., re-posts or logging glitches)
    duplicate_count = 35
    for _ in range(duplicate_count):
        dup_target = random.choice(records[:500])
        records.append(dup_target.copy())

    # 4. Deliberately inject noise: Null CustomerIDs
    null_cust_count = 25
    for _ in range(null_cust_count):
        sku = random.choice(all_skus)
        invoice_seq += 1
        records.append({
            "InvoiceNo": f"INV-{invoice_seq}",
            "CustomerID": None,
            "ProductID": sku,
            "Quantity": 1,
            "InvoiceDate": (base_time + timedelta(days=random.randint(1, 40))).strftime("%Y-%m-%d %H:%M:%S"),
            "UnitPrice": PRODUCT_LOOKUP[sku]["price"]
        })

    # 5. Deliberately inject noise: Null ProductIDs
    null_prod_count = 15
    for _ in range(null_prod_count):
        invoice_seq += 1
        cid = f"C{random.randint(5, 50):03d}"
        records.append({
            "InvoiceNo": f"INV-{invoice_seq}",
            "CustomerID": cid,
            "ProductID": None,
            "Quantity": 1,
            "InvoiceDate": (base_time + timedelta(days=random.randint(1, 40))).strftime("%Y-%m-%d %H:%M:%S"),
            "UnitPrice": 0.0
        })

    # Shuffle to ensure realistic interspersed logs
    random.shuffle(records)

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Generated {len(df)} raw transaction records in '{OUTPUT_FILE}'.")
    print(f"   - Null CustomerID rows: {df['CustomerID'].isna().sum()}")
    print(f"   - Null ProductID rows:  {df['ProductID'].isna().sum()}")
    print(f"   - Total duplicates across [InvoiceNo, CustomerID, ProductID]: {df.duplicated(subset=['InvoiceNo', 'CustomerID', 'ProductID']).sum()}")
    print(f"   - Unique Customers: {df['CustomerID'].dropna().nunique()}")
    print(f"   - Unique Products:  {df['ProductID'].dropna().nunique()}")
    return df

if __name__ == "__main__":
    generate_transactions()
