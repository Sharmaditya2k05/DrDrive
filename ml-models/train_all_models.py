"""
Train all Dr Drive ML models on the Indian car dataset.

Models trained:
  1. fault_classifier.pkl     - detects engine faults from OBD (RandomForest)
  2. health_scorer.pkl        - predicts health score 0-100 (GradientBoosting)
  3. failure_predictor.pkl    - predicts component failures (RandomForest multi-output)
  4. valuation_model.pkl      - predicts market value in INR (GradientBoosting)

Run: python3 train_all_models.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, mean_absolute_error,
                              r2_score, accuracy_score)

os.makedirs("models", exist_ok=True)

print("=" * 60)
print(" DR DRIVE — ML MODEL TRAINING")
print(" Indian Car Dataset: 22 brands, 160 models")
print("=" * 60)


# ── Load datasets ─────────────────────────────────────────────────────────────

obd_df   = pd.read_csv("data/indian_cars_obd.csv")
maint_df = pd.read_csv("data/indian_cars_maintenance.csv")
val_df   = pd.read_csv("data/indian_cars_valuation.csv")

print(f"\nLoaded:")
print(f"  OBD dataset:         {len(obd_df):,} rows")
print(f"  Maintenance dataset: {len(maint_df):,} rows")
print(f"  Valuation dataset:   {len(val_df):,} rows")


# ── Encoders shared across models ────────────────────────────────────────────

fuel_enc    = LabelEncoder().fit(obd_df["fuel_type"])
segment_enc = LabelEncoder().fit(obd_df["segment"])
brand_enc   = LabelEncoder().fit(obd_df["brand"])

def encode_cat(df, col, enc):
    return enc.transform(df[col].fillna("petrol"))


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — FAULT CLASSIFIER (OBD → faulty / not faulty)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("Training Model 1: Fault Classifier (RandomForest)")
print("─" * 60)

OBD_FEATURES = [
    "rpm", "coolant_temp", "intake_air_temp", "engine_load",
    "throttle_pos", "fuel_trim_st", "fuel_trim_lt",
    "battery_voltage", "o2_voltage", "maf", "timing_advance",
    "mil_on", "dtc_count",
    "age_years", "odometer_km",
]

X_fault = obd_df[OBD_FEATURES].fillna(0)
y_fault = obd_df["is_faulty"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_fault, y_fault, test_size=0.2, random_state=42, stratify=y_fault)

fault_model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ))
])

fault_model.fit(X_tr, y_tr)
y_pred = fault_model.predict(X_te)
acc = accuracy_score(y_te, y_pred)
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_te, y_pred,
      target_names=["Healthy", "Faulty"], digits=3))

# Feature importance
importances = fault_model.named_steps["clf"].feature_importances_
feat_imp = sorted(zip(OBD_FEATURES, importances), key=lambda x: -x[1])
print("Top 5 features:")
for f, imp in feat_imp[:5]:
    print(f"  {f:<22} {imp:.4f}")

# Save with feature names and encoders
fault_payload = {
    "model":         fault_model,
    "features":      OBD_FEATURES,
    "accuracy":      acc,
}
with open("models/fault_classifier.pkl", "wb") as f:
    pickle.dump(fault_payload, f)
print("Saved: models/fault_classifier.pkl")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — HEALTH SCORER (car attributes → health score 0-100)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("Training Model 2: Health Scorer (GradientBoosting)")
print("─" * 60)

HEALTH_FEATURES = [
    "age_years", "odometer_km", "engine_health", "transmission_health",
    "battery_health", "brake_health", "tire_health",
    "needs_brake_replacement", "needs_battery_replacement",
    "needs_tire_replacement",
]

# Encode fuel type
maint_enc = maint_df.copy()
maint_enc["fuel_enc"] = LabelEncoder().fit_transform(maint_enc["fuel_type"])
HEALTH_FEATURES_EXT = HEALTH_FEATURES + ["fuel_enc"]

X_health = maint_enc[HEALTH_FEATURES_EXT].fillna(0)
y_health = maint_enc["health_score"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_health, y_health, test_size=0.2, random_state=42)

health_model = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    ))
])

health_model.fit(X_tr, y_tr)
y_pred_h = health_model.predict(X_te)
mae = mean_absolute_error(y_te, y_pred_h)
r2  = r2_score(y_te, y_pred_h)
print(f"MAE: {mae:.2f} points  |  R²: {r2:.4f}")

health_payload = {
    "model":    health_model,
    "features": HEALTH_FEATURES_EXT,
    "mae":      mae,
    "r2":       r2,
}
with open("models/health_scorer.pkl", "wb") as f:
    pickle.dump(health_payload, f)
print("Saved: models/health_scorer.pkl")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — FAILURE PREDICTOR (component failure flags)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("Training Model 3: Failure Predictor (RandomForest multi-output)")
print("─" * 60)

FAILURE_FEATURES = [
    "age_years", "odometer_km",
    "engine_health", "transmission_health", "battery_health",
    "brake_health", "tire_health",
]
FAILURE_TARGETS = [
    "needs_service_soon",
    "needs_brake_replacement",
    "needs_battery_replacement",
    "needs_tire_replacement",
]

X_fail = maint_df[FAILURE_FEATURES].fillna(0)
y_fail = maint_df[FAILURE_TARGETS]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_fail, y_fail, test_size=0.2, random_state=42)

failure_models = {}
for target in FAILURE_TARGETS:
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ))
    ])
    clf.fit(X_tr, y_tr[target])
    acc = accuracy_score(y_te[target], clf.predict(X_te))
    print(f"  {target:<35} acc={acc:.3f}")
    failure_models[target] = clf

failure_payload = {
    "models":   failure_models,
    "features": FAILURE_FEATURES,
    "targets":  FAILURE_TARGETS,
}
with open("models/failure_predictor.pkl", "wb") as f:
    pickle.dump(failure_payload, f)
print("Saved: models/failure_predictor.pkl")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 4 — VALUATION MODEL (car attributes → market value INR)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 60)
print("Training Model 4: Valuation Model (GradientBoosting)")
print("─" * 60)

val_enc = val_df.copy()
val_enc["fuel_enc"]    = LabelEncoder().fit_transform(val_enc["fuel_type"])
val_enc["segment_enc"] = LabelEncoder().fit_transform(val_enc["segment"])
val_enc["city_enc"]    = LabelEncoder().fit_transform(val_enc["city"])
val_enc["brand_enc"]   = LabelEncoder().fit_transform(val_enc["brand"])

VAL_FEATURES = [
    "age_years", "odometer_km", "base_price_lakh", "engine_cc",
    "owner_number", "accidents", "health_score", "damage_count",
    "fuel_enc", "segment_enc", "city_enc", "brand_enc",
]

X_val = val_enc[VAL_FEATURES].fillna(0)
y_val = val_enc["market_value_inr"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_val, y_val, test_size=0.2, random_state=42)

val_model = Pipeline([
    ("scaler", StandardScaler()),
    ("reg", GradientBoostingRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    ))
])

val_model.fit(X_tr, y_tr)
y_pred_v = val_model.predict(X_te)
mae_v = mean_absolute_error(y_te, y_pred_v)
r2_v  = r2_score(y_te, y_pred_v)
print(f"MAE: ₹{mae_v:,.0f}  |  R²: {r2_v:.4f}")

# Save with encoders for inference
val_payload = {
    "model":        val_model,
    "features":     VAL_FEATURES,
    "fuel_enc":     LabelEncoder().fit(val_df["fuel_type"]),
    "segment_enc":  LabelEncoder().fit(val_df["segment"]),
    "city_enc":     LabelEncoder().fit(val_df["city"]),
    "brand_enc":    LabelEncoder().fit(val_df["brand"]),
    "mae":          mae_v,
    "r2":           r2_v,
}
with open("models/valuation_model.pkl", "wb") as f:
    pickle.dump(val_payload, f)
print("Saved: models/valuation_model.pkl")


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(" TRAINING COMPLETE")
print("=" * 60)
for fname in os.listdir("models"):
    size = os.path.getsize(f"models/{fname}") / 1024
    print(f"  {fname:<35} {size:,.0f} KB")

print("\nQuick inference test:")

# Test fault classifier
test_obd = {f: 0 for f in OBD_FEATURES}
test_obd.update({"rpm": 850, "coolant_temp": 110, "fuel_trim_lt": 18,
                 "battery_voltage": 11.8, "mil_on": 1, "dtc_count": 2,
                 "age_years": 8, "odometer_km": 95000})
X_test = pd.DataFrame([test_obd])[OBD_FEATURES]
fault_prob = fault_model.predict_proba(X_test)[0][1]
print(f"  Fault probability (degraded car): {fault_prob:.1%}")

test_obd2 = {f: 0 for f in OBD_FEATURES}
test_obd2.update({"rpm": 800, "coolant_temp": 88, "fuel_trim_lt": 1,
                  "battery_voltage": 13.5, "mil_on": 0, "dtc_count": 0,
                  "age_years": 2, "odometer_km": 15000})
X_test2 = pd.DataFrame([test_obd2])[OBD_FEATURES]
fault_prob2 = fault_model.predict_proba(X_test2)[0][1]
print(f"  Fault probability (new car):      {fault_prob2:.1%}")

# Test valuation
test_val = pd.DataFrame([{
    "age_years": 3, "odometer_km": 35000, "base_price_lakh": 8.0,
    "engine_cc": 1197, "owner_number": 1, "accidents": 0,
    "health_score": 85, "damage_count": 1,
    "fuel_enc": 4, "segment_enc": 3, "city_enc": 0, "brand_enc": 12,
}])
val_pred = val_model.predict(test_val)[0]
print(f"  Maruti Swift 2022 predicted value: ₹{val_pred:,.0f}")
print("\nAll models ready for production!")
