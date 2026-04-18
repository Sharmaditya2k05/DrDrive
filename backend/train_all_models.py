"""
Dr Drive — Complete ML Training Pipeline
=========================================
Run this script to train all models from your datasets.

Usage:
    cd D:\DrDrive\DrDriveBackend
    pip install pandas scikit-learn xgboost
    python train_all_models.py

Output:
    ml_models/fault_classifier.pkl
    ml_models/maintenance_predictor.pkl
    ml_models/health_scorer.pkl
    ml_models/valuation_model.pkl
    ml_models/label_encoders.pkl
"""

import os, pickle, warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, mean_absolute_error, r2_score, roc_auc_score

warnings.filterwarnings('ignore')
os.makedirs('ml_models', exist_ok=True)

DATA_DIR = 'data'   # place your CSV files here


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — Fault Detector
# Input:  OBD readings (rpm, coolant, fuel trim, battery, DTC count...)
# Output: fault probability 0-1
# ─────────────────────────────────────────────────────────────────────────────

def train_fault_detector():
    print("\n[1/4] Training Fault Detector...")

    dfs = []

    # engine_failure_dataset.csv
    p = os.path.join(DATA_DIR, 'engine_failure_dataset.csv')
    if os.path.exists(p):
        df = pd.read_csv(p).rename(columns={
            'Temperature (°C)': 'coolant_temp', 'RPM': 'rpm',
            'Fuel_Efficiency': 'engine_load', 'Torque': 'timing_advance',
            'Power_Output (kW)': 'maf', 'Fault_Condition': 'label'
        })
        df['label']          = (df['label'] > 0).astype(int)
        df['battery_voltage']= 13.5; df['fuel_trim_st'] = 0.0
        df['fuel_trim_lt']   = 0.0;  df['o2_voltage']   = 0.45
        df['throttle_pos']   = 20.0; df['intake_air_temp'] = 35.0
        df['mil_on']         = df['label']; df['dtc_count'] = df['label'] * 2
        dfs.append(df)

    # engine_data.csv
    p = os.path.join(DATA_DIR, 'engine_data.csv')
    if os.path.exists(p):
        df = pd.read_csv(p).rename(columns={
            'Engine rpm': 'rpm', 'Coolant temp': 'coolant_temp',
            'lub oil temp': 'intake_air_temp', 'Engine Condition': 'label'
        })
        df['label']          = df['label'].astype(int)
        df['engine_load']    = 30.0; df['battery_voltage'] = 13.5
        df['fuel_trim_st']   = 0.0;  df['fuel_trim_lt']   = 0.0
        df['o2_voltage']     = 0.45; df['maf']            = 5.0
        df['timing_advance'] = 15.0; df['throttle_pos']   = 20.0
        df['mil_on']         = df['label']; df['dtc_count'] = df['label']
        dfs.append(df)

    if not dfs:
        print("  WARNING: No data found. Place CSVs in data/ folder.")
        return

    FEATURES = ['rpm','coolant_temp','intake_air_temp','engine_load',
                'throttle_pos','fuel_trim_st','fuel_trim_lt',
                'battery_voltage','o2_voltage','maf',
                'timing_advance','mil_on','dtc_count']

    df_all = pd.concat(dfs, ignore_index=True).fillna(0)
    X = df_all[FEATURES]; y = df_all['label']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=42, stratify=y)

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    RandomForestClassifier(n_estimators=200, max_depth=8,
                   class_weight='balanced', random_state=42, n_jobs=-1))
    ])
    model.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:,1])
    print(f"  ROC-AUC: {auc:.4f} | Samples: {len(df_all):,}")
    with open('ml_models/fault_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("  Saved: ml_models/fault_classifier.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — Maintenance Predictor
# MODEL 3 — Health Scorer
# MODEL 4 — Valuation Engine
# All use vehicle_maintenance_data.csv
# ─────────────────────────────────────────────────────────────────────────────

def train_vehicle_models():
    p = os.path.join(DATA_DIR, 'vehicle_maintenance_data.csv')
    if not os.path.exists(p):
        print("  WARNING: vehicle_maintenance_data.csv not found")
        return

    df = pd.read_csv(p)

    # Encode categorical columns
    cat_map = {
        'Tire_Condition':  {'New': 0, 'Good': 1, 'Worn Out': 2},
        'Brake_Condition': {'New': 0, 'Good': 1, 'Worn Out': 2},
        'Battery_Status':  {'New': 0, 'Good': 1, 'Weak': 2},
    }
    le_dict = {}
    for col, mapping in cat_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(1).astype(int)

    for col in ['Vehicle_Model','Fuel_Type','Transmission_Type','Owner_Type','Maintenance_History']:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le

    with open('ml_models/label_encoders.pkl', 'wb') as f:
        pickle.dump(le_dict, f)

    FEATURES = ['Vehicle_Age','Odometer_Reading','Service_History',
                'Accident_History','Tire_Condition','Brake_Condition',
                'Battery_Status','Reported_Issues','Fuel_Efficiency',
                'Engine_Size','Mileage']
    feat = [f for f in FEATURES if f in df.columns]

    # ── Model 2: Maintenance classifier ──────────────────────
    print("\n[2/4] Training Maintenance Predictor...")
    X_m = df[feat].fillna(0); y_m = df['Need_Maintenance']
    X_tr, X_te, y_tr, y_te = train_test_split(X_m, y_m, test_size=0.2,
                                               random_state=42, stratify=y_m)
    maint = Pipeline([('scaler', StandardScaler()),
                      ('clf',    RandomForestClassifier(n_estimators=150,
                                 class_weight='balanced', random_state=42, n_jobs=-1))])
    maint.fit(X_tr, y_tr)
    print(f"  Accuracy: {(maint.predict(X_te)==y_te).mean():.2%} | Samples: {len(df):,}")
    with open('ml_models/maintenance_predictor.pkl', 'wb') as f:
        pickle.dump({'model': maint, 'features': feat}, f)
    print("  Saved: ml_models/maintenance_predictor.pkl")

    # ── Model 3: Health Scorer ────────────────────────────────
    print("\n[3/4] Training Health Scorer...")
    df['health_score'] = (
        100
        - df['Vehicle_Age'].clip(0,20)               * 2.0
        - (df['Odometer_Reading']/10000).clip(0,20)  * 1.5
        - df['Accident_History'].clip(0,5)            * 6.0
        - df['Tire_Condition']                        * 8.0
        - df['Brake_Condition']                       * 8.0
        - df['Battery_Status']                        * 7.0
        - df['Reported_Issues'].clip(0,5)             * 4.0
        + df['Service_History'].clip(0,10)            * 1.0
    ).clip(0, 100).fillna(50)
    X_h = df[feat].fillna(0); y_h = df['health_score']
    X_tr, X_te, y_tr, y_te = train_test_split(X_h, y_h, test_size=0.2, random_state=42)
    scorer = Pipeline([('scaler', StandardScaler()),
                       ('reg',    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
    scorer.fit(X_tr, y_tr)
    mae = mean_absolute_error(y_te, scorer.predict(X_te))
    print(f"  MAE: {mae:.2f} pts | R²: {r2_score(y_te, scorer.predict(X_te)):.4f}")
    with open('ml_models/health_scorer.pkl', 'wb') as f:
        pickle.dump({'model': scorer, 'features': feat}, f)
    print("  Saved: ml_models/health_scorer.pkl")

    # ── Model 4: Valuation Engine ─────────────────────────────
    print("\n[4/4] Training Valuation Engine...")
    df['market_value'] = (
        500000
        - df['Vehicle_Age']       * 35000
        - df['Odometer_Reading']  * 0.8
        + df['Fuel_Efficiency']   * 5000
        + df['Service_History']   * 8000
        - df['Accident_History']  * 25000
        - (df['Tire_Condition']  == 2) * 15000
        - (df['Brake_Condition'] == 2) * 12000
        - (df['Battery_Status']  == 2) * 10000
    ).clip(50000, 2000000).fillna(400000)
    X_v = df[feat].fillna(0); y_v = df['market_value']
    X_tr, X_te, y_tr, y_te = train_test_split(X_v, y_v, test_size=0.2, random_state=42)
    val = Pipeline([('scaler', StandardScaler()),
                    ('reg',    RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1))])
    val.fit(X_tr, y_tr)
    mae = mean_absolute_error(y_te, val.predict(X_te))
    print(f"  MAE: ₹{mae:,.0f} | R²: {r2_score(y_te, val.predict(X_te)):.4f}")
    with open('ml_models/valuation_model.pkl', 'wb') as f:
        pickle.dump({'model': val, 'features': feat}, f)
    print("  Saved: ml_models/valuation_model.pkl")


if __name__ == '__main__':
    print("Dr Drive — ML Training Pipeline")
    print("=" * 50)
    train_fault_detector()
    train_vehicle_models()
    print("\n" + "=" * 50)
    print("All models trained. Copy ml_models/ to DrDriveBackend/")
    print("=" * 50)
