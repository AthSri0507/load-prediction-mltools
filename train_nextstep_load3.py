#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import argparse

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

import joblib

# ---------- TensorFlow ----------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

# ---------- Evidently (v0.6) ----------
try:
    from evidently.report import Report
    from evidently.metrics import (
        RegressionQualityMetric,
        RegressionErrorDistribution,
        DataDriftTable
    )
    EVIDENTLY_AVAILABLE = True
except Exception as e:
    print("Evidently import failed:", e)
    EVIDENTLY_AVAILABLE = False


# ================= METRICS =================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ================= FEATURES =================
def add_calendar(df, tcol):
    t = pd.to_datetime(df[tcol])
    df["hour"] = t.dt.hour
    df["dayofweek"] = t.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df

def add_lags(df, target):
    for l in [1, 2, 6, 12, 24]:
        df[f"lag_{l}"] = df[target].shift(l)

    for w in [6, 24]:
        s = df[target].shift(1)
        df[f"roll_mean_{w}"] = s.rolling(w).mean()
        df[f"roll_max_{w}"] = s.rolling(w).max()
        df[f"roll_std_{w}"] = s.rolling(w).std()

    df["delta_1"] = df[target] - df[target].shift(1)
    return df


# ================= SEQUENCES =================
def make_sequences(X, y, steps):
    Xs, ys = [], []
    for i in range(steps, len(X)):
        Xs.append(X[i-steps:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


# ================= RNN MODELS =================
def build_lstm(input_shape):
    m = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    m.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss=Huber(delta=1.0)
    )
    return m

def build_gru(input_shape):
    m = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(32),
        Dense(1)
    ])
    m.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss=Huber(delta=1.0)
    )
    return m


# ================= MAIN =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--timecol", default="Timestamp")
    ap.add_argument("--target", default="Grid Supply (kW)")
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()

    outdir = Path(args.csv).parent / "energy_load_next"
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------- LOAD ----------
    df = pd.read_csv(args.csv)
    df[args.timecol] = pd.to_datetime(df[args.timecol], errors="coerce")
    df = df.dropna(subset=[args.timecol, args.target])
    df = df.sort_values(args.timecol).reset_index(drop=True)

    # ---------- TARGET ----------
    df["target_next"] = np.log1p(df[args.target].shift(-1))

    # ---------- FEATURES ----------
    df = add_calendar(df, args.timecol)
    df = add_lags(df, args.target)
    df = df.dropna().reset_index(drop=True)

    split = int(len(df) * (1 - args.test_size))

    train_df = df.iloc[:split].copy()
    test_df  = df.iloc[split:].copy()

    feature_cols = [
        c for c in df.columns
        if c not in {args.timecol, args.target, "target_next"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    X_train_raw = train_df[feature_cols].values
    X_test_raw  = test_df[feature_cols].values
    y_train_log = train_df["target_next"].values
    y_test_log  = test_df["target_next"].values

    # ---------- SCALING ----------
    X_scaler = StandardScaler()
    y_scaler = MinMaxScaler()

    X_train = X_scaler.fit_transform(X_train_raw)
    X_test  = X_scaler.transform(X_test_raw)
    y_train = y_scaler.fit_transform(y_train_log.reshape(-1,1)).ravel()

    # ================= XGBOOST =================
    base_reg = XGBRegressor(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1
    )
    base_reg.fit(X_train, y_train)

    base_pred = base_reg.predict(X_test)
    final_log = y_scaler.inverse_transform(base_pred.reshape(-1,1)).ravel()
    final_pred = np.expm1(final_log)
    y_test_real = np.expm1(y_test_log)

    # ================= EVIDENTLY =================
    if EVIDENTLY_AVAILABLE:
        reference_df = train_df.copy()
        reference_df["target"] = np.expm1(train_df["target_next"])
        reference_df["prediction"] = np.expm1(
            y_scaler.inverse_transform(
                base_reg.predict(X_train).reshape(-1,1)
            ).ravel()
        )

        current_df = test_df.copy()
        current_df["target"] = y_test_real
        current_df["prediction"] = final_pred

        try:
            reg_report = Report(metrics=[
                RegressionQualityMetric(),
                RegressionErrorDistribution()
            ])
            reg_report.run(reference_data=reference_df, current_data=current_df)
            reg_report.save_html(
                str(outdir / "evidently_regression_report.html")
            )
            print("✓ Evidently regression report generated")

            drift_report = Report(metrics=[
                DataDriftTable()
            ])
            drift_report.run(reference_data=reference_df, current_data=current_df)
            drift_report.save_html(
                str(outdir / "evidently_drift_report.html")
            )
            print("✓ Evidently data drift report generated")

        except Exception as e:
            print("Evidently runtime error:", e)

    else:
        print("Evidently not available — skipped")

    # ================= SAVE =================
    joblib.dump(base_reg, outdir / "xgb_model.joblib")
    joblib.dump(X_scaler, outdir / "X_scaler.joblib")
    joblib.dump(y_scaler, outdir / "y_scaler.joblib")
    joblib.dump(feature_cols, outdir / "feature_cols.joblib")

    print("\n=== TRAINING + EVIDENTLY COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
