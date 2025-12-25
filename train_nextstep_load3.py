#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor, XGBClassifier

import matplotlib.pyplot as plt
import joblib

# ---------- TensorFlow ----------
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

# ---------- Evidently ----------
from evidently.report import Report
from evidently.metric_preset import RegressionPreset, DataDriftPreset


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
    ap.add_argument("--plots", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.csv).parent / "energy_load_next"
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------- LOAD ----------
    df = pd.read_csv(args.csv)
    df[args.timecol] = pd.to_datetime(df[args.timecol], errors="coerce")
    df = df.dropna(subset=[args.timecol, args.target])
    df = df.sort_values(args.timecol).reset_index(drop=True)

    # ---------- TARGET (LOG) ----------
    df["target_next"] = np.log1p(df[args.target].shift(-1))

    # ---------- FEATURES ----------
    df = add_calendar(df, args.timecol)
    df = add_lags(df, args.target)
    df = df.dropna().reset_index(drop=True)

    split = int(len(df) * (1 - args.test_size))

    # Spike label (train-only threshold)
    spike_thr = np.percentile(df.iloc[:split]["target_next"], 90)
    df["is_spike"] = (df["target_next"] >= spike_thr).astype(int)

    train_df = df.iloc[:split].copy()
    test_df  = df.iloc[split:].copy()

    feature_cols = [
        c for c in df.columns
        if c not in {args.timecol, args.target, "target_next", "is_spike"}
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

    # =====================================================
    #               XGBOOST (2-STAGE)
    # =====================================================
    base_reg = XGBRegressor(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1
    )
    base_reg.fit(X_train, y_train)

    spike_idx = train_df["is_spike"].values == 1
    spike_reg = None

    if spike_idx.sum() > 5:
        spike_reg = XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        )
        spike_reg.fit(X_train[spike_idx], y_train[spike_idx])

    clf = None
    if spike_reg is not None and len(np.unique(train_df["is_spike"])) > 1:
        clf = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            scale_pos_weight=6,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, train_df["is_spike"].values)

    base_pred = base_reg.predict(X_test)
    final_scaled = base_pred.copy()

    if clf is not None and spike_reg is not None:
        mask = clf.predict(X_test) == 1
        if mask.any():
            final_scaled[mask] = spike_reg.predict(X_test[mask])

    final_log = y_scaler.inverse_transform(final_scaled.reshape(-1,1)).ravel()
    final_pred = np.expm1(final_log)
    y_test_real = np.expm1(y_test_log)

    # =====================================================
    #               LSTM / GRU (BASELINES)
    # =====================================================
    seq_len = 128
    X_seq_tr, y_seq_tr = make_sequences(X_train, y_train, seq_len)
    X_seq_te, y_seq_te = make_sequences(X_test, y_test_log, seq_len)

    early = EarlyStopping(patience=8, restore_best_weights=True)

    lstm = build_lstm((seq_len, X_seq_tr.shape[-1]))
    lstm.fit(X_seq_tr, y_seq_tr, epochs=40, batch_size=64,
             validation_split=0.1, callbacks=[early], verbose=0)

    gru = build_gru((seq_len, X_seq_tr.shape[-1]))
    gru.fit(X_seq_tr, y_seq_tr, epochs=40, batch_size=64,
            validation_split=0.1, callbacks=[early], verbose=0)

    lstm_log = y_scaler.inverse_transform(
        lstm.predict(X_seq_te).reshape(-1,1)
    ).ravel()
    gru_log = y_scaler.inverse_transform(
        gru.predict(X_seq_te).reshape(-1,1)
    ).ravel()

    lstm_pred = np.clip(np.expm1(lstm_log), 0, None)
    gru_pred  = np.clip(np.expm1(gru_log), 0, None)
    y_seq_real = np.expm1(y_seq_te)

    # =====================================================
    #               EVIDENTLY AI
    # =====================================================
    train_pred_log = y_scaler.inverse_transform(
        base_reg.predict(X_train).reshape(-1,1)
    ).ravel()
    train_pred = np.expm1(train_pred_log)

    reference_df = train_df.copy()
    reference_df["prediction"] = train_pred

    current_df = test_df.copy()
    current_df["prediction"] = final_pred

    column_mapping = {
        "target": "target_next",
        "prediction": "prediction",
        "numerical_features": feature_cols
    }

    reg_report = Report(metrics=[RegressionPreset()])
    reg_report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping
    )
    reg_report.save_html(outdir / "evidently_regression_report.html")

    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=column_mapping
    )
    drift_report.save_html(outdir / "evidently_drift_report.html")

    # =====================================================
    #               SAVE ARTIFACTS
    # =====================================================
    joblib.dump(base_reg, outdir / "xgb_baseline.joblib")
    if spike_reg is not None:
        joblib.dump(spike_reg, outdir / "xgb_spike.joblib")
    if clf is not None:
        joblib.dump(clf, outdir / "xgb_spike_classifier.joblib")

    joblib.dump(X_scaler, outdir / "X_scaler.joblib")
    joblib.dump(y_scaler, outdir / "y_scaler.joblib")
    joblib.dump(feature_cols, outdir / "feature_cols.joblib")

    # =====================================================
    #               PLOTS
    # =====================================================
    if args.plots:
        t = test_df[args.timecol].iloc[seq_len:]

        plt.figure(figsize=(12,4))
        plt.plot(t, y_seq_real, label="Actual")
        plt.plot(t, final_pred[seq_len:], label="XGBoost")
        plt.plot(t, lstm_pred, label="LSTM")
        plt.plot(t, gru_pred, label="GRU")
        plt.legend()
        plt.title("Next-step Load Forecasting with Monitoring")
        plt.tight_layout()
        plt.savefig(outdir / "all_models_comparison.png")
        plt.close()

    print("\n=== TRAINING + EVIDENTLY COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
