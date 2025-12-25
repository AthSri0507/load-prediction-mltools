#!/usr/bin/env python3
"""
One-step-ahead forecasting for Grid Supply (kW).

Models:
  - XGBoost (tabular with engineered features)
  - LSTM (sequence)
  - GRU  (sequence)

Features:
  - Calendar (hour/dayofweek/month/weekend/night)
  - Target lags and rolling stats (auto tuned to data frequency)
  - All other numeric columns as exogenous inputs

Outputs (next to CSV in energy_load_next/):
  - metrics_scaled.csv / metrics_real.csv (MAE, RMSE, SMAPE/WAPE/nRMSE)
  - preds_[xgb|lstm|gru].csv
  - plot_[xgb|lstm|gru].png (with --plots)
  - X_scaler.joblib, y_scaler.joblib, feature_cols.joblib
  - models/xgb_model.json (+ optional lstm.h5, gru.h5)

MLflow:
  - Use --mlflow to log; set --mlflow_uri or env var MLFLOW_TRACKING_URI
  - Experiment via --experiment (default: grid-load-nextstep)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import math

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

# MLflow & saving
import mlflow
import mlflow.xgboost
import joblib


# -------------------- robust metrics --------------------
def rmse_compat(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return mean_squared_error(y_true, y_pred) ** 0.5

def smape_masked(y_true, y_pred, min_abs=None):
    """SMAPE (%) only where |y_true| >= min_abs (default: 5th pct or 1e-3)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if min_abs is None:
        min_abs = max(np.percentile(np.abs(y_true), 5), 1e-3)
    mask = np.abs(y_true) >= min_abs
    if mask.sum() == 0:
        return np.nan
    yt = y_true[mask]; yp = y_pred[mask]
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    denom = np.where(denom < 1e-6, 1e-6, denom)
    return 100.0 * np.mean(np.abs(yp - yt) / denom)

def wape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom < 1e-6: return np.nan
    return 100.0 * np.sum(np.abs(y_pred - y_true)) / denom

def nrmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    rng = np.max(y_true) - np.min(y_true)
    if rng < 1e-6: return np.nan
    return 100.0 * rmse_compat(y_true, y_pred) / rng

def eval_all(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse_compat(y_true, y_pred),
        "SMAPE_%": smape_masked(y_true, y_pred),
        "WAPE_%": wape(y_true, y_pred),
        "nRMSE_%": nrmse(y_true, y_pred),
    }


# -------------------- feature helpers --------------------
def infer_granularity(ts: pd.Series) -> str:
    t = pd.to_datetime(ts).sort_values()
    diffs = t.diff().dropna().values.astype("timedelta64[m]").astype(int)
    if len(diffs) == 0: return "H"
    m = int(np.median(diffs))
    if m <= 15: return "15min"
    if m <= 30: return "30min"
    if m <= 60: return "H"
    if m <= 1440: return "D"
    return "D"

def lags_and_windows(freq: str):
    if freq == "15min":   return [1,2,4,8,24,96,192,672], [8,24,96,672]
    if freq == "30min":   return [1,2,4,12,24,48,96,336], [4,24,48,336]
    if freq == "H":       return [1,2,6,12,24,48,168],    [3,24,168]
    return [1,7,14,28], [3,7,28]

def add_calendar(df, time_col):
    t = pd.to_datetime(df[time_col])
    df["hour"] = t.dt.hour
    df["dayofweek"] = t.dt.dayofweek
    df["month"] = t.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_night"] = ((df["hour"] <= 6) | (df["hour"] >= 22)).astype(int)
    return df

def add_target_lag_roll(df, target_col, lags, rolls):
    for L in lags:
        df[f"lag_{L}"] = df[target_col].shift(L)
    for W in rolls:
        df[f"roll_mean_{W}"] = df[target_col].shift(1).rolling(W).mean()
        df[f"roll_std_{W}"]  = df[target_col].shift(1).rolling(W).std()
    return df

def make_sequences(X_2d, y_1d, n_steps):
    Xs, ys = [], []
    for i in range(n_steps, len(X_2d)):
        Xs.append(X_2d[i-n_steps:i, :])
        ys.append(y_1d[i])
    return np.array(Xs), np.array(ys)

def build_lstm(input_shape):
    m = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dense(1)
    ])
    m.compile(optimizer="adam", loss="mse")
    return m

def build_gru(input_shape):
    m = Sequential([
        GRU(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(64),
        Dense(1)
    ])
    m.compile(optimizer="adam", loss="mse")
    return m

def plot_series(ts, y, yhat, title, path):
    plt.figure(figsize=(11,4))
    plt.plot(ts, y, label="Actual")
    plt.plot(ts, yhat, label="Predicted")
    plt.title(title); plt.xlabel("Time"); plt.ylabel("Grid Supply (kW)")
    plt.legend(); plt.tight_layout(); plt.savefig(path); plt.close()


# -------------------- MLflow helper (sanitize metric names) --------------------
def mlflow_log_metrics(model_name: str, metrics_dict: dict):
    for k, v in metrics_dict.items():
        if v is None:
            continue
        try:
            vf = float(v)
        except Exception:
            continue
        if math.isnan(vf) or math.isinf(vf):
            continue
        key = f"{model_name.lower()}_{k}".lower()
        key = (key.replace("%", "pct")
                   .replace("/", "_")
                   .replace(" ", "_"))
        mlflow.log_metric(key, vf)


# -------------------- main --------------------
def main():
    # parser must be defined before add_argument
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--timecol", default="Timestamp")
    ap.add_argument("--target",  default="Grid Supply (kW)")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--plots", action="store_true")
    # MLflow flags
    ap.add_argument("--mlflow", action="store_true", help="enable MLflow logging")
    ap.add_argument("--experiment", default="grid-load-nextstep", help="MLflow experiment name")
    ap.add_argument("--mlflow_uri", default=None, help="MLflow tracking URI (overrides env)")

    args = ap.parse_args()

    # MLflow setup
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    path = Path(args.csv)
    outdir = path.parent / "energy_load_next"
    (outdir / "models").mkdir(parents=True, exist_ok=True)

    # Load & fix columns
    df = pd.read_csv(path)
    if "Temperature (Â°C)" in df.columns and "Temperature (°C)" not in df.columns:
        df = df.rename(columns={"Temperature (Â°C)": "Temperature (°C)"})

    time_col = args.timecol
    target_col = args.target

    # Avoid leakage from provided predictions
    if "Predicted Load (kW)" in df.columns:
        df = df.drop(columns=["Predicted Load (kW)"])

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, target_col]).sort_values(time_col).reset_index(drop=True)

    # One-step-ahead target
    df["target_next"] = df[target_col].shift(-1)

    # Features
    df = add_calendar(df, time_col)
    freq = infer_granularity(df[time_col])
    lags, rolls = lags_and_windows(freq)
    df = add_target_lag_roll(df, target_col, lags, rolls)

    need_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("roll_")]
    df = df.dropna(subset=need_cols + ["target_next"]).reset_index(drop=True)

    exclude = {time_col, "target_next"}
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    # Split
    n = len(df); split = int(n * (1 - args.test_size))
    train_df = df.iloc[:split].copy()
    test_df  = df.iloc[split:].copy()

    X_train_raw = train_df[feature_cols].values
    X_test_raw  = test_df[feature_cols].values
    y_train_real = train_df["target_next"].values.reshape(-1, 1)
    y_test_real  = test_df["target_next"].values.reshape(-1, 1)

    X_scaler = StandardScaler()
    y_scaler = MinMaxScaler()

    X_train = X_scaler.fit_transform(X_train_raw)
    X_test  = X_scaler.transform(X_test_raw)
    y_train = y_scaler.fit_transform(y_train_real).ravel()
    y_test  = y_scaler.transform(y_test_real).ravel()

    zeros_pct = (np.abs(y_test_real.ravel()) < 1e-6).mean() * 100.0
    print(f"[diagnostic] % near-zero targets in test: {zeros_pct:.2f}%")

    # -------- XGBoost --------
    xgb = XGBRegressor(
        n_estimators=800, learning_rate=0.05, max_depth=8,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        objective="reg:squarederror", n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    pred_xgb_scaled = xgb.predict(X_test)
    pred_xgb_real = y_scaler.inverse_transform(pred_xgb_scaled.reshape(-1,1)).ravel()

    # -------- Sequences (LSTM/GRU) --------
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    n_steps = args.seq_len

    def seq(X2d, y1d, steps):
        Xs, ys = [], []
        for i in range(steps, len(X2d)):
            Xs.append(X2d[i-steps:i, :]); ys.append(y1d[i])
        return np.array(Xs), np.array(ys)

    X_seq_all, y_seq_all = seq(X_all, y_all, n_steps)
    test_start_idx = X_train.shape[0]
    tgt_idx = np.arange(n_steps, n_steps + len(y_seq_all))
    mask_tr = tgt_idx < test_start_idx
    mask_te = tgt_idx >= test_start_idx
    X_seq_tr, y_seq_tr = X_seq_all[mask_tr], y_seq_all[mask_tr]
    X_seq_te, y_seq_te = X_seq_all[mask_te], y_seq_all[mask_te]

    early = EarlyStopping(patience=8, restore_best_weights=True)

    lstm = build_lstm((n_steps, X_seq_tr.shape[-1]))
    lstm.compile(optimizer="adam", loss="mse")
    lstm.fit(X_seq_tr, y_seq_tr, epochs=args.epochs, batch_size=args.batch_size,
             validation_split=0.1, verbose=0, callbacks=[early])
    pred_lstm_scaled = lstm.predict(X_seq_te, verbose=0).ravel()
    pred_lstm_real = y_scaler.inverse_transform(pred_lstm_scaled.reshape(-1,1)).ravel()

    gru = build_gru((n_steps, X_seq_tr.shape[-1]))
    gru.compile(optimizer="adam", loss="mse")
    gru.fit(X_seq_tr, y_seq_tr, epochs=args.epochs, batch_size=args.batch_size,
            validation_split=0.1, verbose=0, callbacks=[early])
    pred_gru_scaled = gru.predict(X_seq_te, verbose=0).ravel()
    pred_gru_real = y_scaler.inverse_transform(pred_gru_scaled.reshape(-1,1)).ravel()

    # Align timestamps
    test_time = test_df[args.timecol].to_numpy()
    seq_time = test_time[n_steps : n_steps + len(pred_lstm_real)]

    # y_true arrays (real)
    y_true_xgb = y_test_real.ravel()
    y_true_seq = (y_scaler.inverse_transform(y_seq_te.reshape(-1,1)).ravel()
                  if len(y_seq_te) else np.array([]))

    # Metrics tables
    metrics_scaled = pd.DataFrame({
        "XGBoost": {"MAE": mean_absolute_error(y_test, pred_xgb_scaled),
                    "RMSE": rmse_compat(y_test, pred_xgb_scaled),
                    "SMAPE_%": smape_masked(y_test, pred_xgb_scaled),
                    "WAPE_%": wape(y_test, pred_xgb_scaled),
                    "nRMSE_%": nrmse(y_test, pred_xgb_scaled)},
        "LSTM":    ({"MAE": mean_absolute_error(y_seq_te, pred_lstm_scaled),
                     "RMSE": rmse_compat(y_seq_te, pred_lstm_scaled),
                     "SMAPE_%": smape_masked(y_seq_te, pred_lstm_scaled),
                     "WAPE_%": wape(y_seq_te, pred_lstm_scaled),
                     "nRMSE_%": nrmse(y_seq_te, pred_lstm_scaled)}
                    if len(y_seq_te) else {"MAE":np.nan,"RMSE":np.nan,"SMAPE_%":np.nan,"WAPE_%":np.nan,"nRMSE_%":np.nan}),
        "GRU":     ({"MAE": mean_absolute_error(y_seq_te, pred_gru_scaled),
                     "RMSE": rmse_compat(y_seq_te, pred_gru_scaled),
                     "SMAPE_%": smape_masked(y_seq_te, pred_gru_scaled),
                     "WAPE_%": wape(y_seq_te, pred_gru_scaled),
                     "nRMSE_%": nrmse(y_seq_te, pred_gru_scaled)}
                    if len(y_seq_te) else {"MAE":np.nan,"RMSE":np.nan,"SMAPE_%":np.nan,"WAPE_%":np.nan,"nRMSE_%":np.nan})
    }).T

    metrics_real = pd.DataFrame({
        "XGBoost": eval_all(y_true_xgb, pred_xgb_real),
        "LSTM":    (eval_all(y_true_seq, pred_lstm_real) if len(y_true_seq) else {"MAE":np.nan,"RMSE":np.nan,"SMAPE_%":np.nan,"WAPE_%":np.nan,"nRMSE_%":np.nan}),
        "GRU":     (eval_all(y_true_seq, pred_gru_real)  if len(y_true_seq)  else {"MAE":np.nan,"RMSE":np.nan,"SMAPE_%":np.nan,"WAPE_%":np.nan,"nRMSE_%":np.nan})
    }).T

    # Save metrics & predictions
    outdir.mkdir(exist_ok=True, parents=True)
    metrics_scaled.to_csv(outdir / "metrics_scaled.csv")
    metrics_real.to_csv(outdir / "metrics_real.csv")

    pd.DataFrame({
        "timestamp": test_time[:len(y_true_xgb)],
        "y_true": y_true_xgb[:len(y_true_xgb)],
        "y_pred": pred_xgb_real[:len(y_true_xgb)]
    }).to_csv(outdir / "preds_xgb.csv", index=False)

    if len(y_true_seq):
        pd.DataFrame({
            "timestamp": seq_time,
            "y_true": y_true_seq[:len(seq_time)],
            "y_pred": pred_lstm_real[:len(seq_time)]
        }).to_csv(outdir / "preds_lstm.csv", index=False)

        pd.DataFrame({
            "timestamp": seq_time,
            "y_true": y_true_seq[:len(seq_time)],
            "y_pred": pred_gru_real[:len(seq_time)]
        }).to_csv(outdir / "preds_gru.csv", index=False)

    # Plots
    if args.plots:
        plot_series(test_time[:len(y_true_xgb)], y_true_xgb, pred_xgb_real[:len(y_true_xgb)],
                    "XGBoost: Actual vs Pred (next step)", outdir / "plot_xgb.png")
        if len(y_true_seq):
            plot_series(seq_time, y_true_seq[:len(seq_time)], pred_lstm_real[:len(seq_time)],
                        "LSTM: Actual vs Pred (next step)", outdir / "plot_lstm.png")
            plot_series(seq_time, y_true_seq[:len(seq_time)], pred_gru_real[:len(seq_time)],
                        "GRU: Actual vs Pred (next step)", outdir / "plot_gru.png")

    # Persist artifacts for inference & MLflow
    joblib.dump(X_scaler, outdir / "X_scaler.joblib")
    joblib.dump(y_scaler, outdir / "y_scaler.joblib")
    joblib.dump(feature_cols, outdir / "feature_cols.joblib")
    xgb.get_booster().save_model(str(outdir / "models" / "xgb_model.json"))
    try:
        lstm.save(outdir / "models" / "lstm.h5")
        gru.save(outdir / "models" / "gru.h5")
    except Exception:
        pass

    # MLflow logging (sanitized metric names)
    if args.mlflow:
        with mlflow.start_run(run_name="nextstep-forecast"):
            mlflow.log_params({
                "model_xgb": True,
                "n_estimators": 800,
                "max_depth": 8,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "seq_len": args.seq_len,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "test_size": args.test_size,
            })
            if "XGBoost" in metrics_real.index:
                mlflow_log_metrics("xgboost", metrics_real.loc["XGBoost"].to_dict())
            if "LSTM" in metrics_real.index:
                mlflow_log_metrics("lstm", metrics_real.loc["LSTM"].to_dict())
            if "GRU" in metrics_real.index:
                mlflow_log_metrics("gru", metrics_real.loc["GRU"].to_dict())

            # artifacts
            for fname in [
                "metrics_real.csv","metrics_scaled.csv",
                "preds_xgb.csv","preds_lstm.csv","preds_gru.csv",
                "plot_xgb.png","plot_lstm.png","plot_gru.png",
                "X_scaler.joblib","y_scaler.joblib","feature_cols.joblib",
                "models/xgb_model.json","models/lstm.h5","models/gru.h5"
            ]:
                p = outdir / fname
                if p.exists():
                    mlflow.log_artifact(str(p))

            # register/log XGBoost
            try:
                mlflow.xgboost.log_model(
                    xgb, artifact_path="xgb_model_mlflow",
                    registered_model_name="grid-load-nextstep-xgb"
                )
            except Exception:
                mlflow.xgboost.log_model(xgb, artifact_path="xgb_model_mlflow")

    # Console summary
    print("\n=== Metrics (scaled) ===")
    print(metrics_scaled.round(6))
    print("\n=== Metrics (real units) ===")
    print(metrics_real.round(6))
    print(f"\nArtifacts saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
