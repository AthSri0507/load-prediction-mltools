#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import time
import joblib

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

# ================= MLFLOW =================
import mlflow
import mlflow.xgboost
import mlflow.tensorflow

# ================= PROMETHEUS =================
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


# ================= METRIC =================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ================= FEATURE ENGINEERING =================
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
        df[f"roll_std_{w}"] = s.rolling(w).std()
    return df


# ================= SEQUENCE BUILDER =================
def make_sequences(X, y, steps):
    Xs, ys = [], []
    for i in range(steps, len(X)):
        Xs.append(X[i-steps:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


# ================= MODELS =================
def build_lstm(shape, lr):
    m = Sequential([
        LSTM(64, return_sequences=True, input_shape=shape),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    m.compile(optimizer=Adam(lr), loss=Huber())
    return m


def build_gru(shape, lr):
    m = Sequential([
        GRU(64, return_sequences=True, input_shape=shape),
        Dropout(0.2),
        GRU(32),
        Dense(1)
    ])
    m.compile(optimizer=Adam(lr), loss=Huber())
    return m


# ================= PROMETHEUS PUSH =================
def push_metrics(xgb_rmse, lstm_rmse, gru_rmse):
    registry = CollectorRegistry()
    Gauge("rmse_xgb", "XGBoost RMSE", registry=registry).set(xgb_rmse)
    Gauge("rmse_lstm", "LSTM RMSE", registry=registry).set(lstm_rmse)
    Gauge("rmse_gru", "GRU RMSE", registry=registry).set(gru_rmse)
    Gauge("training_timestamp", "Training timestamp", registry=registry).set(time.time())
    push_to_gateway("pushgateway:9091", job="mlops", registry=registry)


# ================= MAIN =================
def main():
    # -------- MLflow setup (Docker-safe) --------
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("grid-load-nextstep")

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--timecol", default="Timestamp")
    ap.add_argument("--target", default="Grid Supply (kW)")
    args = ap.parse_args()

    outdir = Path("energy_load_next")
    outdir.mkdir(exist_ok=True)

    # -------- LOAD DATA --------
    df = pd.read_csv(args.csv)
    df[args.timecol] = pd.to_datetime(df[args.timecol])
    df = df.sort_values(args.timecol)

    df["target_next"] = np.log1p(df[args.target].shift(-1))
    df = add_calendar(df, args.timecol)
    df = add_lags(df, args.target)
    df.dropna(inplace=True)

    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    feature_cols = [c for c in df.columns if c not in
                    {args.timecol, args.target, "target_next"}]

    X_train_raw, X_test_raw = train[feature_cols], test[feature_cols]
    y_train_log, y_test_log = train["target_next"], test["target_next"]

    X_scaler = StandardScaler()
    y_scaler = MinMaxScaler()

    X_train = X_scaler.fit_transform(X_train_raw)
    X_test = X_scaler.transform(X_test_raw)
    y_train = y_scaler.fit_transform(y_train_log.values.reshape(-1, 1)).ravel()

    y_test_real = np.expm1(y_test_log.values)

    # ================= XGBOOST HPO =================
    xgb_param_grid = [
        {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05},
        {"n_estimators": 600, "max_depth": 7, "learning_rate": 0.05},
        {"n_estimators": 600, "max_depth": 8, "learning_rate": 0.03},
    ]

    best_rmse = float("inf")

    for i, params in enumerate(xgb_param_grid):
        with mlflow.start_run(run_name=f"xgb_run_{i+1}", nested=True):

            full_params = {
                **params,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "random_state": 42
            }
            mlflow.log_params(full_params)

            model = XGBRegressor(**full_params, n_jobs=-1)
            model.fit(X_train, y_train)

            preds_log = y_scaler.inverse_transform(
                model.predict(X_test).reshape(-1, 1)
            ).ravel()
            preds = np.expm1(preds_log)

            score = rmse(y_test_real, preds)
            mlflow.log_metric("rmse", score)

            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                registered_model_name="GridLoad_XGBoost"
            )

            if score < best_rmse:
                best_rmse = score
                best_xgb_rmse = score

    # ================= LSTM =================
    seq_len = 96
    X_seq_tr, y_seq_tr = make_sequences(X_train, y_train, seq_len)
    y_test_scaled = y_scaler.transform(y_test_log.values.reshape(-1, 1)).ravel()
    X_seq_te, y_seq_te = make_sequences(X_test, y_test_scaled, seq_len)
    y_seq_real = np.expm1(y_scaler.inverse_transform(y_seq_te.reshape(-1, 1)).ravel())

    lstm_params = {
        "model": "LSTM",
        "sequence_length": seq_len,
        "epochs": 25,
        "batch_size": 64,
        "optimizer": "Adam",
        "learning_rate": 3e-4,
        "loss": "Huber"
    }

    with mlflow.start_run(run_name="lstm_baseline", nested=True):
        mlflow.log_params(lstm_params)

        lstm = build_lstm((seq_len, X_seq_tr.shape[-1]), 3e-4)
        lstm.fit(
            X_seq_tr, y_seq_tr,
            epochs=25,
            batch_size=64,
            validation_split=0.1,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0
        )

        preds = np.expm1(
            y_scaler.inverse_transform(lstm.predict(X_seq_te)).ravel()
        )
        lstm_rmse = rmse(y_seq_real, preds)
        mlflow.log_metric("rmse", lstm_rmse)

        mlflow.tensorflow.log_model(
            lstm,
            artifact_path="model",
            registered_model_name="GridLoad_LSTM"
        )

    # ================= GRU =================
    gru_params = {
        "model": "GRU",
        "sequence_length": seq_len,
        "epochs": 25,
        "batch_size": 64,
        "optimizer": "Adam",
        "learning_rate": 3e-4,
        "loss": "Huber"
    }

    with mlflow.start_run(run_name="gru_baseline", nested=True):
        mlflow.log_params(gru_params)

        gru = build_gru((seq_len, X_seq_tr.shape[-1]), 3e-4)
        gru.fit(
            X_seq_tr, y_seq_tr,
            epochs=25,
            batch_size=64,
            validation_split=0.1,
            callbacks=[EarlyStopping(patience=4, restore_best_weights=True)],
            verbose=0
        )

        preds = np.expm1(
            y_scaler.inverse_transform(gru.predict(X_seq_te)).ravel()
        )
        gru_rmse = rmse(y_seq_real, preds)
        mlflow.log_metric("rmse", gru_rmse)

        mlflow.tensorflow.log_model(
            gru,
            artifact_path="model",
            registered_model_name="GridLoad_GRU"
        )

    # ================= ARTIFACTS =================
    joblib.dump(X_scaler, outdir / "X_scaler.joblib")
    joblib.dump(y_scaler, outdir / "y_scaler.joblib")
    joblib.dump(feature_cols, outdir / "feature_cols.joblib")

    mlflow.log_artifact(str(outdir / "X_scaler.joblib"))
    mlflow.log_artifact(str(outdir / "y_scaler.joblib"))
    mlflow.log_artifact(str(outdir / "feature_cols.joblib"))

    push_metrics(best_xgb_rmse, lstm_rmse, gru_rmse)

    print("\n=== TRAINING + HPO + MLFLOW + GRAFANA COMPLETED ===")


if __name__ == "__main__":
    main()
