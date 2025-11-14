# finrobot/functional/forecast_model.py
"""
Wrapper for training a LightGBM baseline using numeric + text features.
Also contains a simple LSTM/Transformer skeleton if you want deep models.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import joblib
import os

MODEL_DIR = os.getenv("MODEL_DIR", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def prepare_features(df: pd.DataFrame, target_col="close", lookahead=1):
    """
    Example:
      - Creates target as future return or future price shift.
      - Flattens any embedding arrays into numeric columns via PCA/mean (user choice).
    """
    df = df.copy().sort_values("timestamp")
    df["target"] = df[target_col].shift(-lookahead)  # predict next close
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    # numeric features: price returns, vol, etc.
    df["ret_1"] = df["close"].pct_change().fillna(0)
    df["ret_5"] = df["close"].pct_change(5).fillna(0)
    # handle text embedding flattening: compute mean and std of embedding vector
    if "txt_embedding" in df.columns:
        emb_stats = df["txt_embedding"].apply(lambda x: (np.nanmean(x), np.nanstd(x)) if x is not None else (0.0,0.0))
        df["txt_emb_mean"] = emb_stats.apply(lambda t: t[0])
        df["txt_emb_std"] = emb_stats.apply(lambda t: t[1])
    # drop columns that aren't features
    features = ["open","high","low","close","volume","ret_1","ret_5","txt_emb_mean","txt_emb_std"]
    features = [c for c in features if c in df.columns]
    X = df[features].fillna(0)
    y = df["target"]
    return X, y, df

def train_lightgbm(X, y, params=None, model_name="lgbm_baseline"):
    params = params or {
        "objective":"regression",
        "metric":"rmse",
        "verbosity": -1,
        "boosting_type":"gbdt",
    }
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    dtrain = lgb.Dataset(X_train, y_train)
    dval = lgb.Dataset(X_val, y_val, reference=dtrain)
    callbacks = [lgb.early_stopping(stopping_rounds=50),lgb.log_evaluation(period=50)]
    model = lgb.train(params,dtrain,num_boost_round=1000,valid_sets=[dtrain, dval],callbacks=callbacks)
    fname = os.path.join(MODEL_DIR, model_name + ".pkl")
    joblib.dump(model, fname)
    return model, fname

def evaluate_model(model, X, y):
    preds = model.predict(X, num_iteration=model.best_iteration)
    rmse = mean_squared_error(y, preds, squared=False)
    return {"rmse": rmse}
