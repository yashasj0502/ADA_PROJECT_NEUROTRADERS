# finrobot/functional/train.py
"""
Example pipeline:
1) load numeric timeseries from finrobot/data_source/* (existing yfinance_utils etc)
2) fetch news + reddit, embed and sentiment
3) align and fuse -> produce training set
4) train LightGBM baseline
"""

import os
import pandas as pd
from finrobot.data_source import multimodal_loader, news_utils, reddit_utils
from finrobot.functional import text_models, fusion, forecast_model

# --- User: replace with your numeric loader (yfinance_utils etc)
def load_numeric_example():
    # This is a placeholder. Replace with finrobot.data_source.yfinance_utils.fetch_ohlcv(...)
    # Should produce DataFrame with columns ['timestamp','open','high','low','close','volume']
    import numpy as np
    rng = pd.date_range("2022-01-01", periods=500, freq="h", tz="UTC")
    price = 100 + np.cumsum(np.random.randn(len(rng))*0.5)
    df = pd.DataFrame({"timestamp": rng, "open": price, "high": price+0.1, "low": price-0.1, "close": price, "volume": np.random.randint(100,1000,len(rng))})
    return df

def main():
    numeric_df = load_numeric_example()
    # 1. fetch news + reddit
    news_df = news_utils.fetch_all()
    news_df = news_utils.canonicalize(news_df)
    # optional: reddit - requires credentials
    reddit_df = pd.DataFrame()
    try:
        sub = os.getenv("REDDIT_SUBREDDIT", "")
        if sub:
            reddit_df = reddit_utils.fetch_subreddit_with_comments(sub, post_limit=50, comment_limit=200)
    except Exception as e:
        print("reddit fetch failed:", e)

    # Handle missing Reddit data gracefully
    news_part = news_df[["text", "timestamp"]] if "text" in news_df.columns else pd.DataFrame(columns=["text","timestamp"])
    reddit_part = reddit_df[["text", "timestamp"]] if ("text" in reddit_df.columns and "timestamp" in reddit_df.columns) else pd.DataFrame(columns=["text","timestamp"])

    texts = pd.concat([news_part, reddit_part], ignore_index=True, sort=False)
    texts = texts.dropna(subset=["text"]).reset_index(drop=True)

    # 2. align into 1H windows
    num_res, txt_res = multimodal_loader.align_numeric_and_text(numeric_df, texts, window="1H")

    # 3. text embedding + sentiment
    tm = text_models.TextModels()
    txt_enriched = tm.enrich_dataframe(txt_res, text_col="text")

    # 4. fuse
    fused = fusion.attach_text_features_to_numeric(num_res, txt_enriched, window="1H")

    # 5. prepare features and train
    X, y, df = forecast_model.prepare_features(fused, target_col="close", lookahead=1)
    model, path = forecast_model.train_lightgbm(X, y, model_name="lgbm_finrobot")
    print("Saved model to", path)
    print("Eval:", forecast_model.evaluate_model(model, X, y))

if __name__ == "__main__":
    main()
