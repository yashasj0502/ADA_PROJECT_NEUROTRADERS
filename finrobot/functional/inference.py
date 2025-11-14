# finrobot/functional/inference.py
"""
Example inference:
- load saved model
- fetch latest numeric and text
- construct features (same pipeline) and predict
- produce textual summary (top contributing news by similarity)
"""

import joblib
import pandas as pd
import numpy as np
import os
from finrobot.data_source import news_utils, multimodal_loader
from finrobot.functional import text_models, fusion, forecast_model
from sklearn.metrics.pairwise import cosine_similarity

MODEL_PATH = os.getenv("MODEL_PATH", "models/lgbm_finrobot.pkl")

def load_model(path=MODEL_PATH):
    return joblib.load(path)

def find_top_relevant_articles(latest_embedding, articles_df, topk=3):
    # articles_df must have 'embedding' column
    emb_matrix = np.stack([e for e in articles_df["embedding"].fillna(np.zeros_like(latest_embedding)).tolist()])
    sims = cosine_similarity([latest_embedding], emb_matrix)[0]
    idx = np.argsort(-sims)[:topk]
    return articles_df.iloc[idx]

def main():
    # load model
    model = load_model()

    # small numeric fetch (placeholder; user should use real loader)
    numeric_df = pd.DataFrame()  # replace with loader
    # fetch recent news
    news_df = news_utils.fetch_all()
    news_df = news_utils.canonicalize(news_df)
    # take last N hours -> embed
    tm = text_models.TextModels()
    news_enriched = tm.enrich_dataframe(news_df, text_col="text")
    # assume numeric_df already aligned with the news times (use multimodal_loader in production)
    # For demo: take most recent news embedding, find similar news
    if not news_enriched.empty:
        latest_emb = news_enriched.iloc[0]["embedding"]
        top = find_top_relevant_articles(latest_emb, news_enriched, topk=3)
        print("Top relevant articles:\n", top[["title","url","timestamp"]])
    else:
        print("No news found")

if __name__ == "__main__":
    main()
