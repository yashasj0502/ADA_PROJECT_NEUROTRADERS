# finrobot/functional/fusion.py

import pandas as pd
import numpy as np

def pool_embeddings(embeddings, method="mean"):
    """Aggregate list of embedding vectors."""
    if not embeddings:
        return np.zeros(384)
    arr = np.stack(embeddings)
    if method == "mean":
        return np.nanmean(arr, axis=0)
    elif method == "max":
        return np.nanmax(arr, axis=0)
    else:
        return np.nanmean(arr, axis=0)


def attach_text_features_to_numeric(numeric_df: pd.DataFrame,
                                    text_df: pd.DataFrame,
                                    window="1h",
                                    embed_col="embedding",
                                    sentiment_col="sentiment"):
    """
    For each numeric timestamp, aggregate text embeddings and sentiment scores nearby.
    Returns fused dataframe (numeric + text features).
    """
    numeric_df = numeric_df.copy()
    text_df = text_df.copy()

    text_df["timestamp"] = pd.to_datetime(text_df["timestamp"], utc=True)
    numeric_df["timestamp"] = pd.to_datetime(numeric_df["timestamp"], utc=True)

    merged_rows = []

    for ts in numeric_df["timestamp"]:
        subset = text_df[text_df["timestamp"] == ts]
        if subset.empty:
            merged_rows.append({
                "timestamp": ts,
                "txt_emb_mean": 0,
                "txt_emb_std": 0,
                "txt_sent_pos": 0,
                "txt_sent_neg": 0,
                "txt_sent_neu": 0
            })
        else:
            # Pool embeddings
            emb_list = [e for e in subset[embed_col].tolist() if e is not None]
            pooled = pool_embeddings(emb_list)
            emb_mean = np.mean(pooled)
            emb_std = np.std(pooled)

            # Aggregate sentiment
            pos, neg, neu = 0, 0, 0
            for s in subset[sentiment_col]:
                label = s.get("label", "").upper()
                score = s.get("score", 0)
                if "POS" in label:
                    pos += score
                elif "NEG" in label:
                    neg += score
                else:
                    neu += score

            merged_rows.append({
                "timestamp": ts,
                "txt_emb_mean": emb_mean,
                "txt_emb_std": emb_std,
                "txt_sent_pos": pos,
                "txt_sent_neg": neg,
                "txt_sent_neu": neu
            })

    text_features = pd.DataFrame(merged_rows)
    fused_df = pd.merge(numeric_df, text_features, on="timestamp", how="left").fillna(0)
    return fused_df
