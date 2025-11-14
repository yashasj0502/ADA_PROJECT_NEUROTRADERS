# finrobot/functional/text_models.py
"""
Wrappers for text embeddings and sentiment analysis (FinBERT / SentenceTransformer).
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Default models (can override with env vars)
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_SENT_MODEL = os.getenv("SENT_MODEL", "yiyanghkust/finbert-tone")

class TextModels:
    def __init__(self, embed_model_name=None, sentiment_model_name=None, device=-1):
        self.embed_model_name = embed_model_name or DEFAULT_EMBED_MODEL
        self.sentiment_model_name = sentiment_model_name or DEFAULT_SENT_MODEL

        print(f"Loading embedding model: {self.embed_model_name}")
        self.embedder = SentenceTransformer(self.embed_model_name)

        print(f"Loading sentiment model: {self.sentiment_model_name}")
        try:
            self.sentiment_pipeline = pipeline("sentiment-analysis", model=self.sentiment_model_name, device=device)
        except Exception as e:
            print("Falling back to default sentiment pipeline:", e)
            self.sentiment_pipeline = pipeline("sentiment-analysis", device=device)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        texts = [t if isinstance(t, str) else "" for t in texts]
        return self.embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    def sentiment_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.sentiment_pipeline(batch)
            results.extend(batch_results)
        return results

    def enrich_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        df = df.copy()
        texts = df[text_col].fillna("").tolist()
        print("Generating embeddings and sentiment...")
        df["embedding"] = list(self.embed_texts(texts))
        df["sentiment"] = self.sentiment_batch(texts)
        return df
