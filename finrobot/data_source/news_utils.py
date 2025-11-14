# finrobot/data_source/news_utils.py
"""
Fetch and canonicalize financial news / RSS feeds.
Returns records: {'source','title','body','published' (pd.Timestamp),'url'}
"""

import feedparser
import requests
from dateutil import parser as dateparser
import pandas as pd
from typing import List, Dict
import os
from tqdm import tqdm

DEFAULT_FEEDS = [
    # Users should override or extend via env var NEWS_SOURCES (comma separated)
    "https://www.reuters.com/finance/markets/rss",
    "https://www.ft.com/?format=rss",
]

def get_feed_list() -> List[str]:
    env = os.getenv("NEWS_SOURCES", "")
    if env:
        return [u.strip() for u in env.split(",") if u.strip()]
    return DEFAULT_FEEDS

def fetch_feed(url: str) -> List[Dict]:
    parsed = feedparser.parse(url)
    items = []
    for e in parsed.entries:
        try:
            published = None
            if hasattr(e, "published"):
                published = dateparser.parse(e.published)
            elif hasattr(e, "updated"):
                published = dateparser.parse(e.updated)
            content = getattr(e, "summary", "") or getattr(e, "description", "")
            items.append({
                "source": url,
                "title": getattr(e, "title", ""),
                "body": content,
                "published": pd.to_datetime(published) if published else pd.NaT,
                "url": getattr(e, "link", "")
            })
        except Exception:
            continue
    return items

def fetch_all(feeds: List[str] = None) -> pd.DataFrame:
    feeds = feeds or get_feed_list()
    rows = []
    for f in tqdm(feeds, desc="fetching rss feeds"):
        rows.extend(fetch_feed(f))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["title", "url"]).reset_index(drop=True)
    return df

def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize columns, drop NaT timestamps
    df = df.rename(columns={"published": "timestamp"})
    if "timestamp" in df.columns:
        df = df[~df["timestamp"].isna()]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        df["timestamp"] = pd.NaT
    df["text"] = (df["title"].fillna("") + ". " + df["body"].fillna("")).str.strip()
    cols = ["source", "title", "body", "text", "timestamp", "url"]
    return df[[c for c in cols if c in df.columns]]

if __name__ == "__main__":
    df = fetch_all()
    print(f"Fetched {len(df)} articles")
