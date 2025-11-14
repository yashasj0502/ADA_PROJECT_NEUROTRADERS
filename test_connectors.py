from finrobot.utils import register_keys_from_json
from finrobot.data_source.reddit_utils import RedditUtils
from finrobot.data_source import news_utils  # <-- changed line

register_keys_from_json("finrobot/config_api_keys.json")

print("\n🧠 Testing Reddit connector...")
try:
    reddit_df = RedditUtils.get_reddit_posts(
        query="AAPL",
        start_date="2023-05-01",
        end_date="2023-06-01",
        limit=5
    )
    print(reddit_df.head())
except Exception as e:
    print("❌ Reddit test failed:", e)

print("\n📰 Testing News connector...")
try:
    news_df = news_utils.fetch_all()       # <-- changed line
    news_df = news_utils.canonicalize(news_df)
    print(news_df.head())
except Exception as e:
    print("❌ News test failed:", e)

print("\n✅ Test complete.")
