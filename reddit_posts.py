import praw
import pandas as pd
from datetime import datetime

# Initialize Reddit API (replace with your credentials)
reddit = praw.Reddit(
    client_id="RERi5cGDZAjueYXweR_rlQ",
    client_secret="mdRVlbiD3ClK-5e-3QZVaxxtZQ4CxA",
    user_agent="my_reddit_scraper/0.1 by arsenic101_"
)

# Define subreddit and keyword
subreddit = reddit.subreddit("CryptoCurrency")
keyword = "XRP"

# Fetch Reddit posts
posts = []
for post in subreddit.search(keyword, limit=3000:)
    posts.append([
        post.title,  
        post.selftext,  
        post.score,  
        post.created_utc  # Add timestamp
    ])

# Convert to DataFrame
df_reddit = pd.DataFrame(posts, columns=["title", "content", "score", "timestamp"])

# Convert Unix timestamp to date
df_reddit["date"] = pd.to_datetime(df_reddit["timestamp"], unit="s").dt.date

# Save CSV
df_reddit.to_csv("reddit_posts_raw.csv", index=False)
print("Reddit sentiment data saved.")
print(len(df_reddit))