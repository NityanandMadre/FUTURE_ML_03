import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# Create folders if not exist
os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv("data/customer_support.csv")

# Separate customer and support
customer = df[df["inbound"] == True]
support = df[df["inbound"] == False]

# Merge customer question with support answer
customer["response_tweet_id"] = customer["response_tweet_id"].astype(str)
support["tweet_id"] = support["tweet_id"].astype(str)
qa_df = customer.merge(
    support,
    left_on="response_tweet_id",
    right_on="tweet_id",
    suffixes=("_cust", "_supp")
)

qa_df = qa_df[["text_cust", "text_supp"]]
qa_df.columns = ["question", "answer"]

# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text.strip()

qa_df["clean_question"] = qa_df["question"].apply(clean_text)

# Simple intent labeling
def assign_intent(text):
    if "refund" in text:
        return "refund_issue"
    elif "account" in text or "login" in text:
        return "account_issue"
    elif "delay" in text or "delivery" in text:
        return "delivery_issue"
    else:
        return "general_query"

qa_df["intent"] = qa_df["clean_question"].apply(assign_intent)

# Train model
X = qa_df["clean_question"]
y = qa_df["intent"]

vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

# Save model and vectorizer
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("model/intent_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save QA data
qa_df.to_csv("data/qa_data.csv", index=False)

print("✅ Data processed and model saved successfully")
