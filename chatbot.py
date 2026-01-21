import pickle
import pandas as pd
import re
import random

# Load model and data
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))
model = pickle.load(open("model/intent_model.pkl", "rb"))
qa_df = pd.read_csv("data/qa_data.csv")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    return text.strip()

def predict_intent(text):
    vec = vectorizer.transform([clean_text(text)])
    return model.predict(vec)[0]

def clean_response(text):
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+", "", text)
    return text.strip()

def get_response(intent):
    responses = qa_df[qa_df["intent"] == intent]["answer"]
    response = random.choice(responses.tolist())
    return clean_response(response)


def chatbot_reply(user_input):
    intent = predict_intent(user_input)
    return get_response(intent)
