import re
import torch
import pandas as pd
from transformers import AutoTokenizer, RobertaForSequenceClassification

# Load model and tokenizer
model_name = "roberta_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)
model.eval()

# String labels directly
label_map = {0: "Hate Speech", 1: "Neutral", 2: "Offensive"}

def clean_tweet(text):
    """Clean tweet text before prediction."""
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)     # remove mentions
    text = re.sub(r"#", "", text)        # remove hashtag symbols
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)  # remove punctuation
    text = text.lower().strip()          # lowercase
    return text

def predict_single(tweet):
    """Predict class for a single tweet."""
    cleaned = clean_tweet(tweet)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=1).item()
    return label_map[pred_id]

def predict_bulk(tweets):
    """Predict classes for multiple tweets."""
    cleaned_tweets = [clean_tweet(t) for t in tweets]
    inputs = tokenizer(cleaned_tweets, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_ids = torch.argmax(outputs.logits, dim=1).tolist()

    pred_labels = [label_map[p] for p in pred_ids]
    df = pd.DataFrame({"tweet": tweets, "prediction": pred_labels})

    total = len(pred_labels)
    percentages = {label: round(pred_labels.count(label) / total * 100, 2) for label in set(pred_labels)}
    most_common = max(percentages, key=percentages.get)

    return df, percentages, most_common





