from fastapi import FastAPI, UploadFile, File
import pandas as pd
from model import predict_single, predict_bulk

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hate Speech & Offensive Content Detection API Running"}

@app.post("/predict_single/")
def predict_single_tweet(tweet: str):
    prediction = predict_single(tweet)
    recommendation = f"This tweet is classified as {prediction}. Suggested action: {'Flag for review' if prediction != 'Neutral' else 'No action needed'}."
    return {"tweet": tweet, "prediction": prediction, "recommendation": recommendation}

@app.post("/predict_bulk/")
def predict_bulk_tweets(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    if "tweet" not in df.columns:
        return {"error": "CSV file must have a 'tweet' column."}

    results_df, percentages, most_common = predict_bulk(df["tweet"].tolist())
    recommendation = f"The most common classification is {most_common}. Suggested action: {'Investigate flagged tweets' if most_common != 'Neutral' else 'Content seems safe overall'}."

    # Return records as list of dicts for Streamlit
    return {
        "total_tweets": len(results_df),
        "class_percentages": percentages,
        "recommendation": recommendation,
        "records": results_df.to_dict(orient="records")
    }








       