from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ---------------------------
# Define Request Schema
# ---------------------------
class TweetRequest(BaseModel):
    text: str

# ---------------------------
# Initialize FastAPI
# ---------------------------
app = FastAPI(
    title="Kenya Hate Speech Detection API",
    description="Classifies tweets into Hate Speech, Offensive, or Neutral using a fine-tuned RoBERTa Transformer",
    version="1.0"
)

# ---------------------------
# Load Model & Tokenizer
# ---------------------------
model_path = "roberta_model"  # Change to your model directory
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

id2label = {0: "Hate Speech", 1: "Neutral", 2: "Offensive"}

# ---------------------------
# Prediction Function
# ---------------------------
def predict_tweet(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
    return {
        "label": id2label[pred_id],
        "confidence": round(probs[0][pred_id].item() * 100, 2)
    }

# ---------------------------
# Root Endpoint
# ---------------------------
@app.get("/")
def home():
    return {"message": "Welcome to the Kenya Hate Speech Detection API"}

# ---------------------------
# Prediction Endpoint
# ---------------------------
@app.post("/predict")
def classify_tweet(request: TweetRequest):
    result = predict_tweet(request.text)
    return result



       