from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_PATH = "roberta_model"

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Set model to evaluation mode
model.eval()

# Map label IDs to names (update with your labels)
id2label = {0: "Hate Speech", 1: "Neutral", 2: "Offensive"}

def predict_tweet(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
    return {
        "label": id2label[pred_id],
        "confidence": round(probs[0][pred_id].item(), 4)
    }
