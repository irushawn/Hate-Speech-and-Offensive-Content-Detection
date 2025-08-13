# Hate Speech and Offensive Language Detection in Kenyan Tweets

## Overview
This project focuses on building a machine learning solution to automatically detect and classify tweets from the Kenyan digital space into:
- **Hate Speech**
- **Offensive Content**
- **Neutral Content**

The model is designed to support early detection of harmful tweets during elections or politically charged events, aiding moderators, journalists, and civil society in promoting peace and constructive online discourse.

---

## Objectives
- Identify common terms/phrases used in inciting tweets.
- Build a **multi-class classification** model to label tweets as hate speech, offensive, or neutral.
- Compare traditional ML models (Logistic Regression, Naïve Bayes, Random Forest, XGBoost) with transformer-based deep learning models (RoBERTa).
- Evaluate model performance using **F1-score**, accuracy, and ROC-AUC.
- Deploy the best-performing model via **FastAPI** backend and **Streamlit** frontend.

---

## Dataset
**Source:** [Kaggle – HateSpeech_Kenya.csv]  
**Size:** ~48,000 tweets labeled for hate speech and offensive content.  

**Features:**
- `tweet` – Raw tweet text (primary input).
- `label` – Target variable:  
  - `0` → Neutral  
  - `1` → Offensive  
  - `2` → Hate Speech  

**Challenges in Data:**
- Presence of URLs, emojis, hashtags, and mentions.
- Code-switching between English, Swahili, and Kenyan slang.
- Class imbalance (more neutral tweets).

---

## Data Preprocessing
- Lowercasing text  
- Removing URLs, mentions, hashtags, emojis, punctuation  
- Tokenization + Lemmatization  
- Stopword removal (including Kenyan-specific filler words)  
- Feature representation with **TF-IDF** and **word embeddings**  
- Handling class imbalance via **undersampling**  

---

## Modeling
The following models were implemented and evaluated:
- Logistic Regression
- Naïve Bayes
- Random Forest
- XGBoost
- **RoBERTa Transformer Model** (fine-tuned for sequence classification)

---

## Model Evaluation
Metrics used:
- **F1-score** (primary)
- Accuracy
- ROC-AUC
- Confusion Matrix

The **RoBERTa** model achieved the best overall balance across precision, recall, and F1-score, making it the chosen model for deployment.

---

## Deployment Plan
The system is deployed with:
- **FastAPI** – Backend API for inference.
- **Streamlit** – User interface for interactive classification.
- Hosting options: Streamlit Cloud / Render / Hugging Face Spaces.

**User Flow:**
1. User enters tweet text into Streamlit app.
2. App sends request to FastAPI endpoint.
3. Model processes input and returns classification result.
4. Streamlit displays predicted category.




