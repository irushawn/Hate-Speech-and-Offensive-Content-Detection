import streamlit as st
import requests

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Kenya Hate Speech Detector", page_icon="🇰🇪", layout="centered")

st.title("🇰🇪 Kenya Hate Speech Detection")
st.write("Enter a tweet below and get an instant prediction from the model.")

# Input text box
tweet_text = st.text_area("✏️ Tweet Text", height=150)

# Predict button
if st.button("🔍 Predict"):
    if tweet_text.strip() == "":
        st.warning("Please enter a tweet text to classify.")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Send request to FastAPI
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json={"text": tweet_text}
                )

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Prediction: **{result['label']}**")
                    st.write(f"Confidence: {result['confidence']}%")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
