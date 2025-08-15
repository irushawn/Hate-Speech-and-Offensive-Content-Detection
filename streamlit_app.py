import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import time
import re

# =========================
# Config
# =========================
st.set_page_config(page_title="Hate Speech Detection", layout="wide", page_icon="🛡️")

API_URL_BASE   = "http://127.0.0.1:8000"
API_URL_SINGLE = f"{API_URL_BASE}/predict_single/"
API_URL_BULK   = f"{API_URL_BASE}/predict_bulk/"

# =========================
# Sidebar
# =========================
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Choose Mode", ("Single Tweet", "Bulk Tweets", "Live Feed (Bulk)"))

st.sidebar.markdown("---")
st.sidebar.caption("Backend")
st.sidebar.caption("Check if the prediction API is reachable here.")
if st.sidebar.button("Ping API"):
    try:
        r = requests.get(API_URL_BASE + "/")
        st.sidebar.success(r.json().get("message", "API OK"))
    except Exception as e:
        st.sidebar.error(f"API not reachable: {e}")

# =========================
# Helpers
# =========================
TOXIC_LEXICON = set([
    "kill","hate","trash","idiot","stupid","kikuyu","luo","kalenjin","gikuyu",
    "terrorist","die","useless","disgusting","filthy","pig","dog","monkey"
])

def clean_text_min(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def render_prediction_card(label: str):
    if label == "Hate Speech":
        st.markdown('<div style="padding:1.25rem;border-radius:18px;margin-top:.5rem;text-align:center;font-size:1.15rem;font-weight:600;background:rgba(255,82,82,.12);border:1px solid rgba(255,82,82,.35);">🚫 Hate Speech Detected</div>', unsafe_allow_html=True)
    elif label == "Offensive":
        st.markdown('<div style="padding:1.25rem;border-radius:18px;margin-top:.5rem;text-align:center;font-size:1.15rem;font-weight:600;background:rgba(255,159,28,.12);border:1px solid rgba(255,159,28,.35);">⚠️ Offensive Content Detected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="padding:1.25rem;border-radius:18px;margin-top:.5rem;text-align:center;font-size:1.15rem;font-weight:600;background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.35);">✅ Neutral Content</div>', unsafe_allow_html=True)

def render_recommendation_card(rec: str):
    st.markdown(
        f'<div style="padding:1rem;border-radius:16px;margin-top:.5rem;text-align:center;font-size:1.05rem;font-weight:500;background:linear-gradient(90deg, #a0c4ff, #bdb2ff);color:#1f1f1f;box-shadow:0 4px 6px rgba(0,0,0,0.1);">{rec}</div>',
        unsafe_allow_html=True
    )

def highlight_tokens(text: str):
    tokens = text.split()
    out = []
    for tok in tokens:
        base = re.sub(r"[^A-Za-z0-9]", "", tok).lower()
        if base in TOXIC_LEXICON:
            out.append(f'<span style="background: rgba(255,82,82,.25); padding:.06rem .22rem; border-radius:.3rem;">{tok}</span>')
        else:
            out.append(tok)
    return " ".join(out)

def plot_distribution(breakdown: dict):
    if not breakdown:
        return None
    df = pd.DataFrame({
        "Category": list(breakdown.keys()),
        "Percentage": list(breakdown.values())
    })
    fig = px.bar(
        df, x="Category", y="Percentage", color="Category", text="Percentage",
        color_discrete_map={"Hate Speech": "red", "Neutral": "green", "Offensive": "orange"},
        title="Class Distribution"
    )
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    fig.update_layout(yaxis_title="Percentage (%)", xaxis_title="", height=380, margin=dict(l=10, r=10, t=50, b=10))
    return fig

# =========================
# Header
# =========================
st.title("🛡️ Hate Speech & Offensive Content Detection")
st.caption("Detect whether a tweet is harmful, offensive, or neutral. Recommendations are provided to help improve content.")

# =========================
# Single Tweet
# =========================
if mode == "Single Tweet":
    with st.container():
        st.subheader("🔍 Single Tweet")
        st.caption("Enter a tweet below to see if it contains harmful or offensive language.")
        tweet = st.text_area("Enter tweet text:", height=140, placeholder="Type or paste a tweet…")

        cols = st.columns([1,1,3])
        with cols[0]:
            run = st.button("Predict")
        with cols[1]:
            st.caption("Optional: text is lightly cleaned before sending to the model.")

        if run:
            if not tweet.strip():
                st.warning("Please enter a tweet.")
            else:
                try:
                    resp = requests.post(API_URL_SINGLE, params={"tweet": tweet})
                    resp.raise_for_status()
                    data = resp.json()
                    label = data.get("prediction", "Unknown")
                    rec   = data.get("recommendation", "")

                    render_prediction_card(label)
                    if rec:
                        render_recommendation_card(rec)

                    st.markdown("**Explanation (keyword highlights)**")
                    st.caption("Highlighted words may indicate potentially harmful or offensive content.")
                    st.markdown(
                        f'<div style="padding:.6rem 1rem;border-radius:12px;border:1px dashed rgba(0,0,0,.15);margin-top:.2rem;">{highlight_tokens(clean_text_min(tweet))}</div>',
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Request failed: {e}")

# =========================
# Bulk Tweets (Instant)
# =========================
elif mode == "Bulk Tweets":
    with st.container():
        st.subheader("📊 Bulk Prediction")
        st.caption("Upload a CSV with tweets to analyze them in bulk. A sample CSV can help format your file correctly.")

        sample_df = pd.DataFrame({"tweet": ["I love this!", "I hate you!"]})
        st.download_button("📄 Download Sample CSV", sample_df.to_csv(index=False).encode("utf-8"),
                           "sample_tweets.csv", "text/csv")

        file = st.file_uploader("Upload CSV with a 'tweet' column", type=["csv"])

        if file and st.button("Run Predictions"):
            try:
                files = {"file": (file.name, file.getvalue(), "text/csv")}
                resp = requests.post(API_URL_BULK, files=files)
                resp.raise_for_status()
                payload = resp.json()

                if "error" in payload:
                    st.error(payload["error"])
                else:
                    total = payload.get("total_tweets", 0)
                    breakdown = payload.get("class_percentages", {})
                    recommendation = payload.get("recommendation", "")
                    records = payload.get("records", [])

                    st.markdown(f"**Total Tweets:** {total}")
                    if recommendation:
                        render_recommendation_card(recommendation)
                        st.caption("Overall recommendation for the batch of tweets.")

                    fig = plot_distribution(breakdown)
                    if fig: st.plotly_chart(fig, use_container_width=True)

                    if records:
                        df = pd.DataFrame(records)
                        with st.expander("📄 View Predictions Table", expanded=True):
                            st.caption("Detailed results for each tweet, including predicted label.")
                            st.dataframe(df, use_container_width=True)

                        st.download_button(
                            label="📥 Download Predictions CSV",
                            data=df.to_csv(index=False).encode("utf-8"),
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Request failed: {e}")

# =========================
# Live Feed Simulation (Bulk)
# =========================
else:
    with st.container():
        st.subheader("🎞️ Live Feed (Bulk Simulation)")
        st.caption("Simulate a live stream of tweets being analyzed. Use the slider to control speed.")

        file = st.file_uploader("Upload CSV with a 'tweet' column", type=["csv"])
        speed = st.slider("Playback speed (tweets/sec)", min_value=1, max_value=10, value=3)

        if file and st.button("Start Live Feed"):
            try:
                files = {"file": (file.name, file.getvalue(), "text/csv")}
                resp = requests.post(API_URL_BULK, files=files)
                resp.raise_for_status()
                payload = resp.json()

                if "error" in payload:
                    st.error(payload["error"])
                else:
                    records = payload.get("records", [])
                    if not records:
                        st.warning("No records returned.")
                    else:
                        container = st.empty()
                        df_stream = pd.DataFrame(columns=["tweet","prediction"])

                        for i, rec in enumerate(records, 1):
                            df_stream.loc[len(df_stream)] = [rec["tweet"], rec["prediction"]]
                            with container.container():
                                render_prediction_card(rec["prediction"])
                                st.caption(f"Tweet {i}/{len(records)}")
                                st.markdown(
                                    f'<div style="padding:.6rem 1rem;border-radius:12px;border:1px dashed rgba(0,0,0,.15);margin-top:.2rem;">{highlight_tokens(clean_text_min(rec["tweet"]))}</div>',
                                    unsafe_allow_html=True
                                )
                                st.markdown("Recent stream (last 10)")
                                st.caption("Preview the last 10 analyzed tweets in real time.")
                                st.dataframe(df_stream.tail(10), use_container_width=True)
                            time.sleep(1.0/float(speed))

                        st.success("Live feed complete.")
                        st.download_button(
                            label="📥 Download Stream Results CSV",
                            data=df_stream.to_csv(index=False).encode("utf-8"),
                            file_name="stream_results.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Request failed: {e}")
