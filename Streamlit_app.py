import streamlit as st
import pandas as pd

# Import your prediction function
from backend.detect_url import predict_url

# Set Streamlit page settings
st.set_page_config(page_title="PhishCatcher 2.0", layout="centered")
st.title("🔐 PhishCatcher 2.0 – Real-Time URL Phishing Detector")

# Input URL from user
url_input = st.text_input("Enter a URL to check:", placeholder="e.g., http://amaz0n-login.com")

# Button to trigger prediction
if st.button("🔍 Check URL"):
    if url_input:
        with st.spinner("Analyzing..."):
            result = predict_url(url_input)

        # Handle errors from backend
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            # Show URL
            st.success(f"🔗 URL: {result['url']}")

            # 🧠 ML Results
            st.subheader("🧠 Machine Learning Prediction")
            ml_pred = result['ml_model']['prediction']
            ml_conf = result['ml_model']['confidence']
            st.write(f"Prediction: **{ml_pred}**")

            # ML Confidence Chart
            ml_df = pd.DataFrame(list(ml_conf.items()), columns=["Class", "Confidence"])
            st.bar_chart(ml_df.set_index("Class"))

            # 🤖 DL Results
            st.subheader("🤖 Deep Learning Prediction")
            dl_pred = result['dl_model']['prediction']
            dl_conf = result['dl_model']['confidence']
            st.write(f"Prediction: **{dl_pred}**")

            # DL Confidence Chart
            dl_df = pd.DataFrame(list(dl_conf.items()), columns=["Class", "Confidence"])
            st.bar_chart(dl_df.set_index("Class"))
    else:
        st.warning("Please enter a valid URL.")
