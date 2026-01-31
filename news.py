import streamlit as st
import joblib

# Load model and vectorizer
GBC = joblib.load("GBC_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page config
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Title
st.title("📰 Fake News Detection System")
st.write("Enter a news article below to check whether it is **Real** or **Fake**.")

# Input text
news_text = st.text_area(
    "Paste the news text here:",
    height=200,
    placeholder="Type or paste news content..."
)

# Predict button
if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:
        # Vectorize input
        transformed_text = vectorizer.transform([news_text])

        # Prediction
        prediction = GBC.predict(transformed_text)[0]

        # Output
        if prediction == 1:
            st.error("🚨 This news is **Fake**")
        else:
            st.success("✅ This news is **Real**")

# Footer
st.markdown("---")
st.markdown("**Developed by Prabhat Yadav**")
