import streamlit as st
import joblib
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="TweetSense AI",
    page_icon="assets/favicon.jpeg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# LOAD MODEL
# =========================

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# =========================
# NLTK
# =========================

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# =========================
# PREPROCESS FUNCTION
# =========================

def preprocess(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}

.block-container{
    padding-top:1rem;
}

.hero-title{
    text-align:center;
    font-size:50px;
    font-weight:800;
    color:#1f77ff;
}

.hero-subtitle{
    text-align:center;
    color:gray;
    font-size:20px;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
}

textarea{
    border-radius:12px !important;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.image("assets/logo.jpeg", width=150)

    st.title("TweetSense AI")

    st.markdown("---")

    st.markdown("""
### About

TweetSense AI is an NLP-powered sentiment analysis platform that classifies tweets into:

✅ Positive

❌ Negative

😐 Neutral

### Tech Stack

- Python
- NLTK
- TF-IDF
- Logistic Regression
- Streamlit
""")

# =========================
# BANNER
# =========================

col1 , col2 , col3 =st.columns([1,8,1])
with col2:
    st.image("assets/banner.jpeg",width=1200)

# =========================
# TITLE
# =========================

st.markdown("""
<div class="hero-title">
TweetSense AI
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-subtitle">
Sentiment Analysis using NLP & Machine Learning
</div>
""", unsafe_allow_html=True)

st.write("")

# =========================
# INPUT BOX
# =========================

col1, col2 = st.columns([5, 3])

with col1:

    tweet = st.text_area(
        "Enter a Tweet",
        height=100,
        placeholder="Example: I love how smooth this app runs after the update!"
    )

with col2:

    st.subheader("Sample Tweets")
    
    with st.expander("View Sample tweets"):
        st.code("I absolutely love this new update!")

        st.code("This is the worst service I have ever used.")

        st.code("Feeling happy and blessed today!")

        st.code("Customer support solved my issue quickly.")

        st.code("Battery life is terrible after the update.")

        st.code("Hello,how are you?")

        st.code("It's raining today")

# =========================
# ANALYSIS
# =========================

if st.button("Analyze Sentiment"):

    if tweet.strip() == "":
        st.warning("Please enter a tweet.")

    else:

        processed = preprocess(tweet)

        vector = vectorizer.transform([processed])

        prediction = model.predict(vector)[0]

        try:
            confidence = max(
                model.predict_proba(vector)[0]
            )
            confidence_text = f"{confidence:.2%}"

        except:
            confidence_text = "N/A"

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction == "Positive":

            st.success(
                f"😊 Positive Sentiment\n\nConfidence: {confidence_text}"
            )

        elif prediction == "Negative":

            st.error(
                f"😡 Negative Sentiment\n\nConfidence: {confidence_text}"
            )

        else:

            st.info(
                f"😐 Neutral Sentiment\n\nConfidence: {confidence_text}"
            )

# =========================
# FOOTER
# =========================

st.markdown("---")

st.markdown(
"""
<div style='text-align:center;color:gray;'>

TweetSense AI © 2026

Built with Streamlit, NLTK & Machine Learning

</div>
""",
unsafe_allow_html=True
)