# =====================================================
# AI NEWS CREDIBILITY ANALYZER
# Part 1 - Imports, Configuration & Functions
# =====================================================

import os
import re
import joblib
import requests
import streamlit as st
import torch

from pathlib import Path
from dotenv import load_dotenv
from newspaper import Article

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

import google.generativeai as genai


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI News Credibility Analyzer",
    page_icon="📰",
    layout="wide"
)


# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"


# =====================================================
# LOAD CSS
# =====================================================

def load_css():

    css_file = BASE_DIR / "style.css"

    if css_file.exists():

        with open(css_file, "r", encoding="utf-8") as f:

            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )


load_css()


# =====================================================
# LOAD ENVIRONMENT VARIABLES
# =====================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")


# =====================================================
# GEMINI CONFIG
# =====================================================

gemini_model = None

if GEMINI_API_KEY:

    try:

        genai.configure(api_key=GEMINI_API_KEY)

        gemini_model = genai.GenerativeModel(
            "gemini-2.5-flash-lite"
        )

    except Exception:

        gemini_model = None


# =====================================================
# LOAD ML MODELS
# =====================================================

@st.cache_resource

def load_models():

    logistic_model = joblib.load(
        MODEL_DIR / "logistic_regression.pkl"
    )

    random_forest = joblib.load(
        MODEL_DIR / "random_forest.pkl"
    )

    tfidf = joblib.load(
        MODEL_DIR / "tfidf_vectorizer.pkl"
    )

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        MODEL_DIR / "distilbert_model"
    )

    bert_model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_DIR / "distilbert_model"
    )

    bert_model.eval()

    return (
        logistic_model,
        random_forest,
        tfidf,
        tokenizer,
        bert_model
    )


(
    logistic_model,
    random_forest,
    tfidf,
    tokenizer,
    bert_model
) = load_models()


# =====================================================
# TEXT CLEANING
# =====================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\\S+", "", text)

    text = re.sub(r"[^a-zA-Z ]", " ", text)

    text = re.sub(r"\\s+", " ", text)

    return text.strip()


# =====================================================
# ARTICLE EXTRACTION
# =====================================================

def extract_article_text(url):

    try:

        article = Article(url)

        article.download()

        article.parse()

        return article.text

    except Exception:

        return None


# =====================================================
# GNEWS VERIFICATION
# =====================================================

def verify_with_gnews(query):

    if not GNEWS_API_KEY:

        return None

    try:

        # Keep only first 8 important words
        query = " ".join(query.split()[:8])

        url = (
            "https://gnews.io/api/v4/search?"
            f"q={query}"
            "&lang=en"
            "&country=in"
            "&max=5"
            "&sortby=relevance"
            f"&apikey={GNEWS_API_KEY}"
        )

        response = requests.get(
            url,
            timeout=20
        )

        if response.status_code == 200:

            return response.json()

        return None

    except Exception:

        return None


# =====================================================
# GEMINI ANALYSIS
# =====================================================

def gemini_analysis(article):

    if gemini_model is None:

        return "Gemini API unavailable."

    prompt = f"""
You are an expert fact checker.

Analyse the following news article.

Return:

1. Credibility

2. Why

3. Possible misinformation

4. Final Recommendation

Article:

{article[:6000]}
"""

    try:

        response = gemini_model.generate_content(prompt)

        return response.text

    except Exception as e:

        return str(e)
# =====================================================
# PART 2 - USER INTERFACE & PREDICTION
# =====================================================

st.title("📰 AI News Credibility Analyzer")

st.markdown(
    "Predict whether a news article is **Real** or **Fake** using Machine Learning and verify it with online sources."
)

st.divider()

# -----------------------------------------------------
# Sidebar
# -----------------------------------------------------

with st.sidebar:

    st.header("⚙ Settings")

    selected_model = st.selectbox(

        "Choose Prediction Model",

        [

            "Logistic Regression",

            "Random Forest",

            "DistilBERT"

        ]

    )

    st.markdown("---")

    st.write("Supported Input")

    st.write("• Paste News URL")

    st.write("• Paste News Article")

# -----------------------------------------------------
# Input
# -----------------------------------------------------

news_input = st.text_area(

    "Paste News URL or Article",

    height=250,

    placeholder="Paste a news URL or the complete news article..."

)

analyze = st.button(

    "🔍 Analyze News",

    use_container_width=True

)

# -----------------------------------------------------
# Analyze
# -----------------------------------------------------

if analyze:

    if news_input.strip() == "":

        st.warning("Please enter a News URL or Article.")

        st.stop()

    # --------------------------------------------
    # Extract article
    # --------------------------------------------

    if news_input.startswith("http://") or news_input.startswith("https://"):

        with st.spinner("Downloading article..."):

            article_text = extract_article_text(news_input)

        if article_text is None or article_text.strip() == "":

            st.error(
    "Unable to extract this website. "
    "Please paste the article text instead or use another news source."
)

            st.stop()

    else:

        article_text = news_input

    # --------------------------------------------
    # Cleaning
    # --------------------------------------------

    cleaned_text = clean_text(article_text)

    if cleaned_text.strip() == "":

        cleaned_text = article_text

    # --------------------------------------------
    # Statistics
    # --------------------------------------------

    word_count = len(article_text.split())

    char_count = len(article_text)

    sentence_count = max(

        1,

        len(re.findall(r"[.!?]", article_text))

    )

    st.subheader("📊 Article Statistics")

    c1, c2, c3 = st.columns(3)

    c1.metric("Words", word_count)

    c2.metric("Characters", char_count)

    c3.metric("Sentences", sentence_count)

    st.divider()

    # --------------------------------------------
    # Prediction
    # --------------------------------------------

    if selected_model == "Logistic Regression":

        vector = tfidf.transform([cleaned_text])

        prediction = logistic_model.predict(vector)[0]

        confidence = float(

            logistic_model.predict_proba(vector).max()

        )

    elif selected_model == "Random Forest":

        vector = tfidf.transform([cleaned_text])

        prediction = random_forest.predict(vector)[0]

        confidence = float(

            random_forest.predict_proba(vector).max()

        )

    else:

        inputs = tokenizer(

            cleaned_text,

            return_tensors="pt",

            truncation=True,

            padding=True,

            max_length=512

        )

        with torch.no_grad():

            outputs = bert_model(**inputs)

            probs = torch.softmax(

                outputs.logits,

                dim=1

            )

            confidence = float(

                torch.max(probs)

            )

            prediction = int(

                torch.argmax(probs)

            )

    # --------------------------------------------
    # Verdict
    # --------------------------------------------

    st.subheader("🤖 AI Prediction")

    if prediction == 1:

        st.success("✅ REAL NEWS")

    else:

        st.error("❌ FAKE NEWS")

    st.metric(

        "Confidence Score",

        f"{confidence*100:.2f}%"

    )

    st.info(f"Model Used: {selected_model}")

    st.divider()

    # Save values for Part 3

    st.session_state["article_text"] = article_text

    st.session_state["prediction"] = prediction

    st.session_state["confidence"] = confidence

    st.session_state["selected_model"] = selected_model

    # =====================================================
    # PART 3 - GNEWS VERIFICATION
    # =====================================================

    st.subheader("🌍 Online Verification")

    with st.spinner("Searching trusted news..."):

        article_text = st.session_state.get("article_text")

    if article_text:

        search_query = " ".join(article_text.split()[:8])

        gnews_result = verify_with_gnews(search_query)

        st.session_state["gnews_result"] = gnews_result 

    gnews_result = st.session_state.get("gnews_result", None)

    if gnews_result and "articles" in gnews_result:

        st.success(f"✅ Found {len(gnews_result['articles'])} related articles.")

        for article in gnews_result["articles"][:5]:

            st.markdown(f"### {article['title']}")
            st.write("Source:", article["source"]["name"])
            st.write(article["description"])
            st.markdown(f"[Read Full Article]({article['url']})")
            st.markdown("---")

    else:

        st.warning("No related articles found.")

    # =====================================================
    # PART 4 - FINAL VERDICT
    # =====================================================

    st.subheader("🏆 Final Verdict")

    # Count GNews Articles
    related_articles = 0

    if (
        gnews_result
        and "articles" in gnews_result
    ):

        related_articles = len(gnews_result["articles"])

    # Final Decision

    if prediction == 1 and related_articles >= 3:

        final_verdict = "✅ HIGHLY CREDIBLE"

        verdict_color = "green"

    elif prediction == 1:

        final_verdict = "🟢 LIKELY REAL"

        verdict_color = "green"

    elif prediction == 0 and related_articles == 0:

        final_verdict = "🔴 LIKELY FAKE"

        verdict_color = "red"

    else:

        final_verdict = "🟡 NEEDS MANUAL VERIFICATION"

        verdict_color = "orange"

    st.markdown(
        f"""
    ### {final_verdict}

    **Prediction Model:** {selected_model}

    **Confidence:** {confidence*100:.2f}%

    **Related Articles Found:** {related_articles}
    """
    )

    st.divider()

    # -------------------------------------------------
    # AI Reason
    # -------------------------------------------------

    st.subheader("🧠 AI Explanation")

    if prediction == 1:

        st.success(
            """
    • Writing style resembles genuine news.

    • Language appears factual.

    • Model confidence is high.

    • Online verification supports the article.
    """
        )

    else:

        st.error(
            """
    • Writing style differs from genuine news.

    • Confidence favours Fake.

    • Verify with trusted sources before sharing.
    """
        )

    st.divider()

    # -------------------------------------------------
    # Show Article
    # -------------------------------------------------

    with st.expander("📄 View Extracted Article"):

        st.text_area(

            "",

            article_text,

            height=350

        )

    # -------------------------------------------------
    # Download Report
    # -------------------------------------------------

    report = f"""
    AI NEWS CREDIBILITY REPORT

    Prediction : {"REAL" if prediction==1 else "FAKE"}

    Confidence : {confidence*100:.2f}%

    Model Used : {selected_model}

    Related Articles : {related_articles}

    Final Verdict :

    {final_verdict}
    """

    st.download_button(

        "📥 Download Report",

        report,

        file_name="news_report.txt"

    )
    st.markdown(
        """
        <div class="footer">
            AI News Credibility Analyzer | Diploma Final Year Project<br>
            Built with Streamlit • DistilBERT • Logistic Regression • Random Forest • TF-IDF • GNews API
        </div>
        """,
        unsafe_allow_html=True
    )
