# =====================================================
# AI NEWS CREDIBILITY ANALYZER - GRADIO VERSION
# =====================================================

import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import gradio as gr
import joblib
import requests
import torch
from dotenv import load_dotenv
from newspaper import Article
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)


# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR_CANDIDATES = [
    BASE_DIR / "models",
    BASE_DIR.parent / "models",
]

MODEL_DIR = next(
    (path for path in MODEL_DIR_CANDIDATES if path.exists()),
    MODEL_DIR_CANDIDATES[0],
)


# =====================================================
# ENVIRONMENT VARIABLES
# =====================================================

load_dotenv(BASE_DIR / ".env")
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # reserved for future Gemini explanation feature
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")


# =====================================================
# TUNABLE CONSTANTS
# =====================================================

MIN_RELIABLE_WORDS = 25
MAX_DISPLAY_CHARS = 20_000
GNEWS_MAX_RESULTS = 5

MODEL_CHOICES = ["Logistic Regression", "Random Forest", "DistilBERT"]


# =====================================================
# MODEL LOADING
# =====================================================

@lru_cache(maxsize=1)
def load_models():
    """Load and cache all trained models."""
    required_paths = {
        "Logistic Regression": MODEL_DIR / "logistic_regression.pkl",
        "Random Forest": MODEL_DIR / "random_forest.pkl",
        "TF-IDF Vectorizer": MODEL_DIR / "tfidf_vectorizer.pkl",
        "DistilBERT folder": MODEL_DIR / "distilbert_model",
    }

    missing = [
        f"{name}: {path}"
        for name, path in required_paths.items()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required model files were not found.\n" + "\n".join(missing)
        )

    logistic_model = joblib.load(required_paths["Logistic Regression"])
    random_forest = joblib.load(required_paths["Random Forest"])
    tfidf = joblib.load(required_paths["TF-IDF Vectorizer"])

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        required_paths["DistilBERT folder"],
        local_files_only=True,
    )
    bert_model = DistilBertForSequenceClassification.from_pretrained(
        required_paths["DistilBERT folder"],
        local_files_only=True,
    )
    bert_model.eval()

    return (logistic_model, random_forest, tfidf, tokenizer, bert_model)


# =====================================================
# TEXT PROCESSING
# =====================================================

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_valid_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value.strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def confidence_tier(confidence: float) -> str:
    if confidence >= 0.90:
        return "🟢 High"
    if confidence >= 0.70:
        return "🟡 Moderate"
    return "🟠 Low"


# =====================================================
# ARTICLE EXTRACTION
# =====================================================

def extract_article(url: str) -> dict | None:
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text.strip()
        if not text:
            return None
        return {"text": text, "title": (article.title or "").strip()}
    except Exception:
        return None


# =====================================================
# GNEWS VERIFICATION
# =====================================================

def verify_with_gnews(query: str):
    if not GNEWS_API_KEY or not query.strip():
        return None

    try:
        short_query = " ".join(query.split()[:8])

        response = requests.get(
            "https://gnews.io/api/v4/search",
            params={
                "q": short_query,
                "lang": "en",
                "country": "in",
                "max": GNEWS_MAX_RESULTS,
                "sortby": "relevance",
                "apikey": GNEWS_API_KEY,
            },
            timeout=20,
        )

        if response.status_code == 200:
            return response.json()
        return None
    except requests.RequestException:
        return None


# =====================================================
# DISPLAY HELPERS
# =====================================================

def format_gnews_results(gnews_result) -> tuple[str, int]:
    if not gnews_result or "articles" not in gnews_result:
        api_note = (
            "\n\n*GNews API key is not configured.*" if not GNEWS_API_KEY else ""
        )
        return f"### ⚠️ No related articles found.{api_note}", 0

    articles = gnews_result.get("articles", [])[:GNEWS_MAX_RESULTS]

    if not articles:
        return "### ⚠️ No related articles found.", 0

    sections = [f"### ✅ Found {len(articles)} related article(s)"]

    for index, article in enumerate(articles, start=1):
        title = article.get("title") or "Untitled article"
        source = article.get("source", {}).get("name") or "Unknown source"
        description = article.get("description") or "No description available."
        url = article.get("url") or ""

        item = f"#### {index}. {title}\n*Source:* {source}\n\n{description}"
        if url:
            item += f"\n\n[Read full article]({url})"

        sections.append(item)

    return "\n\n---\n\n".join(sections), len(articles)


def create_report(prediction, confidence, selected_model, related_articles, final_verdict) -> str:
    report = f"""AI NEWS CREDIBILITY REPORT

Prediction       : {"REAL" if prediction == 1 else "FAKE"}
Confidence       : {confidence * 100:.2f}% ({confidence_tier(confidence)})
Model Used       : {selected_model}
Related Articles : {related_articles}

Final Verdict:
{final_verdict}
"""

    report_dir = BASE_DIR / "generated_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        prefix="news_report_",
        dir=report_dir,
        delete=False,
        encoding="utf-8",
    ) as file:
        file.write(report)
        return file.name


# ---- visibility helpers for gr.update(...) shorthand ----
def _show():
    return gr.update(visible=True)


def _hide():
    return gr.update(visible=False)


def empty_results(message: str):
    """Only the stats card is shown, carrying the warning message; every
    other result card / accordion / download stays hidden."""
    return (
        _show(),                                   # results_wrapper
        _show(), f"### ⚠️ {message}",                # stats_group, statistics_output
        _hide(), "",                                # prediction_group, prediction_output
        _hide(), "",                                # online_group, online_output
        _hide(), "",                                # verdict_group, final_output
        _hide(), "",                                # explanation_group, explanation_output
        _hide(), "",                                # article_accordion, article_output
        _hide(), None,                               # download_group, report_output
    )


# =====================================================
# MAIN ANALYSIS FUNCTION
# =====================================================

def analyze_news(news_input: str, selected_model: str, progress=gr.Progress()):
    if not news_input or not news_input.strip():
        return empty_results("Please enter a news URL or complete article.")

    news_input = news_input.strip()
    article_title = ""

    try:
        progress(0.05, desc="Preparing analysis")

        if is_valid_http_url(news_input):
            progress(0.15, desc="Downloading article")
            extracted = extract_article(news_input)

            if not extracted:
                return empty_results(
                    "Unable to extract this website. Paste the "
                    "article text instead or use another news source."
                )

            article_text = extracted["text"]
            article_title = extracted["title"]
        else:
            article_text = news_input

        cleaned_text = clean_text(article_text)
        if not cleaned_text:
            cleaned_text = article_text

        word_count = len(article_text.split())
        char_count = len(article_text)
        sentence_count = max(1, len(re.findall(r"[.!?]", article_text)))

        reliability_note = ""
        if word_count < MIN_RELIABLE_WORDS:
            reliability_note = (
                "\n\n⚠️ *This is quite short — predictions on very short "
                "text tend to be less reliable.*"
            )

        statistics_md = f"""### 📊 Article Statistics

| Words | Characters | Sentences |
|---:|---:|---:|
| {word_count:,} | {char_count:,} | {sentence_count:,} |
{reliability_note}"""

        progress(0.35, desc="Loading prediction models")
        (logistic_model, random_forest, tfidf, tokenizer, bert_model) = load_models()

        progress(0.55, desc=f"Running {selected_model}")

        if selected_model == "Logistic Regression":
            vector = tfidf.transform([cleaned_text])
            prediction = int(logistic_model.predict(vector)[0])
            confidence = float(logistic_model.predict_proba(vector).max())

        elif selected_model == "Random Forest":
            vector = tfidf.transform([cleaned_text])
            prediction = int(random_forest.predict(vector)[0])
            confidence = float(random_forest.predict_proba(vector).max())

        elif selected_model == "DistilBERT":
            inputs = tokenizer(
                cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512
            )
            with torch.no_grad():
                outputs = bert_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                confidence = float(torch.max(probabilities).item())
                prediction = int(torch.argmax(probabilities, dim=1).item())
        else:
            return empty_results(f"Unsupported model selected: {selected_model}")

        tier = confidence_tier(confidence)

        if prediction == 1:
            prediction_md = f"""### 🤖 AI Prediction

## ✅ REAL NEWS

*Confidence Score:* {confidence * 100:.2f}% ({tier})
*Model Used:* {selected_model}
"""
        else:
            prediction_md = f"""### 🤖 AI Prediction

## ❌ FAKE NEWS

*Confidence Score:* {confidence * 100:.2f}% ({tier})
*Model Used:* {selected_model}
"""

        progress(0.72, desc="Searching related news")
        search_query = article_title or " ".join(article_text.split()[:8])
        gnews_result = verify_with_gnews(search_query)
        online_md, related_articles = format_gnews_results(gnews_result)
        online_md = f"### 🌐 Online Verification\n\n{online_md}"

        if prediction == 1 and related_articles >= 3:
            final_verdict = "✅ HIGHLY CREDIBLE"
        elif prediction == 1:
            final_verdict = "🟢 LIKELY REAL"
        elif prediction == 0 and related_articles == 0:
            final_verdict = "🔴 LIKELY FAKE"
        else:
            final_verdict = "🟡 NEEDS MANUAL VERIFICATION"

        final_md = f"""### 🏆 Final Verdict

## {final_verdict}

*Prediction Model:* {selected_model}
*Confidence:* {confidence * 100:.2f}% ({tier})
*Related Articles Found:* {related_articles}
"""

        if prediction == 1:
            explanation_md = """### 🧠 AI Explanation

- The writing style resembles news classified as genuine by the selected model.
- The language pattern appears more factual according to the model.
- The confidence score favours the *Real* class.
- Related-source results should still be reviewed before sharing the article.

> A machine-learning prediction is not proof that every claim in an article is factually correct.
"""
        else:
            explanation_md = """### 🧠 AI Explanation

- The writing style differs from patterns learned from genuine-news samples.
- The confidence score favours the *Fake* class.
- The article may contain misleading, sensational, or unsupported wording.
- Verify important claims using multiple trusted sources before sharing.

> A machine-learning prediction is an indicator, not a final fact-check.
"""

        progress(0.9, desc="Creating report")
        report_path = create_report(
            prediction=prediction,
            confidence=confidence,
            selected_model=selected_model,
            related_articles=related_articles,
            final_verdict=final_verdict,
        )

        progress(1.0, desc="Analysis complete")

        display_text = article_text
        if len(display_text) > MAX_DISPLAY_CHARS:
            display_text = display_text[:MAX_DISPLAY_CHARS] + "\n\n… (truncated for display)"

        return (
            _show(),
            _show(), statistics_md,
            _show(), prediction_md,
            _show(), online_md,
            _show(), final_md,
            _show(), explanation_md,
            _show(), display_text,
            _show(), report_path,
        )

    except FileNotFoundError as exc:
        return empty_results(f"Model files are missing.\n\n{exc}")
    except Exception as exc:
        return empty_results(f"Analysis failed: {type(exc).__name__}: {exc}")


# =====================================================
# GRADIO INTERFACE
# =====================================================

DEFAULT_CSS = """
html,
body,
.gradio-container{
    width:100% !important;
    max-width:100% !important;
    min-width:100% !important;
    margin:0 !important;
    padding:0 !important;
}

.contain{
    max-width:100% !important;
}

.main{
    width:100% !important;
}
"""

css_path = BASE_DIR / "style.css"
custom_css = DEFAULT_CSS
if css_path.exists():
    try:
        custom_css += "\n" + css_path.read_text(encoding="utf-8")
    except OSError:
        pass


SIDEBAR_HTML = """
<div id="sidebar-info">
    <div class="sidebar-heading">Project Information</div>
    <div class="info-badge">Diploma Final Year Project</div>

    <div class="sidebar-subheading">Models Used</div>
    <ul class="check-list">
        <li>Logistic Regression</li>
        <li>Random Forest</li>
        <li>DistilBERT</li>
    </ul>

    <div class="sidebar-divider"></div>

    <div class="sidebar-subheading">Dataset</div>
    <div class="sidebar-text">Fake &amp; Real News Dataset</div>

    <div class="sidebar-divider"></div>

    <div class="sidebar-heading-small">Developed Using</div>
    <ul class="plain-list">
        <li>Gradio</li>
        <li>Scikit-Learn</li>
        <li>Hugging Face</li>
        <li>PyTorch</li>
    </ul>

    <div class="sidebar-divider"></div>

    <div class="sidebar-heading-small">Supported Input</div>
    <ul class="plain-list">
        <li>Paste News URL</li>
        <li>Paste News Article</li>
    </ul>
</div>
"""


def register_analyze_handler(trigger):
    trigger(
        fn=analyze_news,
        inputs=[news_input, selected_model],
        outputs=[
            results_wrapper,
            stats_group, statistics_output,
            prediction_group, prediction_output,
            online_group, online_output,
            verdict_group, final_output,
            explanation_group, explanation_output,
            article_accordion, article_output,
            download_group, report_output,
        ],
        show_progress="full",
    )

with open("gradio_app/style.css", "r", encoding="utf-8") as f:
    custom_css = f.read()
    
with gr.Blocks(title="AI News Credibility Analyzer") as demo:

    sidebar_state = gr.State(False)

    sidebar_toggle = gr.Button(
        "☰",
        elem_id="floating-toggle"
    )

    with gr.Row(elem_id="layout"):

        with gr.Column(
            visible=False,
            elem_id="sidebar",
            scale=0,
            min_width=300
        ) as sidebar:

            gr.HTML(SIDEBAR_HTML)

        with gr.Column(
            scale=1,
            elem_id="content"
        ):

            gr.HTML("""
            <div id="hero">
                <h1>📰 AI News Credibility Analyzer</h1>
                <p>
                Predict whether a news article is
                <strong>Real</strong> or
                <strong>Fake</strong>
                using machine learning and verify it with online sources.
                </p>
            </div>
            """)

            news_input = gr.Textbox(
                label="Paste News URL (recommended) or full news article",
                placeholder="Paste a news URL or the complete news article...",
                lines=10,
                elem_id="news-input",
            )

            selected_model = gr.Radio(
                choices=MODEL_CHOICES,
                value=MODEL_CHOICES[0],
                label="Choose Prediction Model",
                elem_id="model-radio",
            )

            with gr.Row(elem_id="action-row"):
                analyze_button = gr.Button(
                    "🔍 Analyze News",
                    variant="primary",
                    elem_id="analyze-btn"
                )

                clear_button = gr.ClearButton(
                    value="🗑️ Clear",
                    elem_id="clear-btn"
                )

            with gr.Column(
                visible=False,
                elem_id="results-wrapper"
            ) as results_wrapper:

                with gr.Group(
                    elem_id="stats-card",
                    elem_classes="result-card",
                    visible=False
                ) as stats_group:
                    statistics_output = gr.Markdown()

                with gr.Group(
                    elem_id="prediction-card",
                    elem_classes="result-card",
                    visible=False
                ) as prediction_group:
                    prediction_output = gr.Markdown()

                with gr.Group(
                    elem_id="online-card",
                    elem_classes="result-card",
                    visible=False
                ) as online_group:
                    online_output = gr.Markdown()

                with gr.Group(
                    elem_id="verdict-card",
                    elem_classes="result-card",
                    visible=False
                ) as verdict_group:
                    final_output = gr.Markdown()

                with gr.Group(
                    elem_id="explanation-card",
                    elem_classes="result-card",
                    visible=False
                ) as explanation_group:
                    explanation_output = gr.Markdown()

                with gr.Accordion(
                    "📄 View Extracted Article",
                    open=False,
                    visible=False
                ) as article_accordion:

                    article_output = gr.Textbox(
                        label="Extracted Article",
                        lines=14,
                        interactive=False
                    )

                with gr.Group(
                    visible=False
                ) as download_group:

                    report_output = gr.DownloadButton(
                        label="📥 Download Report",
                        elem_id="download-btn"
                    )

    clear_button.add([
        news_input,
        statistics_output,
        prediction_output,
        online_output,
        final_output,
        explanation_output,
        article_output,
        report_output,
    ])

    clear_button.click(
        fn=lambda: (
            _hide(), _hide(), _hide(), _hide(),
            _hide(), _hide(), _hide(), _hide()
        ),
        outputs=[
            results_wrapper,
            stats_group,
            prediction_group,
            online_group,
            verdict_group,
            explanation_group,
            article_accordion,
            download_group,
        ],
    )

    register_analyze_handler(analyze_button.click)
    register_analyze_handler(news_input.submit)

    def toggle_sidebar(opened):
        return (
            gr.update(visible=not opened),
            not opened
        )

    sidebar_toggle.click(
        fn=toggle_sidebar,
        inputs=sidebar_state,
        outputs=[
            sidebar,
            sidebar_state
        ]
    )

    gr.HTML("""
    <div class="footer" id="app-footer">
        AI News Credibility Analyzer | Diploma Final Year Project<br>
        Built with Gradio • DistilBERT • Logistic Regression •
        Random Forest • TF-IDF • GNews API
    </div>
    """)


    if __name__ == "__main__":
        demo.queue().launch(
            css=custom_css,
            server_name="127.0.0.1",
            server_port=7860
        )
import os

# ... all your app code above ...

if __name__ == "__main__":
    is_huggingface = os.getenv("SPACE_ID") is not None

    demo.queue().launch(
        css=custom_css,
        server_name="0.0.0.0" if is_huggingface else "127.0.0.1",
        server_port=7860
    )