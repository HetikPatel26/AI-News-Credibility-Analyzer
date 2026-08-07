---
title: AI News Credibility Analyzer
emoji: 📰
colorFrom: orange
colorTo: gray
sdk: gradio
sdk_version: 6.22.0
app_file: gradio_app/app.py
pinned: false
license: mit
---

# 📰 AI News Credibility Analyzer

An AI-powered application that predicts whether a news article is **Real** or **Fake** using Machine Learning and verifies the article using trusted online news sources.

---

## Features

- Predicts Fake or Real News
- Supports News URL and News Text
- Automatically extracts article from URLs
- Article statistics (Words, Characters, Sentences)
- Confidence Score
- Online verification using GNews API
- Final credibility verdict
- AI explanation
- Downloadable prediction report
- Multiple prediction models

---

## Machine Learning Models

| Model | Accuracy |
|------|---------:|
| Logistic Regression | 89.25% |
| Random Forest | 62.12% |
| DistilBERT | 99.95% |

Users can switch between models directly from the application.

---

## Tech Stack

- Python
- Gradio
- Scikit-learn
- Transformers (DistilBERT)
- PyTorch
- Newspaper3k
- GNews API
- Pandas
- NumPy

---

## Installation

```bash
git clone https://github.com/HetikPatel26/AI-News-Credibility-Analyzer.git
cd AI-News-Credibility-Analyzer
pip install -r requirements.txt
python gradio_app/app.py
```

---

## License

MIT License

---

## Author

**Hetik Patel**

Diploma in Computer Engineering
AI/ML Project