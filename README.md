# 📰 AI News Credibility Analyzer

An AI-powered web application that detects whether a news article is **Real** or **Fake** using Machine Learning and Deep Learning models. The system also verifies the prediction using the GNews API and provides a final credibility verdict.

---

# 📌 Project Objective

The objective of this project is to reduce the spread of misinformation by automatically analysing news articles and predicting their credibility using Artificial Intelligence.

---

# ✨ Features

- Predicts whether news is Real or Fake
- Accepts both News URL and News Text
- Extracts article content automatically from supported websites
- Uses three AI models for prediction
- Displays confidence score
- Online verification using GNews API
- Final credibility verdict
- Modern Streamlit user interface
- Error handling for invalid URLs and empty input

---

# 🧠 Models Used

### 1. Logistic Regression
- TF-IDF Vectorization
- Fast prediction
- Good baseline accuracy

### 2. Random Forest
- TF-IDF Vectorization
- Ensemble Machine Learning model
- Robust against overfitting

### 3. DistilBERT
- Transformer-based NLP model
- Understands context and semantics
- Highest prediction accuracy

---

# 📊 Model Performance

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 99.06% |
| Random Forest | 98%+ |
| DistilBERT | 99.89% |

---

# 🌍 Online Verification

After prediction, the application searches trusted news sources using the **GNews API** to verify whether similar articles exist online.

This improves the reliability of the final prediction.

---

# 🛠️ Technology Stack

- Python
- Streamlit
- Scikit-learn
- Transformers (Hugging Face)
- DistilBERT
- Pandas
- NumPy
- Newspaper3k
- Joblib
- GNews API
- HTML & CSS

---

# 📂 Project Structure

```
AI-NEWS-CREDIBILITY/
│
├── data/
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── tfidf_vectorizer.pkl
│   └── distilbert_model/
│
├── streamlit_app/
│   ├── app.py
│   ├── style.css
│   └── .env
│
├── requirements.txt
└── README.md
```

---

# 🚀 Installation

Clone the repository

```bash
git clone <repository-url>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

---

# 📈 Workflow

1. User enters News URL or News Text
2. Article is extracted and cleaned
3. User selects AI model
4. Model predicts Real/Fake
5. Confidence score is generated
6. GNews API verifies the news
7. Final credibility verdict is displayed

---

# 📸 Screenshots

Add screenshots of:

- Home Page
- Real News Prediction
- Fake News Prediction
- Online Verification
- Final Verdict

---

# ✅ Advantages

- Easy to use
- Supports both URL and text input
- Multiple AI models
- High prediction accuracy
- Online news verification
- Fast predictions
- Modern interface

---

# ⚠️ Limitations

- Some websites block article extraction
- GNews Free API has a delay for recent news
- Performance depends on article availability
- Internet connection is required for online verification

---

# 🔮 Future Scope

- Multi-language news detection
- Image and video fake news detection
- Browser extension
- Mobile application
- Real-time social media fact checking
- Explainable AI visualisations

---

# 👨‍💻 Developed By

**Diploma Final Year Project**

AI News Credibility Analyzer

---

# 📄 License

This project is developed for educational purposes.
## Note

The trained models are not included in this repository because the complete model directory is approximately 1 GB.

To run the project, place the following files inside the `models/` folder:

- logistic_regression.pkl
- random_forest.pkl
- tfidf_vectorizer.pkl
- distilbert_model/