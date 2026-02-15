# 🎬 Movie Sentiment Analysis AI

This project uses **Natural Language Processing (NLP)** to classify IMDb movie reviews as either **positive** or **negative**. The model was trained on a dataset of 50,000 records, achieving an optimal balance between processing speed and accuracy.

## 🚀 Features

* **Model Accuracy:** 88.63%
* **Algorithm:** Logistic Regression.
* **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency) with 5,000 features.
* **Interface:** Interactive web application built with Streamlit.

---

## 🛠️ Technologies Used

* **Language:** Python 3.12.4
* **Libraries:** * `Scikit-learn`: For modeling and vectorization.
    * `NLTK`: For text cleaning and stopword removal.
    * `Pandas`: For data manipulation.
    * `Streamlit`: For web interface deployment.
    * `Joblib`: For model and vectorizer serialization.

---

## 🧪 Data Pipeline

To achieve the current accuracy, a rigorous preprocessing pipeline was implemented:

1.  **HTML Cleaning:** Removal of tags such as `<br />`.
2.  **Normalization:** Lowercase conversion and special character removal.
3.  **Tokenization:** Splitting text into individual units (words).
4.  **Stopwords Removal:** Filtering out common words that do not carry sentimental weight (e.g., "the", "is", "at").
5.  **Stemming:** Reducing words to their root form (e.g., "watching" -> "watch") using `PorterStemmer`.

---

## 💻 Installation and Local Usage

1.  **Clone the repository:**
    ```bash
        git clone https://github.com/IvanBohorquez21/Movie-Sentiment-Analysis.git
        cd Movie-Sentiment-Analysis
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # Mac/Linux:
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

---

## 📊 Results

The **Logistic Regression** model was chosen for its high efficiency in binary text classification tasks.

* **Performance:** The model demonstrates solid performance in detecting both negative (class 0) and positive (class 1) reviews, effectively minimizing false positives.
* **Sarcasm:** The model is excellent at detecting direct sentiment, though it faces interesting challenges with complex sarcasm due to its word-frequency based approach (TF-IDF).

---

**Developed by Ivan Bohorquez** - Electronic Engineering / AI Enthusiast.
