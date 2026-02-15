import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- NLTK DATA DOWNLOAD ---
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')

download_nltk_data()

# Page Configuration
st.set_page_config(page_title="IMDb Sentiment Analyzer", page_icon="🎬")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # Make sure these paths match your repository structure
    modelo = joblib.load('models/modelo_sentimientos.pkl')
    vectorizador = joblib.load('models/vectorizador_tfidf.pkl')
    return modelo, vectorizador

try:
    modelo, tfidf = load_models()
except:
    st.error("Models not found. Please ensure they are located in the /models folder.")

# --- CLEANING FUNCTION ---
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# --- USER INTERFACE ---
st.title("🎬 AI Sentiment Analyzer")
st.write("Enter a movie review below, and the AI will determine if the sentiment is Positive or Negative.")

user_input = st.text_area("Write your review here:", placeholder="e.g.: This movie was amazing, I loved the actors!")

if st.button("Analyze Sentiment"):
    if user_input:
        # 1. Clean input text
        text_cleaned = clean_text(user_input)
        
        # 2. Vectorize
        text_vectorized = tfidf.transform([text_cleaned])
        
        # 3. Predict
        prediction = modelo.predict(text_vectorized)[0]
        probabilidad = modelo.predict_proba(text_vectorized)
        
        # 4. Show Results
        st.divider()
        if prediction == 1:
            st.success(f"### Result: POSITIVE 😊")
            st.write(f"**Confidence:** {probabilidad[0][1]*100:.2f}%")
        else:
            st.error(f"### Result: NEGATIVE 😡")
            st.write(f"**Confidence:** {probabilidad[0][0]*100:.2f}%")
    else:
        st.warning("Please enter some text before analyzing.")

# Sidebar Info
st.sidebar.info(
    "This model was trained using an IMDb dataset of 50,000 reviews "
    "and achieved an accuracy of 88.63%."
)
st.sidebar.markdown("Developed by **Ivan Bohorquez**")