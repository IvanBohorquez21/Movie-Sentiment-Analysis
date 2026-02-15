import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords')

download_nltk_data()

# Configuración de la página
st.set_page_config(page_title="Analizador de Sentimientos - IMDb", page_icon="🎬")

# --- CARGAR RECURSOS ---
@st.cache_resource
def load_models():
    modelo = joblib.load('models/modelo_sentimientos.pkl')
    vectorizador = joblib.load('models/vectorizador_tfidf.pkl')
    return modelo, vectorizador

try:
    modelo, tfidf = load_models()
except:
    st.error("No se encontraron los modelos. Asegúrate de que están en la carpeta /models")

# --- FUNCION DE LIMPIEZA (Debe ser igual a la del entrenamiento) ---
stemmer = PorterStemmer()
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# --- INTERFAZ DE USUARIO ---
st.title("🎬 Analizador de Sentimientos con IA")
st.write("Escribe una reseña de una película (en inglés) y la IA determinará si es positiva o negativa.")

user_input = st.text_area("Escribe tu reseña aquí:", placeholder="Ej: This movie was amazing, I loved the actors!")

if st.button("Analizar Sentimiento"):
    if user_input:
        # 1. Limpiar el texto ingresado
        text_cleaned = clean_text(user_input)
        
        # 2. Vectorizar
        text_vectorized = tfidf.transform([text_cleaned])
        
        # 3. Predecir
        prediction = modelo.predict(text_vectorized)[0]
        probabilidad = modelo.predict_proba(text_vectorized)
        
        # 4. Mostrar resultado
        st.divider()
        if prediction == 1:
            st.success(f"### Resultado: POSITIVO 😊")
            st.write(f"Confianza: {probabilidad[0][1]*100:.2f}%")
        else:
            st.error(f"### Resultado: NEGATIVO 😡")
            st.write(f"Confianza: {probabilidad[0][0]*100:.2f}%")
    else:
        st.warning("Por favor, escribe algo antes de analizar.")

st.sidebar.info("Este modelo fue entrenado con un dataset de 50,000 reseñas de IMDb y alcanzó una precisión del 88.63%.")