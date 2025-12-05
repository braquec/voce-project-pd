# Dockerfile

# Usa una imagen base de Python ligera (Python 3.10)
FROM python:3.10-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos de configuración
COPY requirements.txt .

# Instala las dependencias
RUN pip install --no-cache-dir -r requirements.txt


RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

COPY models/lda_model.pkl models/
COPY models/tfidf_vectorizer.pkl models/

# Copiamos el contenido de la carpeta API a la raíz
COPY api/main.py .
COPY api/nlp_prep.py .


# Expone el puerto que usará la API
EXPOSE 8000

# Comando para iniciar el servidor Uvicorn (FastAPI)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]