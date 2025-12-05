import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os
from nlp_prep import clean_text


MODEL_PATH = 'models/lda_model.pkl'
VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        lda_model = pickle.load(f)
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
    print("✅ Modelos LDA y Vectorizador cargados exitosamente.")
except FileNotFoundError:
    print(f"❌ Error: Artefactos no encontrados en {os.getcwd()}. Asegúrate de ejecutar 'dvc pull'.")
    sys.exit(1)

TOPIC_NAMES = {
    10: "moto / buenas / eficiente / motos / solo  ",
    11: "moto / servicio amabilidad / rapido / brindaron / amabilidad",
    2: "rapidez / tiempo / repuestos / espera / disponibilidad",
    3: "exelente / exelente servicio / servicio / honda / mal",
    4: "amable / personal / atendido / atención / bien",
    5: "calidad / si / responder / marca / buena calidad",
    #
}

# --- Inicialización de FastAPI ---
app = FastAPI(title="VoC Topic Prioritization API")

# --- Modelos Pydantic para la Documentación (Swagger) ---
class TextIn(BaseModel):
    comment: str

class PredictionOut(BaseModel):
    topic_id: int
    topic_name: str
    is_hot: bool

# --- ENDPOINT ---
@app.post("/predict_topic", response_model=PredictionOut)
def predict_topic(data: TextIn):
    """
    Recibe un comentario de texto libre y asigna el tópico principal.
    """
    
    # 1. Limpieza de Texto (misma función usada en el entrenamiento)
    cleaned_text = clean_text(data.comment)
    
    # 2. Vectorización (usando el vectorizador entrenado)
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # 3. Predicción (obtener la distribución de probabilidades por tópico)
    topic_distribution = lda_model.transform(vectorized_text)[0]
    
    # 4. Asignación del Tópico Principal (el que tenga mayor probabilidad)
    topic_id = int(topic_distribution.argmax())
    
    # 5. Lógica de Riesgo (simplificada)
    # Por ejemplo, cualquier tópico con ID 10, 11 o 2 es de alto riesgo
    is_hot = topic_id in [10, 11, 2] 
    
    # 6. Devolver Resultado
    return {
        "topic_id": topic_id,
        "topic_name": TOPIC_NAMES.get(topic_id, f"Tópico {topic_id} - No Nombrado"),
        "is_hot": is_hot
    }

# Para ejecutar la API localmente (opcional)
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)