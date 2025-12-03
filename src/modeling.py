import pandas as pd
import yaml
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_params():
    """Lectura desde params.yaml."""
    with open("params.yaml", 'r') as file:
        return yaml.safe_load(file)

def main():
    """Ejecuta la etapa de vectorización y modelado LDA."""
    params = load_params()
    
    INPUT_PATH = 'data/processed/comentarios_limpios.csv'
    VECTORIZER_PATH = 'models/tfidf_vectorizer.pkl'
    MODEL_PATH = 'models/lda_model.pkl'
    
    # Parámetros del modelo (cargados desde params.yaml)
    N_TOPICS = params.get('n_topics', 15)
    MIN_DF = params.get('min_df', 5)
    
    logging.info(f"Cargando datos limpios desde: {INPUT_PATH}")
    try:
        df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        logging.error(f"Archivo no encontrado en {INPUT_PATH}. Ejecuta 'dvc repro' primero.")
        sys.exit(1)

    # Rellenar cualquier valor nulo restante en la columna de texto con una cadena vacía.
    # Esto evita que TfidfVectorizer reciba un np.nan.
    df['texto_limpio'] = df['texto_limpio'].fillna('')
    
    # 1. Vectorización (TF-IDF)
    logging.info(f"Iniciando vectorización con TF-IDF. Mínima frecuencia en documento (min_df): {MIN_DF}")
    
    # TFIDF convierte el texto en valores numéricos ponderados.
    vectorizer = TfidfVectorizer(
        max_df=0.95,       # Ignora términos que aparecen en más del 95% de los documentos
        min_df=MIN_DF,     # Ignora términos que aparecen en menos de MIN_DF documentos (útil para ruido)
        stop_words=None,   # Ya se eliminaron en la fase anterior
        ngram_range=(1, 2) # Considera palabras solas y pares de palabras (bigramas)
    )
    
    # La matriz DTM es la representación numérica del texto
    dtm = vectorizer.fit_transform(df['texto_limpio'])
    
    logging.info(f"DTM creado: {dtm.shape[0]} documentos (comentarios), {dtm.shape[1]} términos (palabras/bigramas)")
    
    # 2. Entrenamiento del Modelo LDA
    logging.info(f"Entrenando modelo LDA con {N_TOPICS} tópicos...")
    
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=42,
        learning_method='batch', # Método de entrenamiento
        max_iter=10,             # Número de iteraciones
        n_jobs=-1                # Usa todos los núcleos de CPU disponibles
    )
    
    # Entrenar el modelo
    lda.fit(dtm)
    
    # 3. Guardar Artefactos
    
    # El vectorizador es crucial para transformar nuevos comentarios en el futuro
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    logging.info(f"Vectorizer guardado en: {VECTORIZER_PATH}")
    
    # El modelo LDA es el que tiene la lógica de los tópicos
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(lda, f)
    logging.info(f"Modelo LDA guardado en: {MODEL_PATH}")

    # Opcional: Asignar el tópico principal a cada comentario para el siguiente paso
    topic_assignment = lda.transform(dtm).argmax(axis=1)
    df['topico_principal'] = topic_assignment
    
    # Guardamos el dataset con el tópico asignado para la Fase 4
    df.to_csv('data/processed/comentarios_con_topico.csv', index=False)
    logging.info("Datos con tópico asignado guardados para la siguiente fase.")

if __name__ == "__main__":
    main()