import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import yaml
import sys
import logging
from nltk.stem.snowball import SnowballStemmer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# solo la primera vez lo descarga
#try:
#    nltk.data.find('corpora/stopwords')
#except nltk.downloader.DownloadError:
#except nltk.downloader.DownloaderError:
#    nltk.download('stopwords')

SPANISH_STOP_WORDS = set(stopwords.words('spanish'))
SPANISH_STEMMER = SnowballStemmer("spanish")

def load_params():
    """Lectura desde params.yaml."""
    with open("params.yaml", 'r') as file:
        return yaml.safe_load(file)

def clean_text(text: str) -> str:
    """Realiza la limpieza básica del texto."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # eliminar URLs
    text = re.sub(r'http\S+', '', text)
    
    # eliminar puntuación, números y caracteres especiales (excepto espacios)
    text = re.sub(r'[^a-záéíóúüñ\s]', '', text)

    # eliminar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    # eliminar Stop Words
    words = text.split()
    words = [word for word in words if word not in SPANISH_STOP_WORDS]
    #words = [SPANISH_STEMMER.stem(word) for word in words]
    
    return " ".join(words)

def main():
    """Ejecuta el pipeline de preparación de datos."""
    params = load_params()
    
    INPUT_PATH = 'data/raw/data.csv'
    OUTPUT_PATH = 'data/processed/comentarios_limpios.csv'
    TEXT_COLUMN = 'texto'
    TARGET_COLUMN = 'calificacion'
    
    logging.info(f"Cargando datos desde: {INPUT_PATH}")
    try:
        df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        logging.error(f"Archivo no encontrado en {INPUT_PATH}. Asegúrate de que el dataset esté allí.")
        sys.exit(1)

    # selecciono columnas y eliminar nulos en el texto
    df_clean = df.loc[df[TEXT_COLUMN].notna(), [TEXT_COLUMN, TARGET_COLUMN]].copy()
    
    logging.info("Aplicando limpieza de texto...")
    df_clean['texto_limpio'] = df_clean[TEXT_COLUMN].apply(clean_text)
    
    # elimino filas con texto vacio
    df_clean = df_clean[df_clean['texto_limpio'] != ""].reset_index(drop=True)

    # guardar el output
    df_clean.to_csv(OUTPUT_PATH, index=False)
    
    logging.info(f"Proceso de limpieza completado. {len(df_clean)} comentarios procesados.")
    logging.info(f"Dataset limpio guardado en: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()