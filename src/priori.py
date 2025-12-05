import pandas as pd
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Calcula las métricas de priorización (Volumen, Satisfacción Media y Prioridad)."""
    
    INPUT_PATH = 'data/processed/comentarios_con_topico.csv'
    OUTPUT_PATH = 'data/processed/topicos_priorizados.csv'
    
    logging.info(f"Cargando datos con tópicos asignados desde: {INPUT_PATH}")
    try:
        df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        logging.error(f"Archivo no encontrado en {INPUT_PATH}. Verifica que la Fase 3 haya corrido correctamente.")
        sys.exit(1)

    # validar que la columna sea numérica
    df['topico_principal'] = pd.to_numeric(df['topico_principal'], errors='coerce')
    
    logging.info("Calculando métricas de agregación por tópico...")
    prioridad_df = df.groupby('topico_principal').agg(
        volumen=('topico_principal', 'count'),  # conteo de comentarios
        satisfaccion_media=('calificacion', 'mean') # media de calificación
    ).reset_index()

    # Cálculo de la Métrica de Riesgo/Prioridad
    # normalizo la puntuación media
    max_satisfaccion = df['calificacion'].max()
    prioridad_df['insatisfaccion_normalizada'] = (max_satisfaccion - prioridad_df['satisfaccion_media']) / max_satisfaccion
    
    # para calcular la prioridad = Insatisfacción * Volumen.
    # para penalizar los temas donde mucha gente está insatisfecha.
    prioridad_df['prioridad_score'] = prioridad_df['insatisfaccion_normalizada'] * prioridad_df['volumen']
    
    prioridad_df = prioridad_df.sort_values(by='prioridad_score', ascending=False)
    
    prioridad_df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"Priorización completada. Resultados guardados en: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()