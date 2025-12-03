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

    # Asegurarse de que la columna de tópico sea numérica (por si se guardó como texto)
    df['topico_principal'] = pd.to_numeric(df['topico_principal'], errors='coerce')
    
    # 1. Agregación de Métricas por Tópico
    logging.info("Calculando métricas de agregación por tópico...")
    prioridad_df = df.groupby('topico_principal').agg(
        volumen=('topico_principal', 'count'),  # Conteo de comentarios por tópico
        satisfaccion_media=('calificacion', 'mean') # Puntuación media
    ).reset_index()

    # 2. Cálculo de la Métrica de Riesgo/Prioridad
    
    # Normalizar la puntuación media (asumiendo escala 1-10 o similar) 
    # Si la escala es del 1 al 10, 'insatisfaccion' será (10 - puntuacion)
    # Si la escala es del 0 al 5, puedes normalizar entre 0 y 1 para calcular la insatisfacción.
    # Usaremos una simple inversa: (Valor Máximo - Satisfacción Media). Ajusta el valor 50 si tu escala no es 1-10.
    
    max_satisfaccion = df['calificacion'].max()
    prioridad_df['insatisfaccion_normalizada'] = (max_satisfaccion - prioridad_df['satisfaccion_media']) / max_satisfaccion
    
    # Prioridad = Insatisfacción * Volumen.
    # Esto penaliza fuertemente los temas donde mucha gente está insatisfecha.
    prioridad_df['prioridad_score'] = prioridad_df['insatisfaccion_normalizada'] * prioridad_df['volumen']
    
    # Opcional: Obtener las 5 palabras clave para nombrar el tópico (requiere cargar el vectorizador y el modelo)
    # Para simplicidad en este script, solo guardaremos las métricas.
    
    # 3. Ordenar y Guardar
    prioridad_df = prioridad_df.sort_values(by='prioridad_score', ascending=False)
    
    prioridad_df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"Priorización completada. Resultados guardados en: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()