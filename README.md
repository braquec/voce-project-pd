# Priorización Voz de cliente e Identificación de Temas Críticos

Analiza y prioriza la voz del cliente mediante insumo de encuestas NPS realizadas a cada cliente

## Objetivo del proyecto
Principalmente responder la pregunta del negocio **¿Qué problema debemos resolver para maximizar la satisfacción del cliente?**

- **Agrupación**: Utiliza el modelo LDA para agrupar miles de comentarios en tópicos coherentes (ej: "Problemas de Repuestos", "Calidad de Atención").
- **Priorización**: Combina el Volumen de comentarios con la **Satisfacción Media** (Puntuación General) para generar un Score de Prioridad ($\text{Insatisfacción} \times \text{Volumen}$).
- **Servicio**: Ofrece un servicio de inferencia en tiempo real utilizando **FastAPI** y **Docker**.

## Descripción de solución
Transforma los comentarios de cliente mediante encuestas realizadas durante la adquisición de un bien o servicio (motociletas, taller o repuestos) utilizando el modelo LDA y MLops con DVC, empaquetado mediante Docker y disponibilizado mediante FastAPI

## Arquitectura MLOps
El proyecto utiliza Data Version Control (DVC) para gestionar las dependencias del pipeline y garantizar la reproducibilidad de todos los resultados (datos, modelos y métricas).

**Flujo del Procesamiento**
- `preparar`: la data: Limpieza de texto, eliminación de stop words, stemming y filtrado.
- `vectorizar`: Vectorización (TF-IDF) y entrenamiento del modelo LDA.
- `priorizar`: Cálculo del Score de Prioridad y generación de la tabla topicos_priorizados.csv.

## Estructura de proyecto
project  
├── api  
│ ├─ main.py # clase principal para la api  
│ ├─ nlp_prep.py # preparación de datos  
|── data  
│ ├── processed  # carpeta para data procesada  
│ └── raw # carpeta para data cruda  
├── models # carpeta donde se guardan los modelos
│ ├── lda_model.pkl # modelo lda  
│ └── tfidf_vectorizer.pkl # vectorizador  
├── notebooks # Notebooks para pruebas y visualizaciones  
│ ├── analisis.ipynb # notebook para análisis y presentación  
│ └── test.ipynb # notebook para pruebas  
├── src # codigo fuente  
├── Dockerfile # archivo para contenedor  
├── dvc.yaml # archivo para stages de DVC  
├── params.yaml # archivo para manejo de hiperparámetros del modelo  
├── requirements.txt #  
└── README.md  

## Reproducibilidad

- Clonacion  
`git clone https://github.com/braquec/voce-project-pd.git`  
`cd voce-project-pd`

- Dependencias  
`pip install -r requirements.txt`  

- Descargar modelo de versionado DVC
`dvc pull`

- Ejecutar el Pipeline
`dvc repro`

- Construir imagen Docker
`docker build -t voc-project-api`

- Levantar el servicio API
`docker run -d --name voc-project-service -p 8000:8000 voc-project-api`

## Probar inferencia
**URL dela API**: `http://localhost:8000/docs`

Utilizar el endpoint `POST /predict_topic` para parobar cualquier comentario de texto libre y obtener la asignacion del topico en tiempo real.


## Autor (es)

*[Berner Raquec](https://github.com/braquec)* - <25002780@galileo.edu>  
*[Oscar Vera](https://github.com/braquec)* - <25002780@galileo.edu>