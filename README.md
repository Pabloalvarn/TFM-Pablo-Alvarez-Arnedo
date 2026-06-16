# Predicción y análisis comparativo del mercado residencial en la Unión Europea

Trabajo Fin de Máster desarrollado en el Máster Universitario en Análisis y Visualización de Datos Masivos.

## Objetivo

Desarrollar un análisis comparativo de distintos modelos predictivos aplicados a la evolución del Índice de Precios de Vivienda (HPI) en países de la Unión Europea, utilizando datos oficiales procedentes de Eurostat y del Banco Central Europeo.

## Estructura del proyecto

- `data/raw`: datos originales descargados desde fuentes oficiales.
- `data/processed`: datasets procesados y preparados para modelización.
- `notebooks`: notebooks secuenciales del pipeline experimental.
- `src/models`: implementación de modelos predictivos.
- `src/evaluation`: métricas y validación temporal.
- `results/figures`: figuras generadas.
- `results/tables`: tablas de resultados.
- `results/models`: modelos entrenados serializados.

## Tecnologías utilizadas

- Python
- pandas
- NumPy
- scikit-learn
- statsmodels
- XGBoost
- matplotlib
- Plotly
- Jupyter Notebook

## Reproducibilidad

El pipeline completo puede ejecutarse secuencialmente mediante los notebooks incluidos en el directorio `notebooks`.

Las dependencias necesarias se recogen en el archivo `requirements.txt`.