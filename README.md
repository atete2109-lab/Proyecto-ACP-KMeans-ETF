# Proyecto-ACP-KMeans-ETF
Segmentación de activos financieros mediante ACP y K-Means
# Segmentación de activos financieros mediante ACP y K-Means

## 1. Descripción del proyecto

Este proyecto busca analizar y segmentar diferentes fondos cotizados en bolsa (ETF) del mercado financiero estadounidense, teniendo en cuenta características relacionadas con su rentabilidad, riesgo y actividad de negociación.

La idea es identificar si existen grupos de ETF que presenten comportamientos similares y reconocer qué características diferencian a cada grupo.

Para desarrollar el análisis se utilizan dos técnicas de aprendizaje no supervisado: el Análisis de Componentes Principales (ACP) y el algoritmo K-Means.

## 2. Pregunta de investigación

¿Es posible identificar grupos de activos financieros con comportamientos similares según sus características de rentabilidad, riesgo y actividad del mercado?

## 3. Objetivo general

Identificar y analizar grupos de ETF con características financieras similares mediante la aplicación conjunta de ACP y K-Means.

## 4. Datos utilizados

La información se obtiene de Yahoo Finance y corresponde a precios y volúmenes históricos de negociación de diferentes ETF del mercado estadounidense.

El periodo de estudio propuesto comprende desde enero de 2021 hasta diciembre de 2025.

Se consideran cuatro variables cuantitativas:

- **Rentabilidad anualizada:** aproximación al rendimiento promedio anual del activo.
- **Volatilidad anualizada:** medida de la variabilidad de sus rendimientos.
- **Máxima caída (Max Drawdown):** mayor caída desde un máximo previo durante el periodo analizado.
- **Volumen promedio negociado en dólares:** aproximación al valor promedio de negociación del activo.

La selección definitiva de activos y la cantidad de observaciones se documentarán junto con la base final.

## 5. Metodología

El proyecto se desarrolla en las siguientes etapas:

1. **Exploración de datos:** revisión de la estructura de la información, estadísticas descriptivas, valores faltantes y posibles inconsistencias.
2. **Preparación de datos:** selección de variables, tratamiento de datos y estandarización.
3. **Análisis de Componentes Principales (ACP):** reducción de dimensiones, análisis de varianza explicada e interpretación de las cargas.
4. **Segmentación con K-Means:** comparación de diferentes cantidades de grupos mediante el método del codo y el coeficiente de silueta.
5. **Interpretación:** descripción de los grupos a partir de las variables financieras originales.

## 6. Herramientas utilizadas

- Python
- R
- R Markdown
- Streamlit
- Pandas y NumPy
- Scikit-learn
- Yahoo Finance
- GitHub

## 7. Estructura del proyecto

La organización del repositorio se completará durante el desarrollo:

- `README.md`: documentación general del proyecto.
- `app.py`: aplicación desarrollada en Streamlit.
- `requirements.txt`: dependencias necesarias para ejecutar la aplicación.
- `analisis/`: documentos del análisis en R Markdown y HTML.
- `datos/`: base de datos utilizada.
- `resultados/`: tablas y resultados generados.

## 8. Ejecución del proyecto

Las instrucciones para instalar las dependencias, obtener los datos y ejecutar la aplicación se completarán cuando estén listos los archivos definitivos.

## 9. Resultados y conclusiones

Los resultados del ACP y la segmentación K-Means se incorporarán después de ejecutar el análisis. Se presentarán las principales dimensiones identificadas, la cantidad de grupos seleccionada y las características financieras de cada segmento.

## 10. Autor

Proyecto académico individual desarrollado para la asignatura Machine Learning aplicado a Economía y Finanzas.

Pontificia Universidad Javeriana Cali.

