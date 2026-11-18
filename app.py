"""Aplicación académica: ACP y K-Means sobre indicadores de ETF (2021-2025).
Ejecutar: streamlit run app.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="ETF | ACP y K-Means", page_icon="📈", layout="wide")
FEATURES = ["Rentabilidad", "Volatilidad", "Max_Drawdown", "Volumen_Dolares"]
LABELS = {"Rentabilidad": "Rentabilidad anualizada", "Volatilidad": "Volatilidad anualizada", "Max_Drawdown": "Máxima caída", "Volumen_Dolares": "Volumen promedio (USD)"}
DATA_PATH = Path(__file__).resolve().parent / "Datos" / "base_datos_etf.csv"

st.markdown("""<style>
.block-container {padding-top: 1.5rem;}
h1,h2,h3 {letter-spacing: -.02em;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    required = ["ETF", "Categoria"] + FEATURES
    missing = [x for x in required if x not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas necesarias: {', '.join(missing)}")
    return df

st.title("📈 Segmentación de ETF con ACP y K-Means")
st.caption("Machine Learning aplicado a Economía y Finanzas · Proyecto académico · Periodo 2021–2025")
if not DATA_PATH.exists():
    st.error("No se encontró `Datos/base_datos_etf.csv`. Revisa el nombre de la carpeta (D mayúscula) y sube el CSV al repositorio.")
    st.stop()
try:
    original = load_data(DATA_PATH)
except Exception as exc:
    st.error(f"No se pudo leer la base: {exc}")
    st.stop()

st.sidebar.header("Configuración")
section = st.sidebar.radio("Ir a", ["Inicio", "Exploración", "ACP", "K-Means", "Resultados y conclusiones"])
use_log = st.sidebar.checkbox("Aplicar log(1 + volumen)", value=False, help="Desactivado por defecto para reproducir las variables del análisis original en R. La transformación puede reducir la influencia de valores extremos.")

raw = original.copy()
raw[FEATURES] = raw[FEATURES].apply(pd.to_numeric, errors="coerce")
initial_count = len(raw)
duplicates = int(raw["ETF"].duplicated().sum())
missing_values = int(raw[FEATURES].isna().sum().sum())
df = raw.drop_duplicates(subset="ETF").dropna(subset=FEATURES).copy()
if use_log:
    if (df["Volumen_Dolares"] < 0).any():
        st.error("El volumen contiene valores negativos y no permite aplicar log(1 + volumen).")
        st.stop()
    df["Volumen_Dolares"] = np.log1p(df["Volumen_Dolares"])
if len(df) < 5:
    st.error("Se necesitan al menos cinco ETF válidos para comparar diferentes valores de K.")
    st.stop()
if any(df[c].std() == 0 for c in FEATURES):
    st.error("Alguna variable es constante. Revisa la base antes de aplicar ACP.")
    st.stop()

X = StandardScaler().fit_transform(df[FEATURES])
pca = PCA().fit(X)
explained = pca.explained_variance_ratio_
accumulated = np.cumsum(explained)
suggested_components = int(np.searchsorted(accumulated, 0.80) + 1)
components = st.sidebar.slider("Componentes para K-Means", 1, len(FEATURES), suggested_components)
scores_all = pca.transform(X)
scores = scores_all[:, :components]
max_k = min(8, len(df) - 1)
k_values = list(range(2, max_k + 1))
metrics = []
for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=30).fit(scores)
    sil = silhouette_score(scores, model.labels_) if len(set(model.labels_)) > 1 else np.nan
    metrics.append({"K": k, "Inercia": model.inertia_, "Silueta": sil})
metrics_df = pd.DataFrame(metrics)
recommended_k = int(metrics_df.loc[metrics_df["Silueta"].idxmax(), "K"])
k = st.sidebar.slider("Número de grupos (K)", 2, max_k, recommended_k)
cluster_model = KMeans(n_clusters=k, random_state=42, n_init=30).fit(scores)
result = original.loc[df.index].copy()
result["Cluster"] = cluster_model.labels_ + 1
result["Grupo"] = result["Cluster"].map(lambda c: f"Grupo {c}")
result["CP1"] = scores_all[:, 0]
result["CP2"] = scores_all[:, 1]
profile = result.groupby("Grupo", as_index=False).agg(
    Cantidad_ETF=("ETF", "count"),
    Rentabilidad_Promedio=("Rentabilidad", "mean"),
    Volatilidad_Promedio=("Volatilidad", "mean"),
    Caida_Maxima_Promedio=("Max_Drawdown", "mean"),
    Volumen_Promedio_Dolares=("Volumen_Dolares", "mean"),
)

if section == "Inicio":
    st.subheader("¿Qué buscamos responder?")
    st.write("¿Es posible identificar grupos de activos financieros con comportamientos similares según su rentabilidad, riesgo y actividad de negociación?")
    a,b,c,d = st.columns(4)
    a.metric("ETF válidos", len(df))
    b.metric("Variables", len(FEATURES))
    c.metric("Componentes usados", components)
    d.metric("Grupos seleccionados", k)
    st.markdown("""### ¿Cómo funciona?
    **1.** Se leen indicadores históricos de ETF. **2.** Se estandarizan las cuatro variables para hacerlas comparables. **3.** ACP resume la información en componentes. **4.** K-Means agrupa ETF similares según sus posiciones en los componentes seleccionados. **5.** Se interpretan los grupos usando las variables financieras originales.

    **Fuente:** indicadores calculados a partir de datos históricos de Yahoo Finance para 2021–2025. Los valores de la aplicación se leen del CSV del repositorio; no se descargan precios en tiempo real.
    """)
    st.info("Cada ETF es una observación. Aunque se utilizaron precios diarios para calcular los indicadores, el modelo de segmentación trabaja con 20 filas, no con miles de días.")

elif section == "Exploración":
    st.subheader("Exploración y calidad de datos")
    a,b,c = st.columns(3)
    a.metric("Registros originales", initial_count)
    b.metric("ETF duplicados", duplicates)
    c.metric("Valores faltantes (variables)", missing_values)
    st.dataframe(original, use_container_width=True, hide_index=True)
    st.markdown("### Estadísticas descriptivas")
    st.dataframe(original[FEATURES].describe().T.rename(index=LABELS).round(4), use_container_width=True)
    st.markdown("### Rentabilidad frente a volatilidad")
    fig = px.scatter(original, x="Volatilidad", y="Rentabilidad", color="Categoria", hover_name="ETF", text="ETF", labels={"Volatilidad":"Volatilidad anualizada", "Rentabilidad":"Rentabilidad anualizada"})
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Correlaciones")
    corr = original[FEATURES].corr()
    fig = px.imshow(corr, x=[LABELS[x] for x in FEATURES], y=[LABELS[x] for x in FEATURES], zmin=-1, zmax=1, color_continuous_scale="RdBu_r", text_auto=".2f", aspect="auto")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("El identificador ETF y la categoría no entran al modelo. Los datos faltantes y los duplicados se excluyen antes de estandarizar.")

elif section == "ACP":
    st.subheader("Análisis de Componentes Principales (ACP)")
    st.write("Primero estandarizamos las cuatro variables. Después calculamos componentes que resumen las diferencias entre los ETF.")
    variance = pd.DataFrame({"Componente": [f"CP{i+1}" for i in range(len(explained))], "Varianza explicada (%)": explained*100, "Acumulada (%)": accumulated*100})
    a,b = st.columns(2)
    with a:
        fig = px.bar(variance, x="Componente", y="Varianza explicada (%)", text="Varianza explicada (%)", title="Gráfico de sedimentación")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.line(variance, x="Componente", y="Acumulada (%)", markers=True, title="Varianza acumulada")
        fig.add_hline(y=80, line_dash="dash", annotation_text="Referencia: 80 %")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(variance.round(2), use_container_width=True, hide_index=True)
    st.success(f"Con {components} componente(s) se conserva el {accumulated[components-1]*100:.2f} % de la variabilidad. El umbral del 80 % es una referencia, no una regla obligatoria.")
    st.markdown("### Cargas de las variables")
    loadings = pd.DataFrame(pca.components_.T, index=[LABELS[x] for x in FEATURES], columns=[f"CP{i+1}" for i in range(len(FEATURES))])
    st.dataframe(loadings.round(3), use_container_width=True)
    st.caption("Las cargas muestran cómo contribuye cada variable a los componentes. El signo de un componente puede invertirse sin cambiar la solución.")
    st.markdown("### Mapa de ETF en los dos primeros componentes")
    fig = px.scatter(result, x="CP1", y="CP2", color="Categoria", text="ETF", hover_name="ETF", title=f"CP1 + CP2 explican {(explained[0]+explained[1])*100:.2f} % de la variación")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Este plano muestra solo los dos primeros componentes, aunque K-Means puede usar más componentes según la selección lateral.")

elif section == "K-Means":
    st.subheader("Segmentación mediante K-Means")
    st.write("Comparamos varios valores de K. El método del codo muestra cómo disminuye la dispersión dentro de los grupos; la silueta mide qué tan separados quedan.")
    a,b = st.columns(2)
    with a:
        st.plotly_chart(px.line(metrics_df, x="K", y="Inercia", markers=True, title="Método del codo"), use_container_width=True)
    with b:
        st.plotly_chart(px.line(metrics_df, x="K", y="Silueta", markers=True, title="Coeficiente de silueta"), use_container_width=True)
    st.dataframe(metrics_df.round(3), use_container_width=True, hide_index=True)
    st.info(f"Para {components} componentes, la silueta máxima entre K=2 y K={max_k} se alcanza en K={recommended_k}. Puedes cambiar K en la barra lateral y contrastar los resultados.")
    fig = px.scatter(result, x="CP1", y="CP2", color="Grupo", text="ETF", hover_name="ETF", title=f"Segmentación con K={k} (visualización en CP1 y CP2)")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Los colores representan grupos, no niveles ordenados de calidad o riesgo. El gráfico es una proyección de dos dimensiones y puede ocultar diferencias presentes en otros componentes.")
    st.markdown("### ETF por grupo")
    st.dataframe(result[["ETF", "Categoria", "Grupo"]].sort_values(["Grupo", "ETF"]), use_container_width=True, hide_index=True)

else:
    st.subheader("Resultados e interpretación financiera")
    st.write("Para interpretar K-Means, volvemos a las variables financieras originales. Los números de grupo son etiquetas arbitrarias: no representan una clasificación de mejor a peor.")
    st.dataframe(profile.round(4), use_container_width=True, hide_index=True)
    st.markdown("### Comparación de los perfiles")
    variable = st.selectbox("Indicador", ["Rentabilidad_Promedio", "Volatilidad_Promedio", "Caida_Maxima_Promedio", "Volumen_Promedio_Dolares"], format_func=lambda x: x.replace("_", " "))
    st.plotly_chart(px.bar(profile, x="Grupo", y=variable, text_auto=".3g", title="Promedio por grupo"), use_container_width=True)
    for group, part in result.groupby("Grupo"):
        st.write(f"**{group} ({len(part)} ETF):** {', '.join(part['ETF'].tolist())}.")
    st.markdown("""### Conclusiones y limitaciones
    - El ACP permite resumir la información de cuatro indicadores financieros y observar patrones entre ETF.
    - K-Means propone grupos de comportamiento parecido, pero su utilidad debe evaluarse con la silueta y con diferencias financieras interpretables.
    - El volumen negociado puede tener valores muy extremos. Activa la transformación logarítmica en la barra lateral para estudiar si los grupos cambian.
    - La muestra es pequeña (20 ETF) y algunos fondos comparten exposiciones de mercado. Los grupos dependen de las variables, el periodo y las decisiones metodológicas.
    - Estos grupos son exploratorios; no constituyen una recomendación de inversión ni predicen rendimientos futuros.
    """)
    st.download_button("Descargar ETF con grupos (CSV)", result.drop(columns=["CP1", "CP2"]).to_csv(index=False).encode("utf-8-sig"), "etf_clusters_streamlit.csv", "text/csv")
    st.download_button("Descargar perfiles (CSV)", profile.to_csv(index=False).encode("utf-8-sig"), "perfil_clusters_streamlit.csv", "text/csv")

st.sidebar.divider()
st.sidebar.caption("Fuente: CSV de indicadores históricos · Semilla K-Means: 42 · 30 inicializaciones")
st.sidebar.caption("Para comparar con R, deja desactivada la transformación logarítmica y usa la misma cantidad de componentes y K.")
