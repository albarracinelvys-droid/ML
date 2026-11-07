from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from tetouan.data import TARGET, features, read
from tetouan.inference import predict

st.set_page_config(page_title="Tetouan | Demanda eléctrica", layout="wide")
st.title("Demanda eléctrica de Tetouan")
st.caption("Zona 1 · horizonte de 60 minutos · unidades originales del dataset")
source = st.sidebar.radio("Datos", ["Dataset local", "Cargar CSV"])
upload = (
    st.sidebar.file_uploader("Historial cada 10 minutos", type="csv")
    if source == "Cargar CSV"
    else None
)
path = upload if source == "Cargar CSV" else Path("data/raw/tetouan.csv")
if path is None:
    st.info("Carga un CSV para comenzar.")
    st.stop()
try:
    frame = read(path)
except (ValueError, FileNotFoundError) as exc:
    st.error(str(exc))
    st.stop()
summary, patterns, climate, forecast, evaluation, interpretation = st.tabs(
    [
        "Datos y calidad",
        "Patrones temporales",
        "Clima y relaciones",
        "Predicción",
        "Evaluación",
        "Interpretación",
    ]
)
with summary:
    a, b, c = st.columns(3)
    a.metric("Registros", f"{len(frame):,}")
    b.metric("Variables predictoras", len(features(frame).columns))
    c.metric("Valores faltantes", int(frame.isna().sum().sum()))
    st.write("Periodo", str(frame.index.min()), "a", str(frame.index.max()))
    st.dataframe(frame.describe().T, width="stretch")
    st.caption(
        "Datos sin imputación: la carga rechaza faltantes, duplicados temporales y saltos de frecuencia."
    )
    st.download_button(
        "Descargar estadísticas", frame.describe().T.to_csv(), "estadisticas.csv"
    )
with patterns:
    zone = st.selectbox("Zona", [c for c in frame if "Power Consumption" in c])
    frequency = st.selectbox(
        "Agregación",
        ["D", "h", "W"],
        format_func=lambda x: {"D": "Diaria", "h": "Horaria", "W": "Semanal"}[x],
    )
    st.line_chart(frame[zone].resample(frequency).mean())
    left, right = st.columns(2)
    left.subheader("Perfil por hora")
    left.line_chart(frame.groupby(frame.index.hour)[zone].mean())
    right.subheader("Perfil por día de semana (0=lunes)")
    right.bar_chart(frame.groupby(frame.index.day_of_week)[zone].mean())
with climate:
    variable = st.selectbox("Variable meteorológica", list(frame.columns[:5]))
    st.scatter_chart(frame.iloc[::20], x=variable, y=TARGET)
    st.caption(
        "Muestra sistemática de una de cada 20 observaciones. Asociación no implica causalidad."
    )
    st.subheader("Correlaciones entre variables observadas")
    st.dataframe(frame.corr().style.background_gradient(cmap="RdBu", vmin=-1, vmax=1))
    st.subheader("Distribución")
    counts, edges = np.histogram(frame[variable], bins=30)
    st.bar_chart(pd.Series(counts, index=np.round((edges[1:] + edges[:-1]) / 2, 3)))
with forecast:
    try:
        bundle = joblib.load("artifacts/model.joblib")
        available = features(frame)
        time = st.selectbox(
            "Instante de referencia", available.index, index=len(available) - 1
        )
        current = available.loc[time]
        if list(available.columns) != bundle["columns"]:
            st.warning("Modelo anterior: reentrena para usar las nuevas variables.")
        else:
            outside = (current < bundle["training_min"]) | (
                current > bundle["training_max"]
            )
            if outside.any():
                st.warning(
                    "Fuera del rango de entrenamiento: "
                    + ", ".join(current.index[outside])
                )
            value = predict(bundle, frame.loc[:time])
            future = time + pd.Timedelta(minutes=60)
            st.metric(f"Predicción para {future}", f"{value:,.2f}")
            st.write("Modelo seleccionado por validación:", bundle["winner"])
            if future in frame.index:
                observed = float(frame.loc[future, TARGET])
                st.metric(
                    "Observado a +60 minutos",
                    f"{observed:,.2f}",
                    f"Error absoluto: {abs(value - observed):,.2f}",
                )
                st.caption(
                    "Consulta histórica; no constituye una nueva evaluación fuera de muestra."
                )
            st.dataframe(current.rename("Valor disponible al predecir"))
    except (ValueError, FileNotFoundError) as exc:
        st.error(str(exc))
with evaluation:
    st.caption(
        "Resultados del experimento guardado, independientes del CSV cargado. Test no se utiliza para seleccionar modelos."
    )
    for name, title in [
        ("validation", "Validación — selección"),
        ("test", "Test — evaluación final"),
    ]:
        report = Path(f"artifacts/{name}.csv")
        if report.exists():
            table = pd.read_csv(report)
            st.subheader(title)
            st.dataframe(table, width="stretch")
            st.bar_chart(table.set_index("model")["rmse"])
    report = Path("artifacts/predictions.csv")
    if report.exists():
        errors = pd.read_csv(report, index_col=0, parse_dates=True)
        st.line_chart(errors[["observed", "predicted"]].iloc[:1008])
        errors["absolute_error"] = errors.error.abs()
        st.subheader("MAE por hora")
        st.bar_chart(errors.groupby(errors.index.hour).absolute_error.mean())
        st.subheader("20 errores más grandes")
        st.dataframe(errors.nlargest(20, "absolute_error"))
with interpretation:
    report = Path("artifacts/importance.csv")
    if report.exists():
        importance = pd.read_csv(report, index_col=0)
        importance.columns = ["Aumento de RMSE al permutar"]
        st.bar_chart(importance.tail(20))
    st.markdown(
        "Las variables incluyen consumo histórico, dinámica diaria, meteorología reciente, relaciones entre zonas y calendario. Las importancias se calculan en validación; variables correlacionadas pueden compartir importancia."
    )
    st.warning(
        "Más variables no garantizan más precisión. Los resultados actuales requieren nuevas ventanas temporales; el test de 2017 ya fue consultado en versiones anteriores."
    )
