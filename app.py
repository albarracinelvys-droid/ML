from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from tetouan.data import TARGET, features, read
from tetouan.inference import predict

st.set_page_config(page_title="Tetouan | Demanda eléctrica", layout="wide")
ROOT = Path(__file__).resolve().parent
st.title("¿Cuánta electricidad se necesitará dentro de una hora?")
st.info(
    "Abre Predicción guiada: elige una fecha y compara la estimación para una hora después con lo que ocurrió."
)
st.caption("Zona 1 · horizonte de 60 minutos · unidades originales del dataset")
source = st.sidebar.radio("Datos", ["Dataset local", "Cargar CSV"])
upload = (
    st.sidebar.file_uploader("Historial cada 10 minutos", type="csv")
    if source == "Cargar CSV"
    else None
)
path = upload if source == "Cargar CSV" else ROOT / "data/raw/tetouan.csv"
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
        "1 · Conoce los datos",
        "2 · Descubre patrones",
        "3 · Explora el clima",
        "Predicción guiada",
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
    import altair as alt

    st.subheader("Una predicción, paso a paso")
    st.markdown(
        "**1. Elige el momento de partida.** El modelo usa el consumo y el clima registrados hasta ese instante. **2. Mira la estimación.** Calcula la demanda de Zona 1 exactamente una hora después. **3. Comprueba el resultado.** Si esa hora existe en el archivo, puedes compararla con lo que ocurrió."
    )
    try:
        bundle = joblib.load(ROOT / "artifacts/model.joblib")
        available = features(frame)
        missing = set(bundle["columns"]) - set(available.columns)
        if missing:
            raise ValueError(
                "El modelo requiere variables que esta versión no calcula: "
                + ", ".join(sorted(missing))
            )
        eligible = available.index[
            available.index + pd.Timedelta(hours=1) <= frame.index[-1]
        ]
        options = available.index
        default_time = (
            eligible[int(len(eligible) * 0.9)] if len(eligible) else options[-1]
        )
        mode = st.radio(
            "¿Qué quieres probar?",
            [
                "Ejemplo histórico: puedo comprobar el resultado",
                "Último dato: estimar la siguiente hora",
            ],
        )
        if mode.startswith("Ejemplo"):
            dates = pd.Index(eligible.date).unique()
            day = st.selectbox(
                "Día del ejemplo", dates, index=list(dates).index(default_time.date())
            )
            hours = eligible[eligible.date == day]
            time = st.select_slider(
                "Hora de partida",
                options=list(hours),
                value=hours[len(hours) // 2],
                format_func=lambda t: t.strftime("%H:%M"),
            )
        else:
            time = options[-1]
            st.write("Última observación disponible:", str(time))
        future = time + pd.Timedelta(hours=1)
        current = available.loc[time, bundle["columns"]]
        outside = (current < bundle["training_min"]) | (
            current > bundle["training_max"]
        )
        if outside.any():
            st.warning(
                "Algunas condiciones están fuera del rango usado para entrenar. Interpreta la estimación con más cautela."
            )
        value = predict(bundle, frame.loc[:time])
        now = float(frame.loc[time, TARGET])
        cols = st.columns(3)
        cols[0].metric("Consumo al partir · " + time.strftime("%H:%M"), f"{now:,.0f}")
        cols[1].metric(
            "Estimación · " + future.strftime("%H:%M"),
            f"{value:,.0f}",
            f"{(value / now - 1) * 100:+.1f}% frente al punto de partida"
            if now
            else None,
        )
        observed = float(frame.loc[future, TARGET]) if future in frame.index else None
        if observed is not None:
            cols[2].metric(
                "Lo que ocurrió · " + future.strftime("%H:%M"), f"{observed:,.0f}"
            )
        else:
            cols[2].metric("Resultado observado", "Aún no está en el CSV")
        st.caption(
            "Valores en unidades originales del archivo; no se convierten a kW porque la tabla de UCI no especifica la unidad."
        )
        history = frame.loc[time - pd.Timedelta(hours=6) : time, [TARGET]].reset_index()
        history.columns = ["Hora", "Demanda"]
        line = (
            alt.Chart(history)
            .mark_line(color="#2563eb", strokeWidth=3)
            .encode(
                x=alt.X("Hora:T", title="Hora"),
                y=alt.Y("Demanda:Q", scale=alt.Scale(zero=False)),
                tooltip=["Hora:T", "Demanda:Q"],
            )
        )
        points = [{"Hora": future, "Demanda": value, "Serie": "Predicción a +60 min"}]
        if observed is not None:
            points.append(
                {"Hora": future, "Demanda": observed, "Serie": "Observado a +60 min"}
            )
        dots = (
            alt.Chart(pd.DataFrame(points))
            .mark_point(size=180, filled=True)
            .encode(
                x="Hora:T",
                y="Demanda:Q",
                color=alt.Color(
                    "Serie:N",
                    scale=alt.Scale(
                        domain=["Predicción a +60 min", "Observado a +60 min"],
                        range=["#f97316", "#16a34a"],
                    ),
                ),
                tooltip=["Serie", "Hora:T", "Demanda:Q"],
            )
        )
        st.altair_chart((line + dots).properties(height=340), width="stretch")
        st.caption(
            "Azul: últimas seis horas conocidas. Naranja: una única predicción para una hora después. Verde: valor real, si existe. No se predice una trayectoria entre los dos instantes."
        )
        if observed is not None:
            error = abs(value - observed)
            baseline_error = abs(now - observed)
            st.success(
                f"En este ejemplo, la predicción se aleja {error:,.0f} unidades del valor real. Mantener el consumo actual como pronóstico se habría alejado {baseline_error:,.0f} unidades."
            )
            st.bar_chart(
                pd.DataFrame(
                    {"Demanda": [now, value, observed]},
                    index=["Al partir", "Predicción +1 h", "Real +1 h"],
                )
            )
            st.caption(
                "Es una reproducción histórica: algunos ejemplos pueden pertenecer al entrenamiento. La evaluación formal está en la pestaña 5."
            )
        st.subheader("¿En qué información se apoya?")
        st.write(
            "Consumo de las tres zonas, valores recientes y del día anterior, calendario y condiciones meteorológicas observadas."
        )
        with st.expander("Ver las variables exactas de esta predicción"):
            st.dataframe(current.rename("Valor conocido al partir"))
        st.caption(
            "Modelo activo: "
            + bundle["winner"]
            + f" · {len(bundle['columns'])} variables · elegido con validación temporal"
        )
    except (ValueError, FileNotFoundError) as exc:
        st.error(str(exc))
with evaluation:
    st.subheader("¿Cómo sabemos si predice bien?")
    st.markdown(
        "**MAE**: error absoluto promedio, en las mismas unidades del consumo. **RMSE**: penaliza más los errores grandes. En ambos, un valor menor es mejor. **R²**: compara con una predicción constante; cerca de 1 es mejor."
    )
    st.caption(
        "Resultados del experimento guardado, independientes del CSV cargado. Test no se utiliza para seleccionar modelos."
    )
    for name, title in [
        ("validation", "Validación — selección"),
        ("test", "Test — evaluación final"),
    ]:
        report = ROOT / f"artifacts/{name}.csv"
        if report.exists():
            table = pd.read_csv(report)
            st.subheader(title)
            st.dataframe(table, width="stretch")
            st.bar_chart(table.set_index("model")["rmse"])
    report = ROOT / "artifacts/predictions.csv"
    if report.exists():
        errors = pd.read_csv(report, index_col=0, parse_dates=True)
        st.line_chart(errors[["observed", "predicted"]].iloc[:1008])
        errors["absolute_error"] = errors.error.abs()
        st.subheader("MAE por hora")
        st.bar_chart(errors.groupby(errors.index.hour).absolute_error.mean())
        st.subheader("20 errores más grandes")
        st.dataframe(errors.nlargest(20, "absolute_error"))
with interpretation:
    st.subheader("¿Qué señales utiliza más?")
    st.write(
        "Cada barra muestra cuánto empeora el error al desordenar una variable. Cuanto más aumenta el error, más depende el modelo de esa señal. Esto explica asociaciones; no demuestra causas."
    )
    report = ROOT / "artifacts/importance.csv"
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
