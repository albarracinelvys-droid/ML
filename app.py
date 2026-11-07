from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from tetouan.data import TARGET, features, read
from tetouan.inference import predict


def explain(reading, finding, decision):
    """Explain a chart using its axes, observed result and practical use."""
    st.markdown("**Cómo leerla:** " + reading)
    st.info("Lo que muestran estos datos: " + finding)
    st.caption("Para qué sirve: " + decision)


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
        "4 · Predicción guiada",
        "5 · Comprueba el modelo",
        "6 · Entiende sus señales",
    ]
)
with summary:
    a, b, c = st.columns(3)
    a.metric("Registros", f"{len(frame):,}")
    b.metric("Variables predictoras", len(features(frame).columns))
    c.metric("Valores faltantes", int(frame.isna().sum().sum()))
    st.write("Periodo", str(frame.index.min()), "a", str(frame.index.max()))
    st.subheader("¿Qué representa cada fila?")
    st.write(
        "Una medición cada 10 minutos: cinco variables meteorológicas y el consumo eléctrico de tres zonas. Las 59 variables del modelo se calculan a partir de este historial; no son 59 sensores distintos."
    )
    with st.expander("Diccionario de estadísticas"):
        st.write(
            "count: número de mediciones · mean: promedio · std: variabilidad · min y max: extremos · 25%, 50% y 75%: cuartiles. El 50% es la mediana, el punto que divide los datos en dos mitades."
        )
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
    st.subheader("Cómo cambia el consumo con el tiempo")
    daily = frame[zone].resample(frequency).mean()
    st.line_chart(
        daily.rename("Consumo promedio"), x_label="Fecha", y_label="Demanda media"
    )
    explain(
        "El eje horizontal es la fecha; el vertical es el consumo promedio del intervalo elegido. Una subida significa mayor demanda, no mayor error del modelo.",
        f"El promedio más alto de la agregación elegida es {daily.max():,.0f}, en {daily.idxmax():%d/%m/%Y %H:%M}; el más bajo es {daily.min():,.0f}.",
        "Identificar periodos de mayor demanda y cambios de nivel. Un promedio diario puede ocultar picos de diez minutos.",
    )
    left, right = st.columns(2)
    left.subheader("Perfil por hora")
    hourly = frame.groupby(frame.index.hour)[zone].mean()
    left.line_chart(
        hourly.rename("Promedio por hora"),
        x_label="Hora del día",
        y_label="Demanda media",
    )
    left.caption(
        f"Cada punto reúne todos los días del archivo a esa hora. El máximo promedio ocurre a las {hourly.idxmax():02d}:00. No representa un día específico."
    )
    right.subheader("¿Qué días consumen más?")
    weekdays = [
        "Lunes",
        "Martes",
        "Miércoles",
        "Jueves",
        "Viernes",
        "Sábado",
        "Domingo",
    ]
    weekly = frame.groupby(frame.index.day_of_week)[zone].mean()
    weekly.index = [weekdays[i] for i in weekly.index]
    right.bar_chart(weekly.rename("Demanda media"), sort=False)
    right.caption(
        f"La barra más alta corresponde a {weekly.idxmax()}. Son promedios: no todos los días de esa categoría consumen lo mismo."
    )
    st.subheader("Mapa de una semana típica")
    grid = (
        frame.assign(Hora=frame.index.hour, Día=frame.index.day_of_week)
        .groupby(["Día", "Hora"])[zone]
        .mean()
        .reset_index(name="Demanda")
    )
    grid["Día"] = grid["Día"].map(dict(enumerate(weekdays)))
    heatmap = (
        alt.Chart(grid)
        .mark_rect()
        .encode(
            x=alt.X("Hora:O", title="Hora del día"),
            y=alt.Y("Día:O", sort=weekdays),
            color=alt.Color("Demanda:Q", scale=alt.Scale(scheme="yelloworangered")),
            tooltip=["Día", "Hora", alt.Tooltip("Demanda:Q", format=",.0f")],
        )
    )
    st.altair_chart(heatmap.properties(height=260), width="stretch")
    explain(
        "Cada casilla cruza una hora y un día de semana. Los tonos más intensos indican mayor consumo promedio.",
        "El mapa resume el patrón semanal del archivo completo; al cambiar de zona también cambia el mapa.",
        "Localizar franjas de mayor demanda para planificar recursos. Es exploración descriptiva, no un pronóstico.",
    )
with climate:
    variable = st.selectbox("Variable meteorológica", list(frame.columns[:5]))
    st.scatter_chart(frame.iloc[::20], x=variable, y=TARGET)
    st.caption(
        "Muestra sistemática de una de cada 20 observaciones. Asociación no implica causalidad."
    )
    correlation = frame[variable].corr(frame[TARGET])
    explain(
        "Cada punto es una observación: clima en horizontal y consumo de Zona 1 en vertical. Una nube ancha indica que el clima por sí solo no explica todo.",
        f"La correlación lineal de {variable} con Zona 1 en el archivo completo es {correlation:.2f}. La gráfica muestra una de cada 20 filas.",
        "Buscar relaciones y posibles interacciones. La estación, la hora y otros factores pueden explicar parte de la asociación.",
    )
    st.subheader("Correlaciones entre variables observadas")
    st.dataframe(frame.corr().style.background_gradient(cmap="RdBu", vmin=-1, vmax=1))
    st.caption(
        "Cada celda compara dos variables: +1 indica que aumentan juntas; −1, que se mueven en sentido contrario; 0 indica poca relación lineal. La diagonal es siempre 1. Una correlación alta entre predictores puede indicar información redundante."
    )
    st.subheader("¿Qué valores del clima son habituales?")
    counts, edges = np.histogram(frame[variable], bins=30)
    st.bar_chart(
        pd.Series(
            counts,
            index=np.round((edges[1:] + edges[:-1]) / 2, 3),
            name="Número de observaciones",
        ),
        x_label=variable,
        y_label="Frecuencia",
    )
    explain(
        "Las barras agrupan valores parecidos de la variable; su altura indica cuántas mediciones caen en cada intervalo.",
        f"La mediana es {frame[variable].median():.2f}; el 50% central está entre {frame[variable].quantile(0.25):.2f} y {frame[variable].quantile(0.75):.2f}.",
        "Reconocer valores frecuentes, asimetrías y extremos. Una barra poco frecuente no demuestra que exista un error de medición.",
    )
with forecast:
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
            ranked = table.sort_values("rmse")
            st.bar_chart(
                ranked.set_index("model")["rmse"],
                sort=False,
                y_label="RMSE · menor es mejor",
            )
            explain(
                "Cada barra es un modelo. Una barra más baja indica menos error, con mayor penalización para los fallos grandes.",
                f"El menor RMSE de esta tabla corresponde a {ranked.iloc[0]['model']}: {ranked.iloc[0]['rmse']:,.2f}.",
                "La validación elige el modelo; el test examina el resultado final. Que otro modelo gane en test no justifica cambiar la selección después de verlo.",
            )
    report = ROOT / "artifacts/predictions.csv"
    if report.exists():
        errors = pd.read_csv(report, index_col=0, parse_dates=True)
        st.subheader("Predicción frente a lo que ocurrió")
        display_errors = (
            errors[["observed", "predicted"]]
            .iloc[:1008]
            .rename(
                columns={
                    "observed": "Real a +60 min",
                    "predicted": "Predicción a +60 min",
                }
            )
        )
        st.line_chart(
            display_errors, x_label="Instante de partida", y_label="Demanda a +60 min"
        )
        explain(
            "Las dos líneas corresponden al mismo instante futuro. Cuanto más se separan, mayor es el error. La fecha del eje es el momento desde el que se emitió el pronóstico.",
            "Se muestran las primeras 1.008 predicciones de test para facilitar la lectura; las métricas usan todo el test.",
            "Revisar si el modelo sigue los picos y las caídas o responde con retraso.",
        )
        errors["absolute_error"] = errors.error.abs()
        st.subheader("MAE por hora")
        st.bar_chart(errors.groupby(errors.index.hour).absolute_error.mean())
        hourly_error = errors.groupby(errors.index.hour).absolute_error.mean()
        st.caption(
            f"La franja con mayor error medio es la de las {hourly_error.idxmax():02d}:00, con {hourly_error.max():,.0f} unidades. Estas horas corresponden al momento de emitir la predicción."
        )
        st.subheader("¿El modelo suele pasarse o quedarse corto?")
        residual_counts, residual_edges = np.histogram(errors.error, bins=35)
        st.bar_chart(
            pd.Series(
                residual_counts,
                index=np.round((residual_edges[1:] + residual_edges[:-1]) / 2),
                name="Predicciones",
            ),
            x_label="Predicción menos valor real",
            y_label="Frecuencia",
        )
        explain(
            "Cero es un acierto exacto. A la derecha de cero el modelo sobreestima; a la izquierda subestima. Las colas largas representan fallos grandes.",
            f"El error con signo promedio es {errors.error.mean():,.1f}; el error absoluto promedio es {errors.absolute_error.mean():,.1f}.",
            "Detectar sesgo y errores extremos. Un promedio con signo cerca de cero puede ocultar errores positivos y negativos que se cancelan.",
        )
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
        st.bar_chart(importance.tail(20), y_label="Aumento del RMSE")
        explain(
            "La barra mide el daño al desordenar esa señal en validación. Más alta significa mayor dependencia del modelo. Un valor negativo puede reflejar ruido o variabilidad del cálculo.",
            f"La señal con mayor importancia en este experimento es {importance.index[-1]}.",
            "Priorizar la calidad de los datos más útiles. Esto describe el modelo global, no la causa de una predicción particular.",
        )
        with st.expander("Traducción de nombres de variables"):
            st.markdown("""- **lag_6**: consumo de hace una hora.
- **lag_138**: consumo de hace 23 horas, útil para comparar con el mismo momento del día anterior al objetivo.
- **mean_144**: promedio de las últimas 24 horas.
- **std / range**: variabilidad y amplitud recientes.
- **hour_sin / hour_cos**: hora codificada como ciclo.
- **zone1_share**: participación de Zona 1 en el consumo total.
- **Temperature / Humidity**: temperatura y humedad observadas.""")
    st.markdown(
        "Las variables incluyen consumo histórico, dinámica diaria, meteorología reciente, relaciones entre zonas y calendario. Las importancias se calculan en validación; variables correlacionadas pueden compartir importancia."
    )
    st.warning(
        "Más variables no garantizan más precisión. Los resultados actuales requieren nuevas ventanas temporales; el test de 2017 ya fue consultado en versiones anteriores."
    )
