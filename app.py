import joblib
import pandas as pd
import streamlit as st

from tetouan.data import features, read
from tetouan.inference import predict

st.title("TETOUAN-POWER-ML")
st.write(
    "Zona 1: horizonte de 60 minutos. Unidades originales del dataset; UCI no especifica la unidad en su tabla."
)
upload = st.file_uploader("Historial CSV continuo cada 10 minutos", type="csv")
if upload:
    try:
        frame = read(upload)
        time = st.selectbox("Instante de referencia", features(frame).index)
        bundle = joblib.load("artifacts/model.joblib")
        current = features(frame.loc[:time]).iloc[-1]
        outside = (current < bundle["training_min"]) | (
            current > bundle["training_max"]
        )
        if outside.any():
            st.warning(
                "Variables fuera del rango de entrenamiento: "
                + ", ".join(current.index[outside])
            )
        value = predict(bundle, frame.loc[:time])
        st.metric(str(time + pd.Timedelta(minutes=60)), f"{value:,.2f}")
        st.write("Modelo:", bundle["winner"])
        st.line_chart(frame.iloc[:, -3:])
    except (ValueError, FileNotFoundError) as exc:
        st.error(str(exc))
