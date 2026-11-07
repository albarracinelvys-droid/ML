# TETOUAN-POWER-ML

Predicción de demanda de Zona 1 a 60 minutos. Aplicación académica de PyTabKit,
sin modificar sus modelos internos. Código original incluido como submódulo en
`vendor/pytabkit`, SHA `d232eecd47b127b29da0378423cb475b88fb0e80`, versión 1.8.0.
Licencia Apache 2.0; autores originales David Holzmüller, Léo Grinsztajn e Ingo Steinwart.

## Instalación Windows

```powershell
git clone --recurse-submodules --branch feature/tetouan-power-ml https://github.com/albarracinelvys-droid/ML.git
cd ML
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .
.venv/Scripts/python -m pytest
.venv/Scripts/python -m tetouan.train
.venv/Scripts/python -m streamlit run app.py
```

El entrenamiento descarga los datos oficiales si no existen. Para el archivo local:
`python -m tetouan.train --data "ruta/Tetuan City power consumption.csv"`.
Los modelos se guardan localmente en `artifacts/`, fuera de Git.
No cargar archivos joblib de terceros: el formato ejecuta código al deserializar.

## Metodología

Pregunta: ¿qué familia ofrece mejor precisión, eficiencia e interpretabilidad?
Objetivo `y(t+60min)`; observaciones meteorológicas y consumo disponibles en t.
Se verifica continuidad de diez minutos antes de desplazar seis filas.
Rezagos, ventanas causales y calendario cíclico; normalización aprendida solo en entrenamiento.
Partición cronológica 70/15/15 con purga de seis filas entre conjuntos.
RealMLP recibe validación explícita, sin partición aleatoria interna.
Selección por RMSE de validación; test reservado para evaluación final.
Persistencia diaria usa consumo de t-23h para predecir t+1h.

Comparación: persistencia, estacional, Dummy, lineal, Ridge, Random Forest,
HistGradientBoosting, XGBoost, LightGBM, CatBoost y RealMLP.
Presupuesto inicial RealMLP: 16 épocas en CPU (no equivale a una comparación
exhaustiva de los defaults del artículo). Cambiar con `--epochs 256`.
Métricas: MAE, RMSE, R², tiempo; errores por hora/día; permutation importance en validación.

## Datos y atribuciones

Salam y El Hibaoui (2018), UCI 849, licencia CC BY 4.0:
https://doi.org/10.24432/C5B034
La página UCI declara 52.417 instancias; el conteo real se registra en audit.json.
No se asumen unidades ausentes de la tabla oficial.
PyTabKit: https://github.com/dholzmueller/pytabkit
Artículo: https://arxiv.org/abs/2407.04491

## Resultados ejecutados

Entorno limpio Python 3.12; 52.416 filas oficiales, checksum idéntico al CSV aportado.
RealMLP fue seleccionado por validación (RMSE 880,55). En test: MAE 588,63,
RMSE 798,93, R² 0,98265. Persistencia: RMSE 2504,18; estacional: 1486,51;
lineal: 991,86. Ver tablas completas y figuras en `reports/`.
Se ejecutaron seis pruebas del proyecto, Ruff, pip check y 12 ensayos Optuna para Ridge.
CUDA no disponible en el entorno utilizado. Modelo entrenado localmente en `artifacts/model.joblib`.

## Limitaciones

Un año y una ciudad no permiten afirmar generalización geográfica ni plurianual.
Permutation importance muestra asociación, no causalidad; predictores correlacionados
pueden repartir importancia. Las métricas de una partición no demuestran superioridad estadística.
Incluye Optuna para Ridge sobre tres ventanas expansivas purgadas:
`python -m tetouan.tune`. Sus resultados son diagnósticos independientes; no reemplazan
automáticamente el modelo seleccionado por el benchmark.
Pendientes de ampliación: walk-forward para todas las familias, optimización de los líderes, intervalos con dependencia temporal,
SHAP local, análisis formal de deriva, validación GPU y ejecución alojada en Colab.
El repositorio no contiene datos ni modelos pesados. Consulte `reports/` para evidencia ejecutada.
