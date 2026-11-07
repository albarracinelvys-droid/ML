# TETOUAN-POWER-ML

PredicciÃ³n de demanda de Zona 1 a 60 minutos. AplicaciÃ³n acadÃ©mica de PyTabKit,
sin modificar sus modelos internos. CÃ³digo original incluido como submÃ³dulo en
`vendor/pytabkit`, SHA `d232eecd47b127b29da0378423cb475b88fb0e80`, versiÃ³n 1.8.0.
Licencia Apache 2.0; autores originales David HolzmÃ¼ller, LÃ©o Grinsztajn e Ingo Steinwart.

## InstalaciÃ³n Windows

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
No cargar archivos joblib de terceros: el formato ejecuta cÃ³digo al deserializar.

## MetodologÃ­a

Pregunta: Â¿quÃ© familia ofrece mejor precisiÃ³n, eficiencia e interpretabilidad?
Objetivo `y(t+60min)`; observaciones meteorolÃ³gicas y consumo disponibles en t.
Se verifica continuidad de diez minutos antes de desplazar seis filas.
Rezagos, ventanas causales y calendario cÃ­clico; normalizaciÃ³n aprendida solo en entrenamiento.
ParticiÃ³n cronolÃ³gica 70/15/15 con purga de seis filas entre conjuntos.
RealMLP recibe validaciÃ³n explÃ­cita, sin particiÃ³n aleatoria interna.
SelecciÃ³n por RMSE de validaciÃ³n; test reservado para evaluaciÃ³n final.
Persistencia diaria usa consumo de t-23h para predecir t+1h.

ComparaciÃ³n: persistencia, estacional, Dummy, lineal, Ridge, Random Forest,
HistGradientBoosting, XGBoost, LightGBM, CatBoost y RealMLP.
Presupuesto inicial RealMLP: 16 Ã©pocas en CPU (no equivale a una comparaciÃ³n
exhaustiva de los defaults del artÃ­culo). Cambiar con `--epochs 256`.
MÃ©tricas: MAE, RMSE, RÂ², tiempo; errores por hora/dÃ­a; permutation importance en validaciÃ³n.

## Datos y atribuciones

Salam y El Hibaoui (2018), UCI 849, licencia CC BY 4.0:
https://doi.org/10.24432/C5B034
La pÃ¡gina UCI declara 52.417 instancias; el conteo real se registra en audit.json.
No se asumen unidades ausentes de la tabla oficial.
PyTabKit: https://github.com/dholzmueller/pytabkit
ArtÃ­culo: https://arxiv.org/abs/2407.04491

## Resultados ejecutados

Entorno limpio Python 3.12; 52.416 filas oficiales, checksum idÃ©ntico al CSV aportado.
RealMLP fue seleccionado por validaciÃ³n (RMSE 880,55). En test: MAE 588,63,
RMSE 798,93, RÂ² 0,98265. Persistencia: RMSE 2504,18; estacional: 1486,51;
lineal: 991,86. Ver tablas completas y figuras en `reports/`.
Se ejecutaron seis pruebas del proyecto, Ruff, pip check y 12 ensayos Optuna para Ridge.
CUDA no disponible en el entorno utilizado. Modelo entrenado localmente en `artifacts/model.joblib`.

## Limitaciones

Un aÃ±o y una ciudad no permiten afirmar generalizaciÃ³n geogrÃ¡fica ni plurianual.
Permutation importance muestra asociaciÃ³n, no causalidad; predictores correlacionados
pueden repartir importancia. Las mÃ©tricas de una particiÃ³n no demuestran superioridad estadÃ­stica.
Incluye Optuna para Ridge sobre tres ventanas expansivas purgadas:
`python -m tetouan.tune`. Sus resultados son diagnÃ³sticos independientes; no reemplazan
automÃ¡ticamente el modelo seleccionado por el benchmark.
Pendientes de ampliaciÃ³n: walk-forward para todas las familias, optimizaciÃ³n de los lÃ­deres, intervalos con dependencia temporal,
SHAP local, anÃ¡lisis formal de deriva, validaciÃ³n GPU y ejecuciÃ³n alojada en Colab.
El repositorio no contiene datos ni modelos pesados. Consulte `reports/` para evidencia ejecutada.


## Ampliación de variables y análisis

La versión ampliada usa 59 predictores (antes 26): rezagos hasta 24 horas,
variabilidad, mínimos y máximos históricos, cambio diario, consumo relativo,
meteorología reciente, interacción temperatura-humedad, consumo total,
participación de Zona 1 y señales históricas de las otras zonas.
Todas están disponibles en t; el objetivo sigue siendo Zona 1 en t+60 minutos.
No se incorporan festivos locales sin una fuente validada.

`python -m tetouan.analyze` genera distribuciones, asimetría, correlaciones con
el objetivo futuro y pares redundantes usando solo entrenamiento.
La aplicación incorpora seis pestañas y abre el dataset local por defecto.
Los gráficos del CSV cargado son exploratorios; las métricas e importancias
pertenecen al experimento guardado, no al archivo subido.

El test ya fue observado al desarrollar la versión original: la comparación
entre versiones es exploratoria y requiere nuevos periodos de evaluación.
Más variables no prueban superioridad. Los consumos pasados no son el objetivo
futuro: pueden usarse si están realmente disponibles en el instante de emisión.
