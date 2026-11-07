# Verificación real

- Instalación en venv nuevo con Python 3.12 completada; dependencias fijadas en requirements.txt.
- UCI descargado y cotejado byte a byte con archivo del usuario.
- 6 pruebas propias aprobadas: causalidad, horizonte, purga, serialización/inferencia, esquema/continuidad y apertura de Streamlit.
- Ruff y pip check aprobados.
- Nueve regresores entrenados incluyendo RealMLP; dos baselines deterministas evaluados.
- Modelos clásicos y RealMLP entrenados con todas las filas disponibles en entrenamiento.
- Doce ensayos Optuna de Ridge con tres ventanas expansivas y gap=6 completados.
- Recarga del modelo ganador y predicción completadas.
- GPU: torch.cuda.is_available() devuelve False.
- Avisos externos: PyTorch/Lightning muestran deprecaciones; PyTabKit avisa de array no escribible de pandas. No impidieron entrenamiento ni predicción.
- Primera prueba Streamlit falló por ruta relativa del test; corregida y repetida con éxito.
- Recolección inicial de pytest incluyó tests de vendor; se interrumpió y se configuró testpaths=tests. No se afirma haber aprobado la suite completa upstream.
- No se afirma haber ejecutado GitHub Actions remoto hasta contar con evidencia del servicio.

- Notebook ejecutado de principio a fin con nbclient en Python 3.12. Se retiraron salidas de consola con rutas locales; tablas de resultados conservadas. Colab remoto no ejecutado.
