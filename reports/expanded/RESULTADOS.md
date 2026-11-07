# Ampliación: evidencia y límites

Se pasó de 26 a 59 predictores causales. Siete pruebas aprobadas, incluyendo causalidad de todas las nuevas variables y apertura de la aplicación. Ruff aprobado.

HistGradientBoosting fue seleccionado exclusivamente por validación (RMSE 852,56); su RMSE test es 1037,76. La selección mantiene el protocolo aunque la regresión lineal obtenga RMSE test 792,64. RealMLP obtiene RMSE test 1114,23: añadir variables no mejoró este modelo.

La versión anterior sigue disponible localmente en artifacts/previous. El test de 2017 ya fue consultado: esta ampliación es exploratoria, no una confirmación independiente. Se requiere evaluación en nuevos periodos y ventanas temporales antes de sostener mejoras generales.

Análisis añadidos: distribuciones, asimetría, correlación con objetivo futuro, pares con correlación absoluta >0,95, diccionario de disponibilidad, perfiles diarios/semanales, relaciones meteorológicas, errores extremos e importancia por permutación.

Documento académico revisado: Proyecto Machine Learning (1).pdf, UPTC. Se amplían fases 3, 5, 6 y 7; no se afirma satisfacer automáticamente todos los criterios gráficos del documento. Notebook actualizado; sus nuevas celdas de análisis se ejecutaron como módulo, pero el notebook ampliado aún no se ejecutó completo.
