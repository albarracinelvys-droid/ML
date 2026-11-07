# Auditoría de integración

- Destino ML: vacío al iniciar, rama de desarrollo independiente; sin reemplazos de trabajos previos.
- Original clonado en `vendor/pytabkit` como submódulo, sin cambios internos.
- Tag v1.8.0 y HEAD auditado coinciden: d232eecd47b127b29da0378423cb475b88fb0e80.
- Revisados README, pyproject.toml, tests/test_sklearn_interfaces.py y sklearn_base.py.
- Clase de regresión verificada en fuente: RealMLP_TD_Regressor.
- fit acepta X, y, X_val, y_val; se pasan particiones cronológicas explícitas.
- No se usa n_cv > 1, que introduciría particiones internas no temporales.
- Apache 2.0 conservada en LICENSE y en el submódulo.
- Base oficial UCI descargada: mismo SHA256 que el archivo entregado.
- Python 3.12 seleccionado para compatibilidad con dependencias científicas.
- Prueba funcional y métricas: consultar evidencia de ejecución; la lectura del código no equivale a validar entrenamiento.
