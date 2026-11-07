# Conclusiones del experimento inicial

1. La continuidad temporal validada permite representar exactamente 60 minutos mediante seis pasos. La purga debe mantenerse en cualquier ampliación.
2. El baseline diario supera a persistencia inmediata en test (RMSE 1486,51 frente a 2504,18): conservar señales de calendario y rezagos diarios.
3. La regresión lineal supera a varios árboles en el periodo final. Complejidad no garantiza mejor extrapolación; mantenerla como referencia.
4. RealMLP obtiene el menor RMSE de validación y test con el presupuesto inicial. Repetir con múltiples ventanas y semillas antes de atribuir superioridad general.
5. Permutation importance destaca consumo actual y lag_138, seguido de calendario/lag_144. Estas asociaciones apoyan la ingeniería histórica, pero la correlación entre rezagos impide interpretar importancias como causalidad.
