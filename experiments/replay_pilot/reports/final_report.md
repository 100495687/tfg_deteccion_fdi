# Piloto de ataques replay sobre P OR H — informe final

## 0. Resumen ejecutivo

El piloto evaluó 60 episodios de replay (2 desplazamientos x 3 duraciones x 10 destinos emparejados) sobre `FINAL_CAL_POOL_180`, con la arquitectura `P OR H` congelada (P=0.049748, H=0.883628). **Resultado: el replay es prácticamente indetectable por la señal actual** — DR inducido OR = 1.67% (1/60 episodios), 0% en la zona interior del ataque. Sin embargo, H sí "nota" el ataque: su score sube una mediana de +0.65 (hasta +0.90), quedando sistemáticamente justo por debajo del umbral calibrado. **Conclusión: C — replay mayoritariamente indetectable por señal; priorizar mecanismos edge de frescura/secuencia/autenticación.**

Durante la construcción del piloto se detectaron y corrigieron **dos bugs técnicos reales** (no motivados por resultados, ambos capturados por smoke-tests/tests antes o independientemente de observar tasas de detección finales) — documentados íntegramente con las versiones v1/v2 preservadas sin sobrescribir.

## 1. Bugs técnicos detectados y corregidos (transparencia obligatoria, sección 13)

**v1 → v2**: el criterio de contexto causal para H exigía 7 días de historia respecto al inicio de *toda* la serie pre-test, no respecto al inicio del propio `FINAL_CAL_POOL_180`. `predictor_causal_lags.puntuar_ataques` descarta internamente cualquier bin con índice `t < MAX_LAG` **relativo a la rejilla densa del pool**, así que 4/60 episodios con destino en los primeros 7 días del pool nunca producían filas de H (ni limpias ni atacadas) — detectado en un smoke-test de 3 episodios, antes de cualquier resultado final.

**v2 → v3**: la selección de destinos se hacía de forma independiente por duración (120/360/1440 min), comprobando solapes solo dentro de la misma duración. 6 pares de destinos de duraciones distintas se solapaban temporalmente. **Esto no corrompió ningún score ya calculado** (cada episodio se inyecta sobre una copia independiente de la serie limpia, `construir_kw15_atacado` nunca comparte estado entre episodios) — era un problema de higiene/interpretabilidad del diseño del manifiesto (criterio 15), detectado por un test de no-solape, no por resultados de detección. Corregido acumulando los destinos ya elegidos entre duraciones.

Ambas versiones anteriores (`manifests/v1_superseded/`, `manifests/v2_superseded/`) quedan preservadas con nota de corrección explícita. El manifiesto activo es v3.

## 2. Periodo, arquitectura y verificación

**Periodo**: `FINAL_CAL_POOL_180` (2009-06-18 → 2009-12-15, 180 días), el único periodo pre-test con un modelo H ya entrenado y congelado en disco (`models/final_or_pretest/histgb_periodic_final.joblib`) — justificado por disponibilidad de artefactos, no por tasa de detección esperada (ver `reports/period_selection_justification.json`).

**Dirección de scores verificada en código** (no asumida): `active_P = score_P > 0.049748`, `active_H = score_H > 0.883628` (estrictamente `>`, nunca `>=`), confirmado en `fusion_p_histgb.estado_ffill`/`construir_estados_episodio`.

**Cero llamadas a `fit()`** en `evaluate_replay.py`/`inject_replay.py` (verificado por test); solo `modelo_h_final.predict()` y `modelo_p.reconstruir()` (autoencoder, inferencia) sobre checkpoints ya congelados.

## 3. Resultados principales

| Métrica | P | H | OR |
|---|---|---|---|
| DR estándar | 1.67% | 1.67% | 3.33% |
| DR inducido | 0.00% | 1.67% | 1.67% |
| DR interior | 0.00% | 0.00% | 0.00% |
| % boundary_only | 0.0% | 1.67% | 1.67% |
| % post_only | 0.0% | 0.0% | 0.0% |
| DR energético (sobre infrarreporte) | 0.0% | 13.0% | 13.0% |

**El único episodio detectado** (`D1440_002_WEEKLY`, 1440 min, infrarreporte) se detecta en la **zona final** (últimos 60 min de 1440), con retraso de 1401 minutos (~97% del ataque ya transcurrido) — es decir, ni siquiera este caso aislado refleja "reconocimiento del contenido histórico": es una detección tardía, de frontera/acumulación, no de contenido.

**Complementariedad P/H**: `H_only=1, P_only=0, both=0, none=59`. P no detecta ningún replay de forma inducida en este piloto.

**Categoría económica**: 33/60 infrarreporte, 26/60 sobrerreporte, 1/60 casi neutro — el replay no está sesgado hacia ocultar energía (es "accidentalmente" bidireccional, coherente con que el origen no se seleccionó por energía).

**Diario vs. semanal**: DAILY 0% DR inducido, WEEKLY 3.33% (el único detectado). Con n=1 detección total, esta diferencia **no es estadísticamente concluyente**, pero es consistente con la hipótesis de la sección 18 (el replay diario coincide con `lag_96`/24h de H, potencialmente más "plausible" para el predictor que el semanal).

**Duración**: 0% en 120/360 min, 5% en 1440 min — el único caso detectado requirió la duración máxima estudiada.

## 4. El hallazgo más importante: el detector "nota" el ataque pero no cruza el umbral

`max_delta_score_H` (máximo incremento de score H atacado vs. limpio en la ventana del episodio): mediana **+0.65**, hasta **+0.90**. `max_score_H_attack` nunca superó 0.875 (umbral=0.8836). Esto **no es un fallo silencioso** — el ataque desplaza el score de forma sustancial y sistemática, simplemente el umbral (calibrado para una FA muy baja, ~0.06/día) deja un margen que el replay no llega a cruzar en la inmensa mayoría de los casos. Este hallazgo conecta directamente con el experimento `threshold_tradeoff` anterior: bajar el umbral aumentaría FA sin margen operativo razonable (ya demostrado allí sobre ataques reduccion/rampa/bypass) — sería razonable esperar el mismo problema aquí, aunque no se ha barrido umbrales en este piloto (fuera de alcance, sección 4).

## 5. Respuestas a las 32 preguntas obligatorias

1. **¿60 episodios válidos?** Sí, los 60 en v3, tras corregir 2 bugs técnicos.
2. **¿Destinos emparejados?** Sí, 30 destinos, cada uno con episodio DAILY y WEEKLY sobre exactamente la misma ventana destino.
3. **¿Solo pre-test?** Sí, exclusivamente `FINAL_CAL_POOL_180`; test nunca accedido (verificado por test y por ausencia de referencias en el código).
4. **¿Modelos congelados?** Sí, cero `fit()`, solo `predict()`/`reconstruir()` sobre checkpoints existentes.
5. **¿Copia exacta?** Sí, verificado bit a bit (`exact_copy_verification.csv`, 0 mismatches en 60/60).
6. **¿H usó historia atacada causalmente?** Sí (`pcl.puntuar_ataques`, reutilizado literalmente, recalcula features sobre la serie atacada completa).
7. **DR estándar OR**: 3.33%.
8. **DR inducido OR**: 1.67%.
9. **DR interior OR**: 0.00%.
10. **% boundary_only**: 1.67%.
11. **% post_only**: 0.00%.
12. **Episodios con alerta limpia preexistente**: 0/60.
13. **¿P o H detectó más?** H (1 inducido) vs P (0 inducidos).
14. **Complementariedad**: nula en la práctica (H_only=1, resto=0).
15. **¿Diario más difícil?** Sí en esta muestra (0% vs 3.33% semanal), no concluyente con n=1.
16. **¿Semanal más difícil?** No, fue el único detectado — de nuevo, no concluyente.
17. **¿Duración?** Solo 1440 min produjo detección (5%); 120/360 min, 0%.
18. **Categoría económica más frecuente**: infrarreporte (33/60), seguida de cerca por sobrerreporte (26/60).
19. **Energía oculta total**: 95.51 kWh (sobre los 33 episodios de infrarreporte).
20. **DR energético**: 13.0% (OR y H idénticos, ya que solo H detectó).
21. **¿Relación con salto de inicio?** No evaluable de forma concluyente con 1 detección; salto medio de inicio ≈ -0.009 kW (prácticamente neutro en promedio, alta varianza ±1.08 kW).
22. **¿Relación con salto final?** Similar; el único detectado se activó en la zona final, compatible con relevancia del salto de cierre, pero no se puede generalizar.
23. **¿Relación con similitud origen-destino?** Correlación origen-destino media = 0.17 (baja/moderada) — el replay no reproduce series muy parecidas al destino real en general; no hay señal clara de que baja similitud facilite la detección con esta única muestra positiva.
24. **¿Relación con diferencia energética?** El episodio detectado es de infrarreporte, consistente con mayor "anomalía" de reducción real de reporte, pero un solo caso no permite establecer relación estadística.
25. **¿El lag coincidente reduce la sensibilidad de H?** Hipótesis no refutada: los deltas de score son ligeramente mayores en DAILY (mediana 0.675) que WEEKLY (mediana 0.654) pero el ÚNICO cruce de umbral ocurrió en WEEKLY — resultado mixto/no concluyente con esta muestra.
26. **¿El detector reconoce contenido o solo empalme?** Ni siquiera el empalme, mayoritariamente: 0% de detección en fronteras de inicio, y el único hit fue en "final" tras casi todo el ataque, no en el arranque. El patrón dominante es la NO detección.
27. **¿Debe ampliarse el experimento?** No en su forma actual (más duraciones del mismo tipo de ataque) — ver conclusión.
28. **¿Debe estudiarse replay suavizado?** No prioritario: el problema no es el empalme (que apenas genera alertas), sino que el contenido histórico es estadísticamente plausible.
29. **¿Debe implementarse protección anti-replay en edge?** Sí — es la recomendación central de este informe (ver conclusión C).
30. **¿Se modificó algún artefacto anterior?** No (verificado: `threshold_p_final.json`/`threshold_h_final.json`/checkpoints sin cambios, `models/`/`results/` no tocados por este piloto).
31. **¿Se usó el test final?** No, en ningún momento.
32. **¿Se entrenó algún modelo?** No, cero llamadas a `fit()`.

## 6. Conclusión obligatoria: **C. Replay mayoritariamente indetectable por señal**

El consumo histórico reproducido permanece, en la inmensa mayoría de los casos (98.3%), dentro de lo que P y H consideran normal — pese a que H registra un incremento de score sustancial y sistemático que nunca llega a traducirse en alarma bajo el umbral operativo actual. Esto es coherente con la naturaleza del ataque: los valores replay son consumo *real*, no sintético, por lo que un detector basado en aprender el patrón estadístico del consumo tiene un límite estructural para distinguirlo del consumo genuino, sea cual sea el umbral (ya se demostró en `threshold_tradeoff` que este sistema no tiene margen operativo para bajar umbrales de forma general).

**Recomendación**: priorizar mecanismos de seguridad a nivel edge que SÍ pueden detectar replay de forma determinista — verificación de frescura de timestamp, números de secuencia monotónicos y/o autenticación de mensajes — en vez de intentar resolver esto con más ajuste de la señal estadística P/H.

## 7. Limitaciones

- Piloto de alcance reducido (60 episodios, 3 duraciones, 1 periodo) — no representativo de todas las posiciones/estaciones posibles; explícitamente no pretendía serlo (sección 4).
- Una sola detección total impide cualquier conclusión estadísticamente robusta sobre diferencias diario/semanal, por duración, o por salto de frontera — se reportan como observaciones descriptivas, no inferencias.
- El periodo (`FINAL_CAL_POOL_180`) es el único técnicamente disponible sin reentrenar H; no se ha verificado si otros periodos pre-test (walk-forward folds) mostrarían el mismo patrón.
- No se ha estudiado el efecto de escalar el ataque a más duraciones, posiciones aleatorias, o variantes suavizadas/adversariales — deliberadamente fuera de alcance de este piloto.

## 8. Recomendación para el experimento definitivo (si se decide ampliar)

Dado el resultado (Categoría C), **no se recomienda ampliar el experimento de señal estadística** (más duraciones/posiciones del mismo replay puro) como siguiente paso de investigación — el techo de detectabilidad por contenido parece estructural, no un artefacto de tamaño muestral. Si se desea invertir tiempo adicional, es más rentable (y así se ha discutido) documentar este hallazgo como limitación arquitectónica en la memoria y, si el calendario lo permite, esbozar (sin implementar) el diseño de un mecanismo anti-replay a nivel edge como trabajo futuro.
