# Detección no supervisada de fraude eléctrico residencial (per-meter)

Detector de infrarreporte de consumo (fraude deductivo) para un único hogar, entrenado sin
etiquetas de fraude real — solo con consumo limpio. Dataset UCI "Individual Household Electric
Power Consumption" (id=235), variable objetivo `Global_active_power` (kW).

## Arquitectura: P OR H

El detector combina dos señales independientes con un OR:

- **P** — un autoencoder convolucional temporal (TCN-AE, `src/models/tcn_ae.py`) que reconstruye
  ventanas de 6h de consumo limpio; cuanto peor reconstruye una ventana nueva, más sospechosa es.
- **H** — un HistGradientBoostingRegressor causal (`src/predictor_causal_lags.py`) que predice
  el consumo a 15 min a partir de lags periódicos (día/semana anteriores) y calendario.

Se calibran por separado y se congelan en `models/final_or_pretest/`
(`OR_PMULTI_HMULTIWINDOW` = `P_MULTI_SEASON` OR `H_MULTIWINDOW_180`). Usar dos detectores en vez
de uno compensa que cada uno falla en casos distintos: el OR conserva el 99.2% de lo que
detectaba P por su cuenta, y el 100% de lo que detectaba H por la suya.

La evaluación final sobre test se etiqueta `RETROSPECTIVE_CONFIRMATORY_EVALUATION`, no como test
"virgen": una versión anterior del pipeline sí llegó a acceder a esa partición en el pasado. La
arquitectura final nunca vio esos datos ni se diseñó consultándolos, pero se documenta la
distinción por honestidad metodológica en vez de llamarlo test prístino sin más.

## Ataques evaluados

7 formas de infrarreporte, repartidas en 3 sitios del código:

- **`src/attacks.py`**: reducción constante, reducción variable, bypass total, bypass residual,
  recorte de picos — transformaciones puras de la ventana actual (`y = h(x)`).
- **`src/experimento_ramp.py`**: rampa (reducción que crece linealmente en el tiempo).
- **`experiments/replay_pilot/`**: replay (sustituir un tramo por otro tramo real de otro
  momento de la serie) — necesita ventana donante + manifiesto, no encaja como función pura.

## Resultados (evaluación final sobre test)

| | P | H | OR |
|---|---|---|---|
| DR energético (% de energía oculta que se detecta) | 59.8% | 69.8% | **73.3%** |
| Falsos positivos | 0.017/día | 0.061/día | 0.078/día |

Clasificación final: `CONFIRMATORY_SUPPORT`, admisible operativamente.

**Limitación conocida, sin resolver por señal**: el replay es prácticamente indetectable por
contenido (DR inducido de solo 1.67% en el piloto dedicado). Por eso el despliegue Edge añade
autenticación en vez de seguir ajustando el modelo (ver más abajo).

## Estructura

```
deteccion_fraude_no_supervisada/
├── configs/base.yaml         # unica fuente de parametros del pipeline offline
├── data/{raw,processed}/     # cache del dataset (no versionado)
├── models/                   # checkpoints y artefactos congelados (P, H, OR)
├── src/                      # pipeline offline: nucleo (11 modulos) + experimentos
├── experiments/replay_pilot/ # piloto de replay, aislado
├── tests/
├── results/{tables,figures}/
└── edge_deployment/          # despliegue: motor online, API, Docker, seguridad
```

## Cómo ejecutar el pipeline offline

```bash
pip install pandas numpy scikit-learn scipy matplotlib pyyaml joblib pyarrow ucimlrepo
pip install torch --index-url https://download.pytorch.org/whl/cpu   # build CPU, mas ligera

python -m src.data_loading        # descarga y cachea el dataset (tarda solo la primera vez)
python -m src.splitting           # verifica las particiones train/val/test
python -m src.windowing           # ventaneo
python -m src.normalization       # normalizacion z-score
python -m src.models.tcn_ae       # entrena (o carga si ya existe) el checkpoint de P
python -m src.evaluation          # evaluacion con las 5 familias de attacks.py
```

Cadena completa para reconstruir la arquitectura final `P OR H` desde cero (cada paso reutiliza
los artefactos que dejaron los anteriores):

```bash
python -m src.predictor_causal_lags         # entrena H
python -m src.fusion_p_histgb               # fusiona P y H
python -m src.optimizacion_histgb_periodic  # optimiza H con validacion walk-forward
python -m src.robustez_temporal_p           # calibra P robusto (nace P_MULTI_SEASON)
python -m src.calibracion_temporal_h        # calibra H (nace H_MULTIWINDOW_180)
python -m src.auditoria_final_p_vs_or       # decide P solo vs fusion OR
python -m src.congelacion_final_or_pretest  # congela la arquitectura final en models/final_or_pretest/
python -m src.evaluacion_final_retrospectiva_test   # evaluacion unica sobre test
```

## Cómo ejecutar el despliegue Edge

El modelo ya congelado (`models/final_or_pretest/`) se sirve como microservicio HTTP. No hace
falta ningún dataset en tiempo de ejecución: los artefactos van horneados en la imagen.

**En local, sin Docker:**

```bash
pip install -r edge_deployment/requirements-edge.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu

uvicorn edge_deployment.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

**Con Docker (imagen de producción):**

```bash
python edge_deployment/build_docker_context.py
docker build -t fdia-edge:local -f edge_deployment/Dockerfile edge_deployment/docker_context
docker run --rm -p 8000:8000 fdia-edge:local
```

**Probar que responde** (en otra terminal, con el servicio ya arrancado):

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready

curl -X POST http://localhost:8000/readings \
  -H "Content-Type: application/json" \
  -d '{"meter_id": "house_01", "timestamp": "2010-01-01T00:00:00", "power_kw": 1.2}'
```

La respuesta incluye si la lectura fue aceptada, los scores de P y H (cuando toque evaluarlos) y
si hay alarma (`alert_or`). Encadenar más lecturas del mismo `meter_id` hace avanzar el estado
interno del motor (buffers, última evaluación de P/H) igual que en un contador real.

**Capa de seguridad opcional (anti-replay)**: activable con la variable de entorno
`FDIA_ANTI_REPLAY_ENABLED=true` al arrancar el contenedor. Añade autenticación HMAC-SHA-256,
número de secuencia y frescura de timestamp sobre una ruta paralela (`POST /secure-readings`),
sin modificar ni sustituir `/readings`.
