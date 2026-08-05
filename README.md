# Detección no supervisada de fraude eléctrico residencial (per-meter)

Detector de infrarreporte de consumo (fraude deductivo) entrenado únicamente sobre el
consumo limpio de un único hogar (dataset UCI "Individual Household Electric Power
Consumption", id=235), sin enfoque poblacional/multi-cliente.

Variable objetivo única: `Global_active_power` (kW). Se ignoran intensidad, tensión,
potencia reactiva y los tres submeters.

Referencias metodológicas en `docs/papers/` (Takiddin: autoencoders + taxonomía de ataques
deductivos; Castangia: arquitectura Conv1D-VAE + umbral media+3σ). Ninguno se usa con su
dataset ni su segmentación original (multi-cliente / por-ciclo): solo como menú de diseño.

## Estado — pipeline completo (v1), extremo a extremo verificado con datos reales

- [x] Módulo 1 — `src/data_loading.py`: carga IHEPC, interpolación lineal de huecos (~1.25%).
- [x] Módulo 2 — `src/splitting.py`: split cronológico train(años 1-2)/val(año 3)/test(año 4),
      mismo criterio que `../replica_massidda_marrocu/`.
- [x] Módulo 3 — `src/windowing.py`: ventaneo parametrizable (1440/720/360 min), solape
      proporcional (1/4 de la ventana) en train, disjuntas en val/test.
- [x] Módulo 4 — `src/normalization.py`: z-score ajustado solo en train.
- [x] Módulo 5 — `src/models/tcn_ae.py`: TCN-AE (encoder dilatado + AvgPool bottleneck +
      decoder simétrico), 23.513 parámetros. Checkpoints entrenados para las 3 ventanas en
      `models/tcn_ae_ventana{360,720,1440}.pt`.
- [x] Módulo 6 — `src/anomaly_score.py`: score por ventana, mae/mse/**deficit**/deficit_max.
- [x] Módulo 7 — `src/thresholding.py`: umbral percentil/media+3σ (solo train) + histograma
      en `results/figures/hist_scores_train_val.png`.
- [x] Módulo 8 — `src/attacks.py`: 5 familias de ataque deductivo, testeadas.
- [x] Módulo 9 — `src/evaluation.py`: episodios independientes por (familia, parámetro,
      duración), tabla en `results/tables/tabla_dr_fa_por_ataque.csv`, curva en
      `results/figures/curva_dr_vs_rho.png`. `generar_episodios_scores` calcula el score una
      vez y `agregar_dr`/`barrer_umbrales_dr_fa` lo reutilizan para comparar umbrales sin
      recalcular nada.

Cada módulo es ejecutable de forma aislada: `python -m src.<modulo>` desde esta carpeta
(reutiliza el checkpoint entrenado vía `src/pipeline.py`, no reentrena cada vez).

## Default actual: ventana=360 min (6h), score=deficit, umbral=percentil 99

Tras comparar 1440/720/360 min con el mismo criterio de umbral (percentil 99 sobre train),
360 min gana con diferencia en las 5 familias de ataque (ver
`results/figures/comparativa_ventanas.png` y `results/tables/comparativa_ventanas.csv`):

| Familia | 1440 min (FA=0.00%) | 720 min (FA=0.14%) | **360 min (FA=1.52%, default)** |
|---|---|---|---|
| bypass_total | 28.6% | 52.8% | **76.7%** |
| bypass_residual | 8.3% | 25.0% | **50.0%** |
| reducción_variable | 2.4% | 2.8% | **23.3%** |
| reducción_constante | 0.7% | 3.2% | **7.1%** |
| recorte_picos | 0% | 0% | 0% |

Razón: un ataque corto se "diluye" menos dentro de una ventana de evaluación más corta. El
coste es más falsas alarmas (0.00% → 1.52%), asumible dada la mejora. `results/figures/
curva_roc_dr_fa.png` (barrido de percentiles 50-99.9, ventana=360) muestra que bajar a
percentil 95 sube el DR global de 17.9% a 31.2% por 6.4% de FA — alternativa válida si se
tolera más FA; ajustar `umbral.percentil` en `configs/base.yaml` según el caso de uso.

Los checkpoints y resultados de 720/1440 min se conservan (no se han borrado) para poder
reproducir la comparación o volver atrás sin reentrenar.

## Hallazgos clave

1. **El error de reconstrucción simétrico (MAE/MSE) da DR=0% estructural.** Una ventana
   atacada (escalada hacia abajo o aplanada) es una señal *más simple* de reconstruir que el
   consumo real, así que el error absoluto BAJA en vez de subir con la severidad del ataque.
   El score **`deficit`** (residuo con signo, media(x̂-x) — mismo principio que el déficit de
   energía D60 de la fase open-loop anterior del TFG) sí es monótono con la severidad y es el
   default desde entonces.
2. **`deficit_max`** (residuo con signo en el peor punto de la ventana, pensado para pillar
   mejor el recorte de picos) se probó a fondo y es **peor que `deficit`** en toda la curva
   DR-vs-FA, incluso específicamente para recorte de picos. Descartado; se deja implementado
   en `anomaly_score.py` como opción documentada, no como default.
3. **El recorte de picos no se detecta en NINGUNA combinación probada** (0% en las 3 ventanas,
   los 2 scores, todos los umbrales del barrido). Es la limitación real y honesta del enfoque
   actual (reconstrucción de ventana completa): afecta a muy pocos minutos, se diluye siempre.
   Candidatos para una fase futura: comparar forma/derivada en vez de magnitud agregada, o un
   detector específico de picos en paralelo.

## Pendiente / a revisar en la próxima sesión

- El TCN-AE agotó las 100 épocas configuradas sin que el early stopping (paciencia 10)
  llegara a cortar en ninguna de las 3 ventanas — la val_loss seguía bajando. Aumentar
  `epochs_max` probablemente mejora la reconstrucción base.
- `n_posiciones_por_duracion=6` da resultados algo ruidosos por tamaño de muestra pequeño
  (pasos de ~17% en vez de continuos); subirlo (p.ej. a 20-30) daría cifras más fiables para
  la memoria, a cambio de más tiempo de evaluación.
- Recorte de picos sigue sin resolverse (ver hallazgo 3).
- USAD y SCVAE (interfaz común ya lista en `src/models/base.py`) — fases posteriores.

## Estructura

```
deteccion_fraude_no_supervisada/
├── configs/base.yaml       # ventana, ρ, umbral, score, modelo, seed — única fuente de parámetros
├── data/{raw,processed}/
├── docs/papers/
├── models/                 # checkpoints entrenados (tcn_ae_ventana{360,720,1440}.pt)
├── src/
│   ├── data_loading.py
│   ├── splitting.py
│   ├── windowing.py
│   ├── normalization.py
│   ├── pipeline.py          # setup compartido (carga+split+ventaneo+norm+modelo cacheado)
│   ├── models/{base,tcn_ae}.py   # usad.py, scvae.py: fases posteriores
│   ├── anomaly_score.py
│   ├── thresholding.py
│   ├── attacks.py
│   └── evaluation.py
├── tests/
└── results/{tables,figures}/
```
