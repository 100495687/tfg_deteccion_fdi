"""Score de anomalia: error de reconstruccion por ventana, con varias formas de calcularlo
(score.tipo en la config). Funciona con cualquier objeto que tenga un metodo reconstruir(X).

  - mae / mse: error simetrico. Falla frente a fraude deductivo -- una ventana atacada
    (escalada hacia abajo) es mas facil de reconstruir, asi que el error baja en vez de
    subir con el ataque (DR=0% en las 5 familias, comprobado en evaluation.py).
  - deficit: residuo con signo, media(xhat - x). Si sube con la severidad del ataque
    (mismo principio que el deficit de energia de la fase open-loop anterior del TFG).
    Puede ser negativo en ventanas limpias, el umbral se calibra igual sobre esa distribucion.
  - deficit_max: como deficit pero solo el peor instante de la ventana, para no diluir
    ataques que afectan a un tramo muy corto (p.ej. recorte de picos).

Se puede correr suelto:
    python -m src.anomaly_score
"""
from __future__ import annotations

import numpy as np


def calcular_scores(modelo, X: np.ndarray, tipo: str = "deficit") -> np.ndarray:
    """Score de reconstruccion por ventana. `tipo`: 'mae', 'mse' o 'deficit'."""
    X_hat = modelo.reconstruir(X)
    if tipo == "mae":
        return np.mean(np.abs(X - X_hat), axis=1)
    if tipo == "mse":
        return np.mean((X - X_hat) ** 2, axis=1)
    if tipo == "deficit":
        return np.mean(X_hat - X, axis=1)
    if tipo == "deficit_max":
        return np.max(X_hat - X, axis=1)
    raise ValueError(f"tipo de score no reconocido: {tipo}")


if __name__ == "__main__":
    from src.pipeline import obtener_modelo, preparar_datos
    from src.windowing import cargar_config

    config = cargar_config()
    datos = preparar_datos(config)
    modelo = obtener_modelo(config, datos["ventanas_norm"])

    for tipo in ("mae", "mse", "deficit", "deficit_max"):
        scores = {
            nombre: calcular_scores(modelo, X, tipo)
            for nombre, X in datos["ventanas_norm"].items()
        }
        print(f"[{tipo}] " + " | ".join(
            f"{nombre}: media={s.mean():.4f} std={s.std():.4f}" for nombre, s in scores.items()
        ))
        for nombre, s in scores.items():
            assert np.isfinite(s).all(), f"score no finito en {nombre}"
            if tipo in ("mae", "mse"):
                assert (s >= 0).all(), f"score negativo en {nombre} (tipo={tipo})"

    print("\nScores calculados para train/val/test sin problemas.")
