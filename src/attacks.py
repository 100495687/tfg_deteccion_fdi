"""Las 5 familias de ataque (infrarreporte deductivo). Cada funcion recibe un tramo real `x`
y devuelve la version atacada y = h(x), mismo shape. La inyeccion en el stream de test (donde
y como) vive en evaluation.py -- aqui solo estan las transformaciones puras.

  - reduccion_constante(x, rho): y = rho * x
  - reduccion_variable(x, rho_min, rho_max, rng): y_t = beta_t * x_t, con beta_t aleatorio en [rho_min, rho_max]
  - bypass_total(x): y = 0
  - bypass_residual(x, rho): igual que reduccion_constante pero con rho bajo (0.05-0.10);
    se separa como familia aparte para poder desglosarla en la tabla de resultados
  - recorte_picos(x, umbral): y = min(x, umbral), solo toca lo que supera el umbral

Todas son deductivas: nunca aumentan la lectura ni bajan de cero (se comprueba en el __main__).

Se puede correr suelto:
    python -m src.attacks
"""
from __future__ import annotations

import numpy as np


def reduccion_constante(x: np.ndarray, rho: float) -> np.ndarray:
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho debe estar en [0,1], recibido {rho}")
    return rho * x


def reduccion_variable(x: np.ndarray, rho_min: float, rho_max: float, rng: np.random.Generator) -> np.ndarray:
    if not (0.0 <= rho_min < rho_max <= 1.0):
        raise ValueError(f"rango invalido: rho_min={rho_min}, rho_max={rho_max}")
    beta_t = rng.uniform(rho_min, rho_max, size=len(x))
    return beta_t * x


def bypass_total(x: np.ndarray) -> np.ndarray:
    return np.zeros_like(x)


def bypass_residual(x: np.ndarray, rho: float) -> np.ndarray:
    return reduccion_constante(x, rho)


def recorte_picos(x: np.ndarray, umbral: float) -> np.ndarray:
    return np.minimum(x, umbral)


FAMILIAS = {
    "reduccion_constante": reduccion_constante,
    "reduccion_variable": reduccion_variable,
    "bypass_total": bypass_total,
    "bypass_residual": bypass_residual,
    "recorte_picos": recorte_picos,
}


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    x = np.array([1.0, 2.0, 3.0, 4.0, 0.5, 5.0], dtype=np.float32)

    y = reduccion_constante(x, rho=0.3)
    print(f"reduccion_constante(rho=0.3): {y}")
    assert np.allclose(y, x * 0.3)

    y = reduccion_variable(x, 0.1, 0.7, rng)
    print(f"reduccion_variable(0.1, 0.7): {y}")
    assert (y <= x * 0.7 + 1e-6).all() and (y >= x * 0.1 - 1e-6).all()

    y = bypass_total(x)
    print(f"bypass_total: {y}")
    assert (y == 0).all()

    y = bypass_residual(x, rho=0.08)
    print(f"bypass_residual(rho=0.08): {y}")
    assert np.allclose(y, x * 0.08)

    umbral = 2.5
    y = recorte_picos(x, umbral)
    print(f"recorte_picos(umbral=2.5): {y}")
    assert (y <= umbral + 1e-6).all()
    assert np.array_equal(y[x <= umbral], x[x <= umbral])  # no toca lo que ya esta por debajo del umbral

    # invariante del modelo de amenaza (fraude deductivo): ninguna familia puede aumentar
    # la lectura reportada ni producir valores negativos
    llamadas = {
        "reduccion_constante": lambda: reduccion_constante(x, 0.3),
        "reduccion_variable": lambda: reduccion_variable(x, 0.1, 0.7, rng),
        "bypass_total": lambda: bypass_total(x),
        "bypass_residual": lambda: bypass_residual(x, 0.08),
        "recorte_picos": lambda: recorte_picos(x, umbral),
    }
    for nombre, llamar in llamadas.items():
        y = llamar()
        assert (y <= x + 1e-6).all(), f"{nombre} aumenta la lectura reportada (viola fraude deductivo)"
        assert (y >= -1e-6).all(), f"{nombre} produce lecturas negativas"
    assert set(llamadas) == set(FAMILIAS)

    print("\nLas 5 familias cumplen el modelo de amenaza: deductivas y sin valores negativos.")
