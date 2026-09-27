"""Partie IV.A — Chaîne librement jointe dans le plan (Q20 à Q22)."""

import numpy as np

_rng = np.random.default_rng()


def conformation(n: int, rng=None) -> np.ndarray:
    """Q20. Conformation aléatoire : n angles θ_i tirés uniformément dans [-π, π[."""
    return (rng or _rng).uniform(-np.pi, np.pi, n)


def allongement(theta, l: float) -> float:
    """Q21. Allongement z = l·Σ cos θ_i : projection de la chaîne sur l'axe de F."""
    return l * float(np.sum(np.cos(theta)))


def nouvelle_conformation(theta, k: int, rng=None) -> np.ndarray:
    """Q22. Copie de theta où k angles successifs, à partir d'un indice aléatoire,
    sont retirés au hasard. theta n'est pas modifiée."""
    rng = rng or _rng
    n = len(theta)
    k = min(k, n)
    debut = rng.integers(0, n - k + 1)  # k angles successifs doivent tenir dans la chaîne
    nouveau = np.array(theta, dtype=float)
    nouveau[debut:debut + k] = rng.uniform(-np.pi, np.pi, k)
    return nouveau


def coordonnees(theta, l: float) -> np.ndarray:
    """Positions (z, y) des n+1 jointures, la première à l'origine (pour le tracé).
    L'axe z est la direction de la force, comme sur la figure 5 du sujet."""
    pas = l * np.column_stack([np.cos(theta), np.sin(theta)])
    return np.vstack([[0.0, 0.0], np.cumsum(pas, axis=0)])


if __name__ == "__main__":
    theta = conformation(10)
    print(theta)
    print(allongement(theta, 1.0))
    print(nouvelle_conformation(theta, 3))
