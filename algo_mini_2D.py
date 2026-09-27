"""Partie III.B.2 — Minimisation 2D par la méthode de Newton (Q17 à Q19).

Q17. Le plan tangent à z = g_x(x, y) en (x0, y0) a pour équation
     z = g_x(x0,y0) + ∂g_x/∂x·(x-x0) + ∂g_x/∂y·(y-y0), idem pour g_y.
     Le point (x1, y1, 0) appartient aux deux plans et au plan z = 0, donc
         -g_x(x0,y0) = ∂g_x/∂x·(x1-x0) + ∂g_x/∂y·(y1-y0)
         -g_y(x0,y0) = ∂g_y/∂x·(x1-x0) + ∂g_y/∂y·(y1-y0)
     soit (-g_x, -g_y)ᵀ = J(x0,y0)·(x1-x0, y1-y0)ᵀ avec
         J = [[∂²G/∂x², ∂²G/∂x∂y], [∂²G/∂y∂x, ∂²G/∂y²]]
     J est la jacobienne du gradient, c'est-à-dire la matrice hessienne de G.
"""

import numpy as np

H_GRADIENT = 1e-5
H_HESSIENNE = 1e-4


def _pas(v: float, h: float) -> float:
    return abs(v) * h if v != 0 else h


def grad(G, X: np.ndarray, h: float = H_GRADIENT) -> np.ndarray:
    """Q18. Gradient approché de G en X = (x, y), différences centrées (III.2)."""
    X = np.asarray(X, dtype=float)
    g = np.empty(2)
    for k in range(2):
        e = np.zeros(2)
        e[k] = _pas(X[k], h)
        g[k] = (G(X + e) - G(X - e)) / (2 * e[k])
    return g


def hessienne(G, X: np.ndarray, h: float = H_HESSIENNE) -> np.ndarray:
    """Matrice J de la question 17 (hessienne de G), par différences finies."""
    X = np.asarray(X, dtype=float)
    dx, dy = _pas(X[0], h), _pas(X[1], h)
    ex, ey = np.array([dx, 0.0]), np.array([0.0, dy])
    g0 = G(X)
    gxx = (G(X + ex) - 2 * g0 + G(X - ex)) / dx ** 2
    gyy = (G(X + ey) - 2 * g0 + G(X - ey)) / dy ** 2
    gxy = (G(X + ex + ey) - G(X + ex - ey) - G(X - ex + ey) + G(X - ex - ey)) / (4 * dx * dy)
    return np.array([[gxx, gxy], [gxy, gyy]])


def min_local_2D(G, X0: np.ndarray, h: float = H_GRADIENT, tol: float = 1e-7,
                 max_iter: int = 200, trajectoire: list = None) -> np.ndarray:
    """Q19. Minimum local de G par Newton : X_{n+1} = X_n - J⁻¹(X_n)·∇G(X_n).

    Arrêt quand |g_x| < tol et |g_y| < tol. Pour la robustesse :
    * si J n'est pas définie positive (ou pas inversible), le pas de Newton ne
      descend pas : on prend la direction opposée au gradient ;
    * le pas est divisé par 2 tant que G n'a pas diminué (recherche linéaire),
      ce qui évite les divergences loin du minimum ;
    * si G ne peut plus diminuer en double précision, on s'arrête : le
      critère |g| < tol, absolu, dépend des unités et peut être inatteignable.
    Si ``trajectoire`` est une liste, les itérés successifs y sont ajoutés.
    """
    X = np.asarray(X0, dtype=float)
    if trajectoire is not None:
        trajectoire.append(X.copy())
    for _ in range(max_iter):
        g = grad(G, X, h)
        if np.all(np.abs(g) < tol):
            return X
        J = hessienne(G, X, max(h, H_HESSIENNE))
        try:
            direction = -np.linalg.solve(J, g)
            if direction @ g >= 0:          # J non définie positive
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            direction = -g
        pas, g_actuel = 1.0, G(X)
        while G(X + pas * direction) >= g_actuel and pas > 1e-12:
            pas /= 2
        if pas <= 1e-12 or np.all(np.abs(pas * direction) <= 1e-13 * (1 + np.abs(X))):
            # Plus aucun progrès possible : G ne diminue plus en double précision.
            # Le gradient résiduel n'est alors que l'erreur des différences finies
            # (d²·G'''/6), qui peut dépasser tol si G est très raide.
            return X
        X = X + pas * direction
        if trajectoire is not None:
            trajectoire.append(X.copy())
    raise RuntimeError(f"min_local_2D n'a pas convergé en {max_iter} itérations")


def fct_dont_je_veux_le_minimum(X: np.ndarray) -> float:
    """Exemple du sujet : fonction de Rosenbrock, minimum en (1, 1)."""
    x, y = X
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


if __name__ == "__main__":
    print(grad(lambda X: (X[0] - 1) ** 2 + (X[1] - 2) ** 2, np.array([0.0, 0.0])))  # ≈ [-2, -4]
    print(min_local_2D(fct_dont_je_veux_le_minimum, np.array([-1.2, 1.0])))       # ≈ [1, 1]
