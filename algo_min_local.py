"""Partie III.B.1 — Minimisation 1D par la méthode de Newton (Q12 à Q16).

Q12. Mantisse de 52 bits (+1 bit implicite) : 2^-53 ≈ 1,1e-16, soit environ
     53·log10(2) ≈ 15,95 : 15 à 16 chiffres significatifs décimaux.
Q13. L'erreur de la formule (III.2) est la somme d'une erreur de méthode en
     O(h²) et d'une erreur d'arrondi en O(ε/h) (ε ≈ 1e-16) :
     - h = 1 : erreur de méthode énorme (et x(1-h) = 0 !) ;
     - h = 1e-16 : x(1±h) == x en flottant, le numérateur vaut 0 (ou n'a plus
       aucun chiffre significatif).
     Le compromis h² ≈ ε/h donne h ≈ ε^(1/3) ≈ 1e-5 pour la dérivée première.
     Pour la dérivée seconde, l'erreur d'arrondi est en O(ε/h²) : h ≈ ε^(1/4) ≈ 1e-4.
"""

H_DERIVEE = 1e-5
H_DERIVEE_SECONDE = 1e-4


def _pas(x: float, h: float) -> float:
    """Pas absolu du schéma relatif x(1±h). En x = 0 le pas relatif est nul :
    on utilise alors h lui-même comme pas absolu (sinon division par zéro)."""
    return abs(x) * h if x != 0 else h


def derive(phi, x: float, h: float = H_DERIVEE) -> float:
    """Q14. Dérivée approchée de phi en x, formule centrée (III.2) d'ordre 2."""
    d = _pas(x, h)
    return (phi(x + d) - phi(x - d)) / (2 * d)


def derive_seconde(phi, x: float, h: float = H_DERIVEE_SECONDE) -> float:
    """Q15. Dérivée seconde approchée : (phi(x+d) - 2phi(x) + phi(x-d)) / d²."""
    d = _pas(x, h)
    return (phi(x + d) - 2 * phi(x) + phi(x - d)) / d ** 2


def min_local(phi, x0: float, h: float = H_DERIVEE, tol: float = 1e-7,
              max_iter: int = 100) -> float:
    """Q16. Abscisse d'un minimum local par Newton appliqué à phi' :
    x_{k+1} = x_k - phi'(x_k)/phi''(x_k), jusqu'à |phi'(x)| < tol.

    Garde-fous : nombre d'itérations borné et refus d'un pas si phi'' <= 0
    (Newton convergerait alors vers un maximum) ; on fait alors un pas de
    descente de gradient.
    """
    x = x0
    for _ in range(max_iter):
        d1 = derive(phi, x, h)
        if abs(d1) < tol:
            return x
        d2 = derive_seconde(phi, x, max(h, H_DERIVEE_SECONDE))
        x = x - d1 / d2 if d2 > 0 else x - d1
    raise RuntimeError(f"min_local n'a pas convergé en {max_iter} itérations")


if __name__ == "__main__":
    import math
    print(derive(lambda x: x ** 2, 1.0, 1e-5))              # ≈ 2
    print(derive_seconde(lambda x: x ** 2, 1.0, 1e-4))      # ≈ 2
    print(min_local(lambda x: (x - 2) ** 2, 0.0, 1e-5))     # ≈ 2 (départ en 0 !)
    print(min_local(math.cos, 3.0, 1e-5), math.pi)          # ≈ π
