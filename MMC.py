"""Partie IV.B et IV.C — Metropolis Monte Carlo sur la chaîne librement jointe (Q23, Q24)."""

from collections import deque

import numpy as np
from scipy.special import ive

from constantes import K_B
from modelisation_plane import conformation, allongement, nouvelle_conformation

_rng = np.random.default_rng()

FENETRE = 500  # nombre de dernières conformations utilisées (énoncé IV.C)


def energie(theta, F: float, l: float) -> float:
    """Énergie mécanique E = -z·F (formule IV.1)."""
    return -allongement(theta, l) * F


def selection_conformation(thetaA, thetaB, F: float, l: float, T: float, rng=None):
    """Q23. Critère de Metropolis : thetaB est gardée si elle abaisse l'énergie,
    sinon avec la probabilité P = exp((E1 - E2)/(kB·T)) ; sinon on garde thetaA."""
    E1, E2 = energie(thetaA, F, l), energie(thetaB, F, l)
    if E2 < E1:
        return thetaB
    return thetaB if (rng or _rng).random() < np.exp((E1 - E2) / (K_B * T)) else thetaA


def simulation(F: float, n: int, l: float, T: float, k: int, epsilon: float,
               max_iter: int = 1_000_000, rng=None, theta0=None,
               iterations_min: int = 0):
    """Moteur commun : renvoie (allongement moyen, historique des z, conformation finale).

    Structure de données : une file (``deque`` de taille bornée à 500) contient
    les allongements des 500 dernières conformations. L'ajout à droite éjecte
    automatiquement l'élément le plus ancien en O(1), contrairement à
    ``list.pop(0)`` qui coûte O(500). La mémoire reste constante quel que soit
    le nombre d'itérations.

    Remarque physique : à l'équilibre, z fluctue encore (agitation thermique) ;
    le critère « variance < epsilon » n'est donc atteignable que si epsilon
    dépasse la variance d'équilibre (voir ``variance_equilibre``). D'où le
    garde-fou ``max_iter``.

    Piège : à forte force, presque tous les essais sont rejetés ; la chaîne se
    fige loin de l'équilibre, la variance sur 500 itérations devient minuscule
    et le critère se déclenche trop tôt (allongement sous-estimé). Le paramètre
    ``iterations_min`` impose une période de rodage avant de tester la
    convergence (0 = comportement strict de l'énoncé).
    """
    rng = rng or _rng
    theta = conformation(n, rng) if theta0 is None else np.array(theta0, dtype=float)
    derniers = deque(maxlen=FENETRE)
    historique = []
    for iteration in range(max_iter):
        theta = selection_conformation(theta, nouvelle_conformation(theta, k, rng), F, l, T, rng)
        z = allongement(theta, l)
        derniers.append(z)
        historique.append(z)
        if (iteration >= iterations_min and len(derniers) == FENETRE
                and np.var(derniers) < epsilon):
            break
    else:
        raise RuntimeError(f"pas de convergence en {max_iter} itérations "
                           f"(epsilon trop petit devant la variance thermique ?)")
    return float(np.mean(derniers)), np.array(historique), theta


def monte_carlo(F: float, n: int, l: float, T: float, k: int, epsilon: float) -> float:
    """Q24. Allongement moyen des 500 dernières conformations, une fois la
    variance de l'allongement sur ces 500 conformations inférieure à epsilon."""
    return simulation(F, n, l, T, k, epsilon)[0]


# --- Solution exacte pour valider la simulation ----------------------------
# Dans le plan, les segments sont indépendants et chacun suit la loi de
# Boltzmann p(θ) ∝ exp(x·cos θ) avec x = F·l/(kB·T). Les fonctions de Bessel
# modifiées donnent <cos θ> = I1(x)/I0(x) et <cos² θ> = (1 + I2(x)/I0(x))/2.

def allongement_theorique(F, n: int, l: float, T: float):
    """Allongement moyen exact de la chaîne librement jointe plane."""
    x = np.asarray(F, dtype=float) * l / (K_B * T)
    return n * l * ive(1, x) / ive(0, x)  # ive = I·exp(-x) : pas de débordement


def longueur_monomere_equivalente(Lp: float) -> float:
    """Longueur l des monomères de la chaîne plane qui reproduit le modèle du ver
    aux faibles forces.

    Faible force : la chaîne plane donne ⟨cos θ⟩ ≈ x/2, soit F ≈ 2·kB·T·z/(L0·l) ;
    le modèle du ver (III.1) donne F ≈ 3·kB·T·z/(2·Lp·L0). L'égalité impose
    l = 4·Lp/3 (et non la longueur de Kuhn 2·Lp, valable pour une chaîne 3D).
    """
    return 4 * Lp / 3


def variance_equilibre(F, n: int, l: float, T: float):
    """Variance exacte de z à l'équilibre (somme des variances des n segments)."""
    x = np.asarray(F, dtype=float) * l / (K_B * T)
    r1, r2 = ive(1, x) / ive(0, x), ive(2, x) / ive(0, x)
    return n * l ** 2 * ((1 + r2) / 2 - r1 ** 2)


if __name__ == "__main__":
    from constantes import LP_ADN, T_AMBIANTE, PN
    n, l, T, F = 50, longueur_monomere_equivalente(LP_ADN), T_AMBIANTE, 0.1 * PN
    eps = 1.2 * variance_equilibre(F, n, l, T)
    print("Monte Carlo :", monte_carlo(F, n, l, T, k=3, epsilon=eps))
    print("Théorie     :", allongement_theorique(F, n, l, T))
