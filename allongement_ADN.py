"""Partie II.B — Allongement du brin d'ADN : profil de diffraction (Q8, Q9).

La cote z de la bille se lit dans l'espacement des anneaux de diffraction. On
calcule le profil radial de l'image seuillée, puis on le compare à une
bibliothèque de profils de calibration enregistrés à des cotes connues (c'est
ainsi que procèdent les vraies pinces magnétiques).

Q9 — Complexité : ``profil`` visite chaque pixel une seule fois et fait un
travail en O(1) pour chacun (calcul d'une distance, d'un indice d'anneau, deux
incrémentations), puis une passe en O(n) pour les proportions : O(p² + n).
"""

import numpy as np

from mesures_exp import seuillage, pixel_centre_bille
import experience_virtuelle


def indice_anneau(distance: np.ndarray, largeur: float) -> np.ndarray:
    """Numéro de l'anneau (de largeur ``largeur``) contenant un point à ``distance``
    du centre ; l'anneau 0 est le disque central."""
    return np.floor(distance / largeur).astype(int)


def profil(A: np.ndarray, n: int, rayon_max: float = None) -> np.ndarray:
    """Q8. Proportion de pixels blancs dans chacun des n anneaux concentriques.

    Les anneaux sont centrés sur la bille (``pixel_centre_bille``) et ont tous la
    même largeur ``rayon_max / n``. Par défaut, ``rayon_max`` est la distance du
    centre au bord le plus proche, pour que chaque anneau soit complet.
    Un pixel appartient à l'anneau qui contient son centre.
    """
    ci, cj = pixel_centre_bille(A)
    if rayon_max is None:
        rayon_max = min(ci, cj, A.shape[0] - 1 - ci, A.shape[1] - 1 - cj)
    i, j = np.indices(A.shape)
    k = indice_anneau(np.hypot(i - ci, j - cj), rayon_max / n).ravel()
    dedans = k < n
    # Compte des pixels (total et blancs) par anneau, en une seule passe
    total = np.bincount(k[dedans], minlength=n)
    blancs = np.bincount(k[dedans], weights=A.ravel()[dedans], minlength=n)
    return np.divide(blancs, total, out=np.zeros(n), where=total > 0)


def profil_naif(A: np.ndarray, n: int, rayon_max: float = None) -> list:
    """Même calcul que ``profil`` écrit avec des boucles explicites (style concours)."""
    ci, cj = pixel_centre_bille(A)
    if rayon_max is None:
        rayon_max = min(ci, cj, A.shape[0] - 1 - ci, A.shape[1] - 1 - cj)
    largeur = rayon_max / n
    total, blancs = [0] * n, [0] * n
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            k = int(((i - ci) ** 2 + (j - cj) ** 2) ** 0.5 // largeur)
            if k < n:
                total[k] += 1
                blancs[k] += int(A[i, j])
    return [b / t if t else 0.0 for b, t in zip(blancs, total)]


# --- Calibration : du profil à la cote z -----------------------------------

N_ANNEAUX = 40
RAYON_PROFIL = 44.0
SEUIL = 100


def bibliotheque_calibration(cotes: np.ndarray, n: int = N_ANNEAUX,
                             rayon_max: float = RAYON_PROFIL, seuil: int = SEUIL,
                             repetitions: int = 8, graine: int = 0) -> np.ndarray:
    """Profils moyens enregistrés pour des cotes connues (bille fixée, platine
    piézo-électrique déplacée pas à pas). Renvoie un tableau (len(cotes), n)."""
    rng = np.random.default_rng(graine)
    c = (experience_virtuelle.TAILLE_IMAGE - 1) / 2
    biblio = np.empty((len(cotes), n))
    for m, z in enumerate(cotes):
        # Petits décalages sous-pixel aléatoires pour moyenner l'effet de grille
        profils = [profil(seuillage(experience_virtuelle.image_diffraction(
            c + rng.uniform(-0.5, 0.5, 2), z, rng=rng), seuil), n, rayon_max)
            for _ in range(repetitions)]
        biblio[m] = np.mean(profils, axis=0)
    return biblio


def cote_bille(prof: np.ndarray, cotes: np.ndarray, biblio: np.ndarray) -> float:
    """Cote z dont le profil de calibration ressemble le plus à ``prof``.

    On cherche le minimum de l'écart quadratique puis on l'affine par une
    interpolation parabolique sur les trois points voisins.
    """
    ecarts = np.sum((biblio - prof) ** 2, axis=1)
    m = int(np.argmin(ecarts))
    if 0 < m < len(cotes) - 1:
        e0, e1, e2 = ecarts[m - 1], ecarts[m], ecarts[m + 1]
        courbure = e0 - 2 * e1 + e2
        if courbure > 0:
            decalage = 0.5 * (e0 - e2) / courbure
            return float(cotes[m] + decalage * (cotes[m + 1] - cotes[m]))
    return float(cotes[m])


if __name__ == "__main__":
    A = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 1, 1],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
    ])
    print(profil(A, 2))
    print(profil_naif(A, 2))
