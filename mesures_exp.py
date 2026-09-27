"""Partie II.A — Position de la bille (Q4 à Q7).

Les clichés proviennent de ``experience_virtuelle.prendre_photo`` qui simule la
caméra CCD (voir ce module pour la physique du mouvement brownien).
"""

import numpy as np

from constantes import K_B
import experience_virtuelle


def seuillage(A: np.ndarray, seuil: int) -> np.ndarray:
    """Q4. 1 là où le pixel est strictement inférieur au seuil, 0 ailleurs."""
    return (np.asarray(A) < seuil).astype(np.uint8)


def centre_bille(A: np.ndarray) -> (float, float):
    """Barycentre (ligne, colonne) des pixels à 1, en flottants (précision sous-pixel)."""
    indices = np.argwhere(A == 1)
    if len(indices) == 0:
        raise ValueError("aucun pixel blanc : seuil mal choisi ?")
    i, j = indices.mean(axis=0)
    return float(i), float(j)


def pixel_centre_bille(A: np.ndarray) -> (int, int):
    """Q5. Indices du pixel le plus proche du barycentre des pixels à 1.

    On arrondit (``round``) et non pas on tronque (``int``) : c'est bien le pixel
    *le plus proche* du barycentre qui est demandé.
    """
    i, j = centre_bille(A)
    return round(i), round(j)


def positions(n: int, seuil: int, sous_pixel: bool = False,
              photo=None) -> [(int, int)]:
    """Q6. Prend n clichés et renvoie la liste des positions successives de la bille.

    ``sous_pixel=True`` renvoie les barycentres flottants (plus précis) ;
    ``photo`` permet d'injecter une autre source de clichés (tests, caméra réelle).
    """
    photo = experience_virtuelle.prendre_photo if photo is None else photo
    localiser = centre_bille if sous_pixel else pixel_centre_bille
    return [localiser(seuillage(photo(), seuil)) for _ in range(n)]


def fluctuations(P: [(int, int)], t: float) -> float:
    """Q7. Moyenne des carrés des distances à la position d'équilibre (m²).

    La position d'équilibre est le barycentre des positions observées ; t est la
    longueur correspondant à un pixel.
    """
    P = np.asarray(P, dtype=float)
    ecarts = P - P.mean(axis=0)
    return float(np.mean(np.sum(ecarts ** 2, axis=1))) * t ** 2


def force_depuis_fluctuations(fluct: float, z: float, T: float) -> float:
    """Synthèse II.C : force de traction déduite des fluctuations browniennes.

    La bille se comporte comme un pendule de longueur z et de raideur transverse
    F/z. L'équipartition donne kB·T/2 par direction, soit dans le plan (xOy) :
    <δr²> = <δx²> + <δy²> = 2·kB·T·z/F   donc   F = 2·kB·T·z/<δr²>.
    La force est bien inversement proportionnelle aux fluctuations (énoncé II).
    """
    return 2 * K_B * T * z / fluct


if __name__ == "__main__":
    A = np.array([[157, 200, 100], [50, 180, 90], [120, 250, 60]])
    print(seuillage(A, 150))
    print(pixel_centre_bille(seuillage(A, 150)))
    P = positions(20, seuil=100)
    print(P[:5])
    print(fluctuations(P, experience_virtuelle.PIXEL), "m²")
