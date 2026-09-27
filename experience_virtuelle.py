"""Banc d'essai virtuel : une pince magnétique simulée qui remplace le matériel.

Le sujet suppose l'existence d'une fonction ``prendre_photo()`` pilotant une
caméra CCD réelle. Ce module la simule de manière physiquement plausible pour
que toute la chaîne de mesure (parties II et III) puisse être exécutée :

* le brin d'ADN suit le modèle du ver (WLC) avec des paramètres « cachés »
  (Lp, L0) que l'analyse devra retrouver ;
* la bille subit un mouvement brownien : assimilée à un pendule inversé de
  longueur z tiré par la force F, sa raideur transverse vaut F/z, donc
  <δx²> = <δy²> = kB·T·z/F (équipartition) ;
* l'image de diffraction est un motif d'anneaux dont la période grandit avec
  la défocalisation, c'est-à-dire avec la cote z de la bille ;
* le capteur ajoute un bruit gaussien et quantifie sur 8 bits.
"""

import numpy as np

from constantes import K_B, T_AMBIANTE, LP_ADN, L0_LAMBDA, NM, UM

TAILLE_IMAGE = 192      # côté du capteur (pixels)
PIXEL = 40 * NM         # côté t d'un pixel ramené dans le plan objet (m)
FOND = 150              # niveau de gris du fond
AMPLITUDE = 110         # contraste des anneaux
RAYON_ENVELOPPE = 45.0  # étendue du motif de diffraction (pixels)
BRUIT = 6.0             # écart-type du bruit du capteur (niveaux de gris)


def periode_anneaux(z: float) -> float:
    """Période radiale (pixels) des anneaux pour une bille à la cote z (m).

    Loi de calibration (fictive mais monotone, comme en pratique) : plus la bille
    s'éloigne du plan focal, plus les anneaux s'élargissent.
    """
    return 5.0 + 0.5 * (z / UM)


def image_diffraction(centre, z: float, taille: int = TAILLE_IMAGE,
                      bruit: float = BRUIT, rng=None) -> np.ndarray:
    """Image 8 bits d'une bille centrée en ``centre`` = (ligne, colonne) (flottants).

    Intensité : I(r) = FOND - AMPLITUDE·exp(-(r/R)²)·cos(2πr/P(z)) + bruit.
    Le centre est sombre, comme sur la figure 2 du sujet.
    """
    rng = np.random.default_rng() if rng is None else rng
    i, j = np.indices((taille, taille), dtype=float)
    r = np.hypot(i - centre[0], j - centre[1])
    intensite = FOND - AMPLITUDE * np.exp(-(r / RAYON_ENVELOPPE) ** 2) \
        * np.cos(2 * np.pi * r / periode_anneaux(z))
    if bruit > 0:
        intensite = intensite + rng.normal(0.0, bruit, intensite.shape)
    return np.clip(np.rint(intensite), 0, 255).astype(np.uint8)


class PinceMagnetique:
    """Expérience de pince magnétique simulée à force F constante.

    Paramètres
    ----------
    F : force de traction (N)
    Lp, L0 : paramètres « vrais » (cachés) du brin d'ADN (m)
    T : température (K)
    t : taille d'un pixel dans le plan objet (m)
    """

    def __init__(self, F: float, Lp: float = LP_ADN, L0: float = L0_LAMBDA,
                 T: float = T_AMBIANTE, t: float = PIXEL, taille: int = TAILLE_IMAGE,
                 bruit: float = BRUIT, graine=None):
        # Import local pour éviter une dépendance circulaire au chargement
        from calcul_parametres_ADN import extension_wlc, force

        self.F, self.Lp, self.L0, self.T, self.t = F, Lp, L0, T, t
        self.taille, self.bruit = taille, bruit
        self.rng = np.random.default_rng(graine)
        self.z_moyen = float(extension_wlc(F, Lp, L0, T))
        # Fluctuations transverses : raideur F/z  ->  <δx²> = kB T z / F
        self.sigma_xy = np.sqrt(K_B * T * self.z_moyen / F)
        # Fluctuations longitudinales : raideur dF/dz (dérivée du modèle du ver)
        dz = 1e-4 * L0
        raideur_z = (force(np.array([self.z_moyen + dz]), Lp, L0, T)[0]
                     - force(np.array([self.z_moyen - dz]), Lp, L0, T)[0]) / (2 * dz)
        self.sigma_z = np.sqrt(K_B * T / raideur_z)
        self.centre_image = (taille - 1) / 2

    def prendre_photo(self) -> np.ndarray:
        """Déclenche un cliché : tire une position brownienne puis forme l'image."""
        dx, dy = self.rng.normal(0.0, self.sigma_xy, 2) / self.t
        z = self.z_moyen + self.rng.normal(0.0, self.sigma_z)
        centre = (self.centre_image + dx, self.centre_image + dy)
        return image_diffraction(centre, z, self.taille, self.bruit, self.rng)


# Expérience courante utilisée par la fonction ``prendre_photo`` du sujet
_experience = None


def configurer(F: float, **parametres) -> PinceMagnetique:
    """Installe une nouvelle expérience (force F) comme source des clichés."""
    global _experience
    _experience = PinceMagnetique(F, **parametres)
    return _experience


def prendre_photo() -> np.ndarray:
    """Fonction fournie par le sujet : renvoie un cliché de la caméra CCD."""
    if _experience is None:
        configurer(1e-12)  # expérience par défaut : 1 pN
    return _experience.prendre_photo()
