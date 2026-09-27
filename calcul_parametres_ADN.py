"""Partie III.A — Modèle du ver (WLC) et ajustement de ses paramètres (Q10, Q11)."""

import numpy as np
import scipy.optimize

from constantes import K_B, NM, PN, UM


def force(z: np.ndarray, Lp: float, L0: float, T: float) -> np.ndarray:
    """Q10. Force (N) du modèle du ver (formule III.1) pour chaque allongement z (m).

    F(z) = kB·T/Lp · ( 1/(4(1 - z/L0)²) - 1/4 + z/L0 )
    Le modèle n'a de sens que pour 0 <= z < L0 : la force diverge en z = L0.
    """
    x = np.asarray(z, dtype=float) / L0
    return K_B * T / Lp * (0.25 / (1 - x) ** 2 - 0.25 + x)


def extension_wlc(F, Lp: float, L0: float, T: float):
    """Inverse du modèle du ver : allongement z (m) produit par la force F (N).

    F(z) est strictement croissante sur [0, L0[, on résout F(z) = F par la
    méthode de Brent pour chaque valeur de F.
    """
    def une_extension(f):
        if f <= 0:
            return 0.0
        return L0 * scipy.optimize.brentq(
            lambda x: force(x * L0, Lp, L0, T) - f, 0.0, 1 - 1e-12, xtol=1e-14)

    F = np.asarray(F, dtype=float)
    return np.vectorize(une_extension)(F) if F.ndim else une_extension(float(F))


def ajusteWLC(Fz: np.ndarray, T: float, avec_incertitudes: bool = False):
    """Q11. Paramètres (Lp, L0) du modèle du ver ajustés aux mesures Fz.

    Fz : tableau (M, 2) ; colonne 0 = force (N), colonne 1 = allongement (m).

    Deux précautions rendent l'ajustement robuste :
    * on travaille en unités naturelles (pN, nm, µm) pour que les paramètres soient
      d'ordre 1 : en SI, Lp ≈ 5e-8 et la tolérance de curve_fit n'aurait aucun sens ;
    * on impose L0 > max(z) (bornes), sinon la force du modèle diverge ou change de
      signe et l'optimiseur s'égare.
    """
    Fz = np.asarray(Fz, dtype=float)
    F_pN, z_um = Fz[:, 0] / PN, Fz[:, 1] / UM

    def modele(z_um, Lp_nm, L0_um):
        return force(z_um * UM, Lp_nm * NM, L0_um * UM, T) / PN

    z_max = z_um.max()
    p0 = [50.0, 1.1 * z_max]  # longueur de persistance typique de l'ADN : 50 nm
    bornes = ([0.1, 1.0001 * z_max], [1e4, 100 * z_max])
    popt, pcov = scipy.optimize.curve_fit(modele, z_um, F_pN, p0=p0, bounds=bornes)
    Lp, L0 = popt[0] * NM, popt[1] * UM
    if avec_incertitudes:
        dLp, dL0 = np.sqrt(np.diag(pcov)) * [NM, UM]
        return Lp, L0, dLp, dL0
    return Lp, L0


def ecart_quadratique(Fz: np.ndarray, T: float):
    """E(Lp, L0) de la partie III.B.2, en unités naturelles (Lp en nm, L0 en µm,
    E en pN²) afin d'être minimisée par ``algo_mini_2D.min_local_2D``."""
    F_pN, z = Fz[:, 0] / PN, Fz[:, 1]

    def E(X):
        Lp_nm, L0_um = X
        if L0_um * UM <= z.max() or Lp_nm <= 0:
            return np.inf
        return float(np.sum((F_pN - force(z, Lp_nm * NM, L0_um * UM, T) / PN) ** 2))
    return E


if __name__ == "__main__":
    from constantes import LP_ADN, L0_LAMBDA, T_AMBIANTE
    rng = np.random.default_rng(1)
    F = np.geomspace(0.05, 10, 15) * PN
    z = extension_wlc(F, LP_ADN, L0_LAMBDA, T_AMBIANTE) * (1 + rng.normal(0, 0.01, F.size))
    Lp, L0, dLp, dL0 = ajusteWLC(np.column_stack([F, z]), T_AMBIANTE, True)
    print(f"Lp = {Lp / NM:.1f} ± {dLp / NM:.1f} nm, L0 = {L0 / UM:.2f} ± {dL0 / UM:.2f} µm")
