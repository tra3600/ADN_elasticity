"""Tests des fonctions du sujet (exemples de l'énoncé + vérifications physiques)."""

import math

import numpy as np
import pytest

from constantes import K_B, LP_ADN, L0_LAMBDA, T_AMBIANTE, PN, UM, NM
from Statistiques import moyenne, variance, somme
from mesures_exp import seuillage, pixel_centre_bille, centre_bille, positions, \
    fluctuations, force_depuis_fluctuations
from allongement_ADN import profil, profil_naif, bibliotheque_calibration, cote_bille
from calcul_parametres_ADN import force, extension_wlc, ajusteWLC, ecart_quadratique
from algo_min_local import derive, derive_seconde, min_local
from algo_mini_2D import grad, hessienne, min_local_2D, fct_dont_je_veux_le_minimum
from modelisation_plane import conformation, allongement, nouvelle_conformation
from MMC import selection_conformation, monte_carlo, simulation, allongement_theorique, \
    variance_equilibre, longueur_monomere_equivalente
import experience_virtuelle as ev


# --- Partie I ---------------------------------------------------------------
def test_statistiques_exemples_du_sujet():
    X = [1, 2, 3, 4]
    assert moyenne(X) == 2.5
    assert variance(X) == 1.25
    assert X == [1, 2, 3, 4]
    assert somme([[[1, 2], [3, 4, 5]], 6, [7, 8], 9]) == 45
    assert somme(np.array([[1.5, 2.5], [3.0, 1.0]])) == 8.0


# --- Partie II --------------------------------------------------------------
def test_seuillage_et_centre():
    A = np.array([[157, 200, 100], [50, 180, 90], [120, 250, 60]])
    S = seuillage(A, 150)
    assert S.tolist() == [[0, 0, 1], [1, 0, 1], [1, 0, 1]]
    assert pixel_centre_bille(S) == (1, 1)  # barycentre (1.2, 1.2) -> pixel (1, 1)


def test_pixel_centre_arrondit_au_plus_proche():
    B = np.zeros((5, 5), dtype=int)
    B[4, 4] = B[3, 4] = B[4, 3] = 1  # barycentre (3.67, 3.67) ; int() donnerait (3, 3)
    assert pixel_centre_bille(B) == (4, 4)


def test_localisation_sous_pixel_sur_image_simulee():
    rng = np.random.default_rng(0)
    vrai = (97.3, 91.8)
    image = ev.image_diffraction(vrai, 6 * UM, rng=rng)
    i, j = centre_bille(seuillage(image, 100))
    assert abs(i - vrai[0]) < 0.3 and abs(j - vrai[1]) < 0.3


def test_fluctuations():
    P = [(0, 0), (2, 0), (0, 2), (2, 2)]  # barycentre (1, 1), distance² = 2 partout
    assert fluctuations(P, 0.5) == pytest.approx(2 * 0.25)


def test_profil_vectorise_egal_profil_naif():
    rng = np.random.default_rng(1)
    A = (rng.random((41, 37)) < 0.3).astype(int)
    assert np.allclose(profil(A, 6), profil_naif(A, 6))
    p = profil(A, 6)
    assert len(p) == 6 and np.all((0 <= p) & (p <= 1))


def test_calibration_retrouve_la_cote():
    cotes = np.arange(0, 17.01, 0.2) * UM
    biblio = bibliotheque_calibration(cotes, repetitions=2)
    rng = np.random.default_rng(2)
    for z in (2.3, 7.7, 13.1):
        A = seuillage(ev.image_diffraction((95.6, 96.2), z * UM, rng=rng), 100)
        assert cote_bille(profil(A, 40, 44.0), cotes, biblio) == pytest.approx(z * UM, abs=0.15 * UM)


def test_force_depuis_fluctuations_sur_experience_virtuelle():
    pince = ev.configurer(1 * PN, graine=3)
    P = positions(400, 100, sous_pixel=True)
    F = force_depuis_fluctuations(fluctuations(P, ev.PIXEL), pince.z_moyen, T_AMBIANTE)
    assert F == pytest.approx(1 * PN, rel=0.15)


# --- Partie III -------------------------------------------------------------
def test_force_wlc_et_inverse():
    z = np.array([0.0, 8 * UM, 15 * UM])
    F = force(z, LP_ADN, L0_LAMBDA, T_AMBIANTE)
    assert F[0] == 0 and np.all(np.diff(F) > 0)
    assert extension_wlc(F, LP_ADN, L0_LAMBDA, T_AMBIANTE) == pytest.approx(z, abs=1e-12)


def test_ajuste_wlc_retrouve_les_parametres():
    F = np.geomspace(0.05, 10, 12) * PN
    z = extension_wlc(F, LP_ADN, L0_LAMBDA, T_AMBIANTE)
    Lp, L0 = ajusteWLC(np.column_stack([F, z]), T_AMBIANTE)
    assert Lp == pytest.approx(LP_ADN, rel=1e-4)
    assert L0 == pytest.approx(L0_LAMBDA, rel=1e-5)


def test_derivees_et_min_local():
    assert derive(lambda x: x ** 3, 2.0, 1e-5) == pytest.approx(12, rel=1e-8)
    assert derive(lambda x: x ** 2, 0.0, 1e-5) == pytest.approx(0, abs=1e-10)  # x = 0 géré
    assert derive_seconde(math.sin, 1.0, 1e-4) == pytest.approx(-math.sin(1.0), rel=1e-5)
    assert min_local(lambda x: (x - 2) ** 2, 0.0, 1e-5) == pytest.approx(2, abs=1e-6)
    assert min_local(math.cos, 3.0, 1e-5) == pytest.approx(math.pi, abs=1e-6)


def test_min_local_2D():
    assert grad(lambda X: (X[0] - 1) ** 2 + (X[1] - 2) ** 2, np.zeros(2)) == \
        pytest.approx([-2, -4], abs=1e-6)
    H = hessienne(lambda X: X[0] ** 2 + 3 * X[0] * X[1] + 5 * X[1] ** 2, np.array([1.0, 2.0]))
    assert H == pytest.approx(np.array([[2, 3], [3, 10]]), rel=1e-4)
    assert min_local_2D(fct_dont_je_veux_le_minimum, np.array([-1.2, 1.0])) == \
        pytest.approx([1, 1], abs=1e-5)


def test_newton_2D_sur_ecart_quadratique_concorde_avec_curve_fit():
    rng = np.random.default_rng(4)
    F = np.geomspace(0.05, 10, 12) * PN
    z = extension_wlc(F, LP_ADN, L0_LAMBDA, T_AMBIANTE)
    Fz = np.column_stack([F * (1 + rng.normal(0, 0.03, F.size)), z])
    Lp, L0 = ajusteWLC(Fz, T_AMBIANTE)
    X = min_local_2D(ecart_quadratique(Fz, T_AMBIANTE), np.array([90.0, 17.5]))
    assert X[0] == pytest.approx(Lp / NM, rel=1e-3)
    assert X[1] == pytest.approx(L0 / UM, rel=1e-4)


# --- Partie IV --------------------------------------------------------------
def test_conformations():
    rng = np.random.default_rng(5)
    theta = conformation(50, rng)
    assert theta.shape == (50,) and np.all((-np.pi <= theta) & (theta < np.pi))
    assert allongement(np.zeros(7), 2.0) == 14.0
    nouveau = nouvelle_conformation(theta, 4, rng)
    differences = np.flatnonzero(nouveau != theta)
    assert len(differences) == 4 and np.all(np.diff(differences) == 1)
    assert nouvelle_conformation(theta, 50, rng).shape == (50,)


def test_metropolis():
    l, T, F = 1e-7, 300.0, 1e-12
    etire, replie = np.zeros(10), np.full(10, np.pi / 2)
    assert selection_conformation(replie, etire, F, l, T) is etire  # énergie plus basse
    # Hausse d'énergie énorme -> rejet (probabilité exp(-240) ≈ 0)
    assert selection_conformation(etire, replie, F, l, T) is etire
    # Force nulle : toutes les conformations sont équiprobables -> acceptation
    assert selection_conformation(etire, replie, 0.0, l, T) is replie


def test_monte_carlo_converge_vers_la_theorie():
    n, l, T = 40, longueur_monomere_equivalente(LP_ADN), T_AMBIANTE
    F = 0.2 * PN
    eps = 1.5 * variance_equilibre(F, n, l, T)
    _, historique, _ = simulation(F, n, l, T, 1, eps, rng=np.random.default_rng(6),
                                  iterations_min=400 * n)
    assert historique[100 * n:].mean() == pytest.approx(allongement_theorique(F, n, l, T), rel=0.05)
    assert monte_carlo(F, n, l, T, 2, 10 * eps) > 0


def test_monte_carlo_signale_epsilon_irrealiste():
    with pytest.raises(RuntimeError):
        simulation(1e-13, 20, 1e-7, 300.0, 2, epsilon=0.0, max_iter=2000)
