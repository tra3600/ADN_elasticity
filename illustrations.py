"""Génère toutes les figures du projet dans le dossier ``figures/``.

    python illustrations.py            # toutes les figures (~1 min)
    python illustrations.py --rapide   # moins de clichés et d'itérations

Scénario : une « expérience virtuelle » de pince magnétique est réalisée sur un
ADN de phage lambda dont les paramètres (Lp = 50 nm, L0 = 16,4 µm) sont cachés
dans le simulateur. L'analyse d'images (partie II) mesure force et allongement,
les méthodes de la partie III retrouvent Lp et L0, et la chaîne librement
jointe simulée par Monte Carlo (partie IV) est confrontée aux mêmes mesures.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from constantes import T_AMBIANTE, LP_ADN, L0_LAMBDA, PN, UM, NM
import experience_virtuelle as ev
from mesures_exp import seuillage, centre_bille, pixel_centre_bille, fluctuations, \
    force_depuis_fluctuations
from allongement_ADN import profil, bibliotheque_calibration, cote_bille, \
    N_ANNEAUX, RAYON_PROFIL, SEUIL
from calcul_parametres_ADN import force, extension_wlc, ajusteWLC, ecart_quadratique
from algo_mini_2D import min_local_2D
from modelisation_plane import conformation, coordonnees, allongement
from MMC import simulation, allongement_theorique, variance_equilibre, selection_conformation, \
    longueur_monomere_equivalente
from modelisation_plane import nouvelle_conformation

DOSSIER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# Palette catégorielle validée (daltonisme) et encres neutres
BLEU, ORANGE, VERT = "#2a78d6", "#eb6834", "#1baf7a"
ENCRE, ENCRE_2, ENCRE_3, GRILLE, FOND = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": FOND, "axes.facecolor": FOND, "savefig.facecolor": FOND,
    "axes.edgecolor": ENCRE_3, "axes.labelcolor": ENCRE_2, "text.color": ENCRE,
    "xtick.color": ENCRE_2, "ytick.color": ENCRE_2, "axes.grid": True,
    "grid.color": GRILLE, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2,
    "legend.frameon": False, "figure.dpi": 110,
})


def sauver(fig, nom):
    chemin = os.path.join(DOSSIER, nom)
    fig.savefig(chemin, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print("  ->", os.path.relpath(chemin))


# ---------------------------------------------------------------------------
def figure_mesure(cotes, biblio):
    """Figure 1 : de l'image CCD au profil de diffraction et à la cote z."""
    rng = np.random.default_rng(3)
    c = (ev.TAILLE_IMAGE - 1) / 2
    z_ex = 8 * UM
    image = ev.image_diffraction((c + 4.3, c - 6.8), z_ex, rng=rng)
    A = seuillage(image, SEUIL)
    ci, cj = pixel_centre_bille(A)

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.9), gridspec_kw={"width_ratios": [1, 1, 1.3, 1.1]})
    ax[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("① Cliché CCD simulé")
    ax[1].imshow(A, cmap="gray_r")
    largeur = RAYON_PROFIL / N_ANNEAUX
    for k in range(0, N_ANNEAUX + 1, 8):
        ax[1].add_patch(plt.Circle((cj, ci), k * largeur, fill=False, color=ORANGE, lw=1))
    ax[1].plot(cj, ci, "+", color=BLEU, ms=14, mew=2.5)
    ax[1].set_title("② Seuillage, barycentre, anneaux")
    for a in ax[:2]:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)

    rayons = (np.arange(N_ANNEAUX) + 0.5) * largeur
    for z, couleur in zip((3, 9, 15), (BLEU, ORANGE, VERT)):
        m = int(np.argmin(abs(cotes - z * UM)))
        ax[2].plot(rayons, biblio[m], color=couleur, label=f"z = {z} µm")
    ax[2].set_xlabel("rayon de l'anneau (pixels)")
    ax[2].set_ylabel("proportion de pixels blancs")
    ax[2].set_title("③ Profils de calibration")
    ax[2].legend(loc="upper right")

    ax[3].imshow(biblio, aspect="auto", cmap="Blues", origin="lower",
                 extent=[0, RAYON_PROFIL, cotes[0] / UM, cotes[-1] / UM])
    z_mesure = cote_bille(profil(A, N_ANNEAUX, RAYON_PROFIL), cotes, biblio)
    ax[3].axhline(z_mesure / UM, color=ORANGE, lw=2)
    ax[3].text(1, z_mesure / UM + 0.4, f"mesure : {z_mesure / UM:.2f} µm (vrai 8)",
               color=ENCRE, fontsize=9, bbox=dict(fc=FOND, ec="none", alpha=0.85))
    ax[3].set_xlabel("rayon (pixels)"); ax[3].set_ylabel("cote z (µm)")
    ax[3].set_title("④ Bibliothèque → cote z"); ax[3].grid(False)
    fig.suptitle("Partie II — Localiser la bille en 3D à partir de sa figure de diffraction",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    sauver(fig, "1_mesure_bille.png")


# ---------------------------------------------------------------------------
def experience(forces, cotes, biblio, n_photos):
    """Réalise l'expérience virtuelle pour chaque force : renvoie les mesures."""
    resultats = []
    for F in forces:
        pince = ev.configurer(F, graine=int(F / PN * 1000))
        P, Z = [], []
        for _ in range(n_photos):
            A = seuillage(pince.prendre_photo(), SEUIL)
            P.append(centre_bille(A))
            Z.append(cote_bille(profil(A, N_ANNEAUX, RAYON_PROFIL), cotes, biblio))
        z = float(np.mean(Z))
        F_mes = force_depuis_fluctuations(fluctuations(P, ev.PIXEL), z, T_AMBIANTE)
        resultats.append(dict(F_vrai=F, z_vrai=pince.z_moyen, F=F_mes, z=z,
                              dz=np.std(Z) / np.sqrt(n_photos),
                              dF=F_mes / np.sqrt(n_photos), P=np.array(P)))
        print(f"  F = {F / PN:6.3f} pN : mesuré {F_mes / PN:6.3f} pN, "
              f"z = {z / UM:6.3f} µm (vrai {pince.z_moyen / UM:6.3f})")
    return resultats


def figure_brownien(res):
    """Figure 2 : mouvement brownien de la bille et force déduite des fluctuations."""
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})
    choix = [res[1], res[len(res) // 2], res[-1]]
    for r, couleur in zip(choix, (BLEU, ORANGE, VERT)):
        P = (r["P"] - r["P"].mean(axis=0)) * ev.PIXEL / NM
        ax[0].scatter(P[:, 1], P[:, 0], s=9, color=couleur, alpha=0.55, lw=0,
                      label=f"F = {r['F_vrai'] / PN:.2g} pN")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x (nm)"); ax[0].set_ylabel("y (nm)")
    ax[0].set_title("Positions de la bille (barycentres)")
    ax[0].legend(loc="upper right", markerscale=2)

    Fv = np.array([r["F_vrai"] for r in res]) / PN
    Fm = np.array([r["F"] for r in res]) / PN
    dF = np.array([r["dF"] for r in res]) / PN
    ax[1].plot([Fv.min() / 1.5, Fv.max() * 1.5], [Fv.min() / 1.5, Fv.max() * 1.5],
               color=ENCRE_3, lw=1.2, ls="--", label="mesure parfaite")
    ax[1].errorbar(Fv, Fm, yerr=dF, fmt="o", ms=7, color=BLEU, mec=FOND, mew=1.5,
                   capsize=3, label="F = 2·kB·T·z / ⟨δr²⟩")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("force imposée par les aimants (pN)")
    ax[1].set_ylabel("force mesurée (pN)")
    ax[1].set_title("Force déduite des fluctuations browniennes")
    ax[1].legend(loc="upper left")
    fig.suptitle("Plus on tire, moins la bille tremble : ⟨δr²⟩ ∝ z / F",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    sauver(fig, "2_mouvement_brownien.png")


def figure_newton(Fz):
    """Figure 4 : paysage de l'écart quadratique E(Lp, L0) et itérés de Newton."""
    E = ecart_quadratique(Fz, T_AMBIANTE)
    trajet = []
    X = min_local_2D(E, np.array([95.0, 17.6]), trajectoire=trajet)
    trajet = np.array(trajet)
    Lp = np.linspace(20, 100, 220)
    L0 = np.linspace(Fz[:, 1].max() / UM + 0.02, 17.8, 220)
    grille = np.array([[E((a, b)) for a in Lp] for b in L0])

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    niveaux = np.geomspace(np.nanmin(grille[np.isfinite(grille)]) * 1.05,
                           np.nanmax(grille[np.isfinite(grille)]), 18)
    cs = ax.contourf(Lp, L0, grille, levels=niveaux, cmap="Blues_r",
                     norm=matplotlib.colors.LogNorm())
    ax.contour(Lp, L0, grille, levels=niveaux, colors=FOND, linewidths=0.5)
    fig.colorbar(cs, ax=ax, label="E(Lp, L0)  (pN²)")
    ax.plot(trajet[:, 0], trajet[:, 1], "-o", color=ORANGE, ms=7, mec=FOND, mew=1.5,
            label=f"itérés de Newton ({len(trajet) - 1} pas)")
    ax.plot(LP_ADN / NM, L0_LAMBDA / UM, "*", color=ENCRE, ms=15, mec=FOND,
            label="valeurs cachées du simulateur")
    ax.annotate(f"Lp = {X[0]:.1f} nm\nL0 = {X[1]:.2f} µm", X, xytext=(X[0] + 12, X[1] + 0.45),
                color=ENCRE, arrowprops=dict(arrowstyle="->", color=ENCRE_2),
                bbox=dict(fc=FOND, ec="none", alpha=0.9))
    ax.set_xlabel("longueur de persistance Lp (nm)")
    ax.set_ylabel("longueur de contour L0 (µm)")
    ax.set_title("Partie III.B — Newton 2D sur l'écart quadratique", fontsize=12)
    ax.legend(loc="upper right"); ax.grid(False)
    fig.tight_layout()
    sauver(fig, "4_newton_2D.png")
    return X


def figure_force_extension(res, Lp, L0, X_newton, mmc, n, l):
    """Figure 3 : la courbe force-extension et les trois modèles."""
    z = np.array([r["z"] for r in res]) / UM
    F = np.array([r["F"] for r in res]) / PN
    dF = np.array([r["dF"] for r in res]) / PN
    dz = np.array([r["dz"] for r in res]) / UM
    zz = np.linspace(0.5, 0.995 * L0 / UM, 400)

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.plot(zz, force(zz * UM, LP_ADN, L0_LAMBDA, T_AMBIANTE) / PN, color=ENCRE_3,
            lw=1.3, ls=":", label="modèle caché du simulateur")
    ax.plot(zz, force(zz * UM, Lp, L0, T_AMBIANTE) / PN, color=BLEU,
            label=f"ver (WLC) ajusté : Lp = {Lp / NM:.1f} nm, L0 = {L0 / UM:.2f} µm")
    Fg = np.geomspace(0.01, 30, 300) * PN
    ax.plot(allongement_theorique(Fg, n, l, T_AMBIANTE) / UM, Fg / PN,
            color=ORANGE, ls="--", lw=1.6,
            label=f"chaîne librement jointe 2D exacte ({n} × {l / NM:.0f} nm, l = 4·Lp/3)")
    ax.plot(mmc[:, 1] / UM, mmc[:, 0] / PN, "D", color=ORANGE, ms=7, mec=FOND, mew=1.5,
            label="chaîne librement jointe 2D — Metropolis Monte Carlo")
    ax.errorbar(z, F, xerr=dz, yerr=dF, fmt="o", color=ENCRE, ms=6, mec=FOND, mew=1.2,
                capsize=2.5, label="mesures (expérience virtuelle)")
    ax.set_yscale("log")
    ax.set_ylim(0.01, 30); ax.set_xlim(0, 17.2)
    ax.set_xlabel("allongement z (µm)")
    ax.set_ylabel("force F (pN)")
    ax.set_title("Élasticité de l'ADN λ : force en fonction de l'allongement", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.text(0.99, 0.03, f"Newton 2D (III.B) : Lp = {X_newton[0]:.1f} nm, L0 = {X_newton[1]:.2f} µm\n"
            f"curve_fit (III.A)  : Lp = {Lp / NM:.1f} nm, L0 = {L0 / UM:.2f} µm\n"
            f"valeurs cachées   : Lp = 50.0 nm, L0 = 16.40 µm",
            transform=ax.transAxes, ha="right", va="bottom", family="monospace", fontsize=8.5,
            color=ENCRE_2, bbox=dict(fc=FOND, ec=GRILLE))
    fig.tight_layout()
    sauver(fig, "3_force_extension.png")


def monte_carlo_forces(forces, n, l, rodage, production):
    """Allongement moyen par Monte Carlo : rodage, puis moyenne sur une longue phase
    de production (z est corrélé sur ~n itérations, 500 états ne suffisent pas)."""
    points = []
    for F in forces:
        eps = 1.3 * variance_equilibre(F, n, l, T_AMBIANTE)
        # La moyenne sur 500 itérations (énoncé) est bruitée à faible force car z
        # y est corrélé sur ~n itérations : on moyenne toute la phase après rodage.
        _, historique, _ = simulation(F, n, l, T_AMBIANTE, 1, eps, rng=np.random.default_rng(7),
                                      iterations_min=rodage + production)
        z = historique[rodage:].mean()
        points.append((F, z))
        print(f"  MMC F = {F / PN:6.3f} pN : z = {z / UM:6.3f} µm "
              f"(théorie {allongement_theorique(F, n, l, T_AMBIANTE) / UM:6.3f})")
    return np.array(points)


def figure_chaine(n, l, iterations_min, rodage_trace):
    """Figure 5 : conformations d'équilibre et convergence de Metropolis."""
    rng = np.random.default_rng(11)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    decalage = 0
    for F, couleur in zip((0.02, 0.3, 5.0), (BLEU, ORANGE, VERT)):
        F *= PN
        eps = 1.3 * variance_equilibre(F, n, l, T_AMBIANTE)
        _, _, theta = simulation(F, n, l, T_AMBIANTE, 1, eps, rng=rng, iterations_min=iterations_min)
        xy = coordonnees(theta, l) / UM
        ax[0].plot(xy[:, 0], xy[:, 1] + decalage, color=couleur, lw=1.6,
                   label=f"F = {F / PN:g} pN, z = {allongement(theta, l) / UM:.1f} µm")
        ax[0].plot(xy[-1, 0], xy[-1, 1] + decalage, "o", color=couleur, ms=9, mec=FOND, mew=1.5)
        decalage -= 4
    ax[0].annotate("", xy=(17.5, -6), xytext=(13.5, -6),
                   arrowprops=dict(arrowstyle="-|>", color=ENCRE, lw=2))
    ax[0].text(15.5, -5.5, "F", ha="center", fontweight="bold")
    ax[0].set_aspect("equal"); ax[0].set_xlabel("z (µm), direction de la force")
    ax[0].set_yticks([]); ax[0].set_title(f"Conformations d'équilibre ({n} monomères de {l / NM:.0f} nm)")
    ax[0].legend(loc="upper center", fontsize=9, bbox_to_anchor=(0.5, -0.16), ncol=3)

    F = 0.3 * PN
    eps = 1.3 * variance_equilibre(F, n, l, T_AMBIANTE)
    moyenne, historique, _ = simulation(F, n, l, T_AMBIANTE, 1, eps, rng=rng,
                                        iterations_min=rodage_trace)
    it = np.arange(1, len(historique) + 1)
    ax[1].axvspan(0, rodage_trace, color=GRILLE, alpha=0.6, lw=0)
    ax[1].text(rodage_trace / 2, 7, "rodage", ha="center", color=ENCRE_2)
    ax[1].plot(it, historique / UM, color=BLEU, lw=1, label="allongement z instantané")
    ax[1].axhline(allongement_theorique(F, n, l, T_AMBIANTE) / UM, color=ORANGE, ls="--",
                  lw=1.6, label="⟨z⟩ exact (Bessel I₁/I₀)")
    ax[1].axvspan(len(historique) - 500, len(historique), color=VERT, alpha=0.25, lw=0,
                  label=f"500 dernières : ⟨z⟩ = {moyenne / UM:.2f} µm")
    ax[1].set_xlabel("itération de Metropolis"); ax[1].set_ylabel("z (µm)")
    ax[1].set_title("Convergence de la simulation (F = 0,3 pN)")
    ax[1].legend(loc="lower right", fontsize=9)
    fig.suptitle("Partie IV — Chaîne librement jointe et Metropolis Monte Carlo",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    sauver(fig, "5_chaine_monte_carlo.png")


def animation_chaine(n, l, images=90, pas_par_image=1500):
    """Animation : la chaîne se déplie quand la force augmente progressivement."""
    rng = np.random.default_rng(5)
    forces = np.geomspace(0.005, 20, images) * PN
    theta = conformation(n, rng)
    etats = []
    for F in forces:
        for _ in range(pas_par_image):
            theta = selection_conformation(theta, nouvelle_conformation(theta, 1, rng),
                                           F, l, T_AMBIANTE, rng)
        etats.append((F, theta.copy()))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ligne, = ax.plot([], [], color=BLEU, lw=1.8)
    bille, = ax.plot([], [], "o", color=ORANGE, ms=14, mec=FOND, mew=2)
    ax.plot(0, 0, "s", color=ENCRE, ms=10)
    texte = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top", family="monospace")
    ax.set_xlim(-6, 17.5); ax.set_ylim(-6, 6); ax.set_aspect("equal")
    ax.set_xlabel("z (µm)"); ax.set_yticks([])
    ax.set_title("Metropolis Monte Carlo : l'ADN se déplie sous la traction")

    def dessiner(k):
        F, th = etats[k]
        xy = coordonnees(th, l) / UM
        ligne.set_data(xy[:, 0], xy[:, 1])
        bille.set_data([xy[-1, 0]], [xy[-1, 1]])
        texte.set_text(f"F = {F / PN:7.3f} pN\nz = {allongement(th, l) / UM:6.2f} µm")
        return ligne, bille, texte

    anim = FuncAnimation(fig, dessiner, frames=len(etats), blit=True)
    chemin = os.path.join(DOSSIER, "chaine_se_deplie.gif")
    anim.save(chemin, writer=PillowWriter(fps=12), dpi=80)
    plt.close(fig)
    print("  ->", os.path.relpath(chemin))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rapide", action="store_true", help="version allégée (tests)")
    args = parser.parse_args()
    os.makedirs(DOSSIER, exist_ok=True)
    n_photos = 60 if args.rapide else 300
    # Chaîne plane équivalente à l'ADN caché (pour les illustrations de la partie IV)
    l_eq = longueur_monomere_equivalente(LP_ADN)
    n_eq = int(round(L0_LAMBDA / l_eq))

    print("Calibration de la caméra…")
    cotes = np.arange(0, 17.01, 0.1) * UM
    biblio = bibliotheque_calibration(cotes, repetitions=3 if args.rapide else 8)
    figure_mesure(cotes, biblio)

    print("Expérience virtuelle (pince magnétique)…")
    forces = np.geomspace(0.03, 10, 6 if args.rapide else 12) * PN
    res = experience(forces, cotes, biblio, n_photos)
    figure_brownien(res)

    Fz = np.array([(r["F"], r["z"]) for r in res])
    Lp, L0 = ajusteWLC(Fz, T_AMBIANTE)
    print(f"curve_fit : Lp = {Lp / NM:.1f} nm, L0 = {L0 / UM:.3f} µm")
    X = figure_newton(Fz)
    print(f"Newton 2D : Lp = {X[0]:.1f} nm, L0 = {X[1]:.3f} µm")

    print("Monte Carlo (chaîne librement jointe)…")
    # Chaîne plane construite à partir des paramètres *mesurés* Lp et L0
    l = longueur_monomere_equivalente(Lp)
    n = int(round(L0 / l))
    rodage = (60 if args.rapide else 300) * n
    mmc = monte_carlo_forces(np.geomspace(0.02, 10, 4 if args.rapide else 9) * PN,
                             n, l, rodage, 2 * rodage)
    figure_force_extension(res, Lp, L0, X, mmc, n, l)
    figure_chaine(n_eq, l_eq, (60 if args.rapide else 300) * n_eq, 40 * n_eq)
    if not args.rapide:
        animation_chaine(n_eq, l_eq)


if __name__ == "__main__":
    main()
