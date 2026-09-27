"""Constantes physiques et paramètres de référence partagés par tous les modules.

Toutes les grandeurs sont exprimées dans le Système International (m, N, K, J).
"""

K_B = 1.380649e-23  # Constante de Boltzmann (J/K)
T_AMBIANTE = 298.0  # Température de travail par défaut (K)

# Valeurs de référence pour l'ADN du phage lambda (48 502 paires de bases),
# la molécule utilisée dans les expériences de pinces magnétiques de l'ENS.
LP_ADN = 50e-9       # Longueur de persistance (m)
L0_LAMBDA = 16.4e-6  # Longueur de contour (extension maximale) (m)
KUHN_ADN = 2 * LP_ADN  # Longueur d'un monomère de la chaîne librement jointe équivalente (m)

PN = 1e-12  # 1 piconewton en newtons
UM = 1e-6   # 1 micromètre en mètres
NM = 1e-9   # 1 nanomètre en mètres
