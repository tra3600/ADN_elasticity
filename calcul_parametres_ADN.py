import numpy as np

# La constante de Boltzmann en unités de J/K
K_B = 1.380649e-23

def force(z: np.ndarray, Lp: float, L0: float, T: float) -> np.ndarray:
    """
    Calcule la force de traction pour chaque élément du vecteur z selon la formule (III.1).
    
    Paramètres:
    z (np.ndarray): Vecteur des allongements.
    Lp (float): Longueur de persistance du polymère.
    L0 (float): Extension maximale du polymère.
    T (float): Température en Kelvin.
    
    Retourne:
    np.ndarray: Vecteur des forces correspondantes.
    """
    term1 = 1 / (4 * (1 - z / L0) ** 2)
    term2 = -1 / 4
    term3 = z / L0
    force = (K_B * T / Lp) * (term1 + term2 + term3)
    return force

# Exemple d'utilisation
z = np.array([0.1, 0.2, 0.3])
Lp = 0.5
L0 = 1.0
T = 300
print(force(z, Lp, L0, T))

import scipy.optimize

def ajusteWLC(Fz: np.ndarray, T: float) -> (float, float):
    """
    Ajuste les paramètres de la formule (III.1) pour les valeurs expérimentales de Fz.
    
    Paramètres:
    Fz (np.ndarray): Tableau 2D des valeurs expérimentales, avec la force en première colonne et l'allongement en deuxième colonne.
    T (float): Température en Kelvin.
    
    Retourne:
    (float, float): Valeurs optimales de Lp et L0.
    """
    # Séparer les données expérimentales
    F_exp = Fz[:, 0]
    z_exp = Fz[:, 1]
    
    # Fonction modèle pour curve_fit
    def model(z, Lp, L0):
        return force(z, Lp, L0, T)

    # Ajuster les paramètres en utilisant curve_fit
    popt, pcov = scipy.optimize.curve_fit(model, z_exp, F_exp, p0=[1.0, 1.0])
    
    # Récupérer les valeurs optimales de Lp et L0
    Lp_opt, L0_opt = popt
    
    return Lp_opt, L0_opt

# Exemple d'utilisation
Fz = np.array([
    [0.12, 0.1],
    [0.15, 0.2],
    [0.18, 0.3],
    [0.22, 0.4],
    [0.28, 0.5]
])
T = 300
print(ajusteWLC(Fz, T))