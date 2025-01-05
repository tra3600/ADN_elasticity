import numpy as np

K_B = 1.380649e-23  # Constante de Boltzmann en J/K

def energie(theta: np.ndarray, F: float, l: float) -> float:
    """
    Calcule l'énergie d'une conformation.
    
    Paramètres:
    theta (np.ndarray): Conformation (vecteur des angles θ).
    F (float): Intensité de la force de traction.
    l (float): Longueur de chaque monomère.
    
    Retourne:
    float: Énergie de la conformation.
    """
    z = allongement(theta, l)
    return -z * F

def selection_conformation(thetaA: np.ndarray, thetaB: np.ndarray, F: float, l: float, T: float) -> np.ndarray:
    """
    Sélectionne la conformation à conserver selon le critère de Metropolis.
    
    Paramètres:
    thetaA (np.ndarray): Conformation initiale (précurseur).
    thetaB (np.ndarray): Nouvelle conformation.
    F (float): Intensité de la force de traction.
    l (float): Longueur de chaque monomère.
    T (float): Température en Kelvin.
    
    Retourne:
    np.ndarray: Conformation conservée.
    """
    E1 = energie(thetaA, F, l)
    E2 = energie(thetaB, F, l)
    
    if E2 < E1:
        return thetaB
    else:
        P = np.exp((E1 - E2) / (K_B * T))
        if np.random.random() < P:
            return thetaB
        else:
            return thetaA

# Exemple d'utilisation
thetaA = conformation(10)
thetaB = nouvelle_conformation(thetaA, 3)
F = 1.0
l = 1.0
T = 300
selected_theta = selection_conformation(thetaA, thetaB, F, l, T)
print(selected_theta)


from collections import deque

def monte_carlo(F: float, n: int, l: float, T: float, k: int, epsilon: float) -> float:
    """
    Simule l'application d'une force de traction sur un brin d'ADN et retourne l'allongement moyen des 500 dernières conformations.
    
    Paramètres:
    F (float): Intensité de la force de traction.
    n (int): Nombre de monomères.
    l (float): Longueur de chaque monomère.
    T (float): Température en Kelvin.
    k (int): Nombre de valeurs successives à modifier.
    epsilon (float): Critère de convergence (variance de l'allongement).
    
    Retourne:
    float: Allongement moyen des 500 dernières conformations après convergence.
    """
    # Initialisation
    theta = conformation(n)
    allongements = deque(maxlen=500)
    
    while True:
        # Créer une nouvelle conformation
        new_theta = nouvelle_conformation(theta, k)
        
        # Sélectionner la conformation à conserver
        theta = selection_conformation(theta, new_theta, F, l, T)
        
        # Calculer l'allongement et ajouter à la file
        z = allongement(theta, l)
        allongements.append(z)
        
        # Vérifier la convergence
        if len(allongements) == 500 and np.var(allongements) < epsilon:
            break
    
    # Calculer l'allongement moyen des 500 dernières conformations
    return np.mean(allongements)

# Exemple d'utilisation
F = 1.0
n = 10
l = 1.0
T = 300
k = 3
epsilon = 1e-5
mean_allongement = monte_carlo(F, n, l, T, k, epsilon)
print(mean_allongement)