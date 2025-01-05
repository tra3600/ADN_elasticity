import numpy as np

def conformation(n: int) -> np.ndarray:
    """
    Génère une conformation aléatoire d'un brin d'ADN composé de n segments.
    
    Paramètres:
    n (int): Nombre de segments.
    
    Retourne:
    np.ndarray: Vecteur de longueur n correspondant à l'orientation (angle θ) de chaque segment.
    """
    return np.random.uniform(-np.pi, np.pi, n)

# Exemple d'utilisation
n = 10
theta = conformation(n)
print(theta)


def allongement(theta: np.ndarray, l: float) -> float:
    """
    Calcule l'allongement z de la chaîne dans la conformation theta pour une longueur de segment l.
    
    Paramètres:
    theta (np.ndarray): Vecteur des angles θ.
    l (float): Longueur de chaque segment.
    
    Retourne:
    float: Allongement total z de la chaîne.
    """
    # Calcule la somme des cosinus des angles multipliée par la longueur du segment
    z = l * np.sum(np.cos(theta))
    return z

# Exemple d'utilisation
l = 1.0
z = allongement(theta, l)
print(z)


def nouvelle_conformation(theta: np.ndarray, k: int) -> np.ndarray:
    """
    Crée une nouvelle conformation en modifiant k valeurs successives à partir d'un indice aléatoire.
    
    Paramètres:
    theta (np.ndarray): Conformation initiale (vecteur des angles θ).
    k (int): Nombre de valeurs successives à modifier.
    
    Retourne:
    np.ndarray: Nouvelle conformation modifiée.
    """
    n = len(theta)
    new_theta = theta.copy()
    
    # Sélectionner un indice aléatoire de départ
    start_index = np.random.randint(0, n - k + 1)
    
    # Modifier k valeurs successives à partir de l'indice de départ
    new_theta[start_index:start_index + k] = np.random.uniform(-np.pi, np.pi, k)
    
    return new_theta

# Exemple d'utilisation
k = 3
new_theta = nouvelle_conformation(theta, k)
print(new_theta)


