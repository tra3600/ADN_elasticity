import numpy as np

def grad(G, X: np.ndarray, h: float) -> np.ndarray:
    """
    Calcule une approximation du gradient de G au point X en utilisant h pour le calcul des dérivées.
    
    Paramètres:
    G (callable): Fonction réelle de deux variables réelles.
    X (np.ndarray): Point (x, y) où calculer le gradient.
    h (float): Pas pour le calcul approché des dérivées.
    
    Retourne:
    np.ndarray: Gradient approximé de G au point X.
    """
    x, y = X
    dG_dx = (G(np.array([x + h, y])) - G(np.array([x - h, y]))) / (2 * h)
    dG_dy = (G(np.array([x, y + h])) - G(np.array([x, y - h]))) / (2 * h)
    return np.array([dG_dx, dG_dy])

# Exemple d'utilisation
def fct_dont_je_veux_le_minimum(X):
    x, y = X
    return (x - 1)**2 + (y - 2)**2

X = np.array([0.0, 0.0])
h = 1e-5
print(grad(fct_dont_je_veux_le_minimum, X, h))  # Devrait afficher une valeur proche de [-2, -4]


def min_local_2D(G, X0: np.ndarray, h: float) -> np.ndarray:
    """
    Trouve un minimum local de G en partant du point X0 en utilisant la méthode de Newton.
    
    Paramètres:
    G (callable): Fonction réelle de deux variables réelles.
    X0 (np.ndarray): Point initial (x0, y0).
    h (float): Pas pour le calcul approché des dérivées.
    
    Retourne:
    np.ndarray: Valeurs approximées de (xm, ym) correspondant au minimum local de G.
    """
    X = X0
    while True:
        grad_G = grad(G, X, h)
        if np.linalg.norm(grad_G) < 1e-7:
            break
        
        # Calcul de la matrice Jacobienne
        x, y = X
        J = np.array([
            [(G(np.array([x + h, y])) - G(np.array([x - h, y]))) / (2 * h),
             (G(np.array([x, y + h])) - G(np.array([x, y - h]))) / (2 * h)]
        ]).reshape(2, 2)
        
        try:
            # Calcul de l'inverse de la matrice Jacobienne
            J_inv = np.linalg.inv(J)
        except np.linalg.LinAlgError:
            print("La matrice Jacobienne n'est pas inversible.")
            break

        # Mise à jour de X en utilisant la méthode de Newton
        X = X - J_inv @ grad_G
    
    return X

# Exemple d'utilisation
X0 = np.array([0.0, 0.0])
h = 1e-5
print(min_local_2D(fct_dont_je_veux_le_minimum, X0, h))  # Devrait afficher une valeur proche de [1, 2]

