def derive(phi, x: float, h: float) -> float:
    """
    Calcule une valeur approchée de la dérivée de phi en x.
    
    Paramètres:
    phi (callable): Fonction réelle d'une variable réelle.
    x (float): Point où calculer la dérivée.
    h (float): Pas de la dérivée.
    
    Retourne:
    float: Valeur approchée de la dérivée.
    """
    return (phi(x * (1 + h)) - phi(x * (1 - h))) / (2 * x * h)

# Exemple d'utilisation
phi = lambda x: x**2
x = 1.0
h = 1e-5
print(derive(phi, x, h))  # Devrait afficher une valeur proche de 2


def derive_seconde(phi, x: float, h: float) -> float:
    """
    Calcule une valeur approchée de la dérivée seconde de phi en x.
    
    Paramètres:
    phi (callable): Fonction réelle d'une variable réelle.
    x (float): Point où calculer la dérivée seconde.
    h (float): Pas de la dérivée.
    
    Retourne:
    float: Valeur approchée de la dérivée seconde.
    """
    return (phi(x + h) - 2 * phi(x) + phi(x - h)) / (h ** 2)

# Exemple d'utilisation
phi = lambda x: x**2
x = 1.0
h = 1e-5
print(derive_seconde(phi, x, h))  # Devrait afficher une valeur proche de 2


def min_local(phi, x0: float, h: float) -> float:
    """
    Trouve l'abscisse d'un minimum local de phi en utilisant la méthode de Newton.
    
    Paramètres:
    phi (callable): Fonction réelle d'une variable réelle.
    x0 (float): Point initial pour la recherche du minimum local.
    h (float): Pas de la dérivée.
    
    Retourne:
    float: Abscisse du minimum local.
    """
    x = x0
    while abs(derive(phi, x, h)) >= 1e-7:
        x = x - derive(phi, x, h) / derive_seconde(phi, x, h)
    return x

# Exemple d'utilisation
phi = lambda x: (x - 2)**2
x0 = 0.0
h = 1e-5
print(min_local(phi, x0, h))  # Devrait afficher une valeur proche de 2