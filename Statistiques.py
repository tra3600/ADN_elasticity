def moyenne(X) -> float:
    """
    Calcule la moyenne d'une séquence de nombres.
    
    Paramètres:
    X (sequence of numbers): La séquence de nombres.
    
    Retourne:
    float: La moyenne des nombres dans la séquence.
    """
    return sum(X) / len(X)

# Exemple d'utilisation
print(moyenne([1, 2, 3, 4]))  # Devrait afficher 2.5

def variance(X) -> float:
    """
    Calcule la variance d'une séquence de nombres.
    
    Paramètres:
    X (sequence of numbers): La séquence de nombres.
    
    Retourne:
    float: La variance des nombres dans la séquence.
    """
    mean = moyenne(X)
    return sum((x - mean) ** 2 for x in X) / len(X)

# Exemple d'utilisation
print(variance([1, 2, 3, 4]))  # Devrait afficher 1.25

import numbers

def somme(M):
    """
    Calcule la somme de tous les éléments d'une séquence imbriquée.
    
    Paramètres:
    M (sequence): Séquence imbriquée de profondeur et de structure quelconques.
    
    Retourne:
    float: La somme de tous les éléments numériques dans la séquence.
    """
    total = 0
    for element in M:
        if isinstance(element, numbers.Real):
            total += element
        else:
            total += somme(element)
    return total

# Exemple d'utilisation
print(somme([[[1, 2], [3, 4, 5]], 6, [7, 8], 9]))  # Devrait afficher 45