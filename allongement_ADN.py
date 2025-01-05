import math
import numpy as np
import random

def pixel_centre_bille(A: np.ndarray) -> (int, int):
    """
    Calcule le barycentre des pixels à 1 dans l'image seuillée pour déterminer le centre de la bille.
    
    Paramètres:
    A (np.ndarray): Tableau seuillé avec 1 pour les pixels intéressants et 0 pour les autres.
    
    Retourne:
    (int, int): Indices (ligne, colonne) du pixel le plus proche du centre de la bille.
    """
    # Indices des pixels à 1
    indices = np.argwhere(A == 1)
    
    # Calcul des barycentres (moyenne des indices)
    if len(indices) == 0:
        return None  # Aucun pixel à 1 trouvé
    
    barycentre_i = int(np.mean(indices[:, 0]))
    barycentre_j = int(np.mean(indices[:, 1]))
    
    return (barycentre_i, barycentre_j)


import numpy as np

def distance(point1, point2):
    """Calcule la distance euclidienne entre deux points."""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def anneau_index(distance, max_distance, n):
    """Détermine l'index de l'anneau auquel appartient un pixel."""
    return int(distance / max_distance * n)

def profil(A: np.ndarray, n: int):
    """
    Construit le profil d'une figure de diffraction seuillée en la découpant en n anneaux concentriques.
    
    Paramètres:
    A (np.ndarray): Image seuillée.
    n (int): Nombre d'anneaux concentriques.
    
    Retourne:
    list: Proportion de pixels blancs dans chaque anneau.
    """
    # Déterminer le centre de la bille
    centre = pixel_centre_bille(A)
    if centre is None:
        return [0] * n  # Aucun pixel blanc trouvé

    # Initialiser les compteurs de pixels pour chaque anneau
    anneaux = [0] * n
    blancs = [0] * n

    # Déterminer la distance maximale du centre aux coins de l'image
    max_distance = np.sqrt((A.shape[0] // 2) ** 2 + (A.shape[1] // 2) ** 2)

    # Parcourir tous les pixels de l'image
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            dist = distance((i, j), centre)
            index = anneau_index(dist, max_distance, n)
            if index < n:
                anneaux[index] += 1
                if A[i, j] == 1:
                    blancs[index] += 1

    # Calculer la proportion de pixels blancs dans chaque anneau
    proportions = [blancs[i] / anneaux[i] if anneaux[i] > 0 else 0 for i in range(n)]
    
    return proportions

# Exemple d'utilisation
A = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 1],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0]
])
n = 5
print(profil(A, n))