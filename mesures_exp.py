import numpy as np

def seuillage(A: np.ndarray, seuil: int) -> np.ndarray:
    """
    Effectue le seuillage de l'image.
    
    Paramètres:
    A (np.ndarray): Tableau d'entiers à deux dimensions représentant l'image.
    seuil (int): Valeur seuil du niveau de gris.
    
    Retourne:
    np.ndarray: Tableau seuillé de même forme avec 1 pour les pixels < seuil, et 0 pour les autres.
    """
    # Utilisation de l'opérateur conditionnel pour effectuer le seuillage
    return np.where(A < seuil, 1, 0)

# Exemple d'utilisation
A = np.array([[157, 200, 100], [50, 180, 90], [120, 250, 60]])
seuil = 150
print(seuillage(A, seuil))

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

# Exemple d'utilisation
A_seuil = seuillage(A, seuil)
print(pixel_centre_bille(A_seuil))

def prendre_photo() -> np.ndarray:
    """
    Simule la prise d'une photo par la caméra CCD.
    
    Retourne:
    np.ndarray: Image prise sous la forme d'un tableau à deux dimensions.
    """
    # Exemple de simulation d'une image prise par la caméra
    return np.random.randint(0, 256, (100, 100))

def positions(n: int, seuil: int) -> [(int, int)]:
    """
    Prend n photographies de la bille et renvoie la liste de ses positions.
    
    Paramètres:
    n (int): Nombre de photographies à prendre.
    seuil (int): Valeur seuil du niveau de gris.
    
    Retourne:
    [(int, int)]: Liste des positions du centre de la bille dans chaque photographie.
    """
    positions = []
    
    for _ in range(n):
        photo = prendre_photo()
        A_seuil = seuillage(photo, seuil)
        centre = pixel_centre_bille(A_seuil)
        if centre:
            positions.append(centre)
    
    return positions

# Exemple d'utilisation
print(positions(5, seuil))

def fluctuations(P: [(int, int)], t: float) -> float:
    """
    Calcule la moyenne des déplacements quadratiques de la bille.
    
    Paramètres:
    P ([(int, int)]): Liste des positions successives de la bille.
    t (float): Longueur correspondant à un pixel.
    
    Retourne:
    float: Valeur moyenne des déplacements quadratiques.
    """
    # Conversion des positions en numpy array pour faciliter les calculs
    positions = np.array(P)
    
    # Calcul du barycentre des différentes positions observées
    barycentre = np.mean(positions, axis=0)
    
    # Calcul des écarts quadratiques moyens
    ecarts_quadratiques = np.sum((positions - barycentre) ** 2, axis=1)
    
    # Moyenne des écarts quadratiques
    moyenne_deplacements_quadratiques = np.mean(ecarts_quadratiques) * t ** 2
    
    return moyenne_deplacements_quadratiques

# Exemple d'utilisation
positions_exemple = positions(5, seuil)
longueur_pixel = 0.1  # Exemple de longueur correspondant à un pixel
print(fluctuations(positions_exemple, longueur_pixel))