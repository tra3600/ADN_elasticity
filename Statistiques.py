"""Partie I — Fonctions utilitaires (Q1 à Q3)."""

import numbers


def moyenne(X) -> float:
    """Q1. Moyenne d'une séquence de nombres (X n'est pas modifiée)."""
    if len(X) == 0:
        raise ValueError("moyenne d'une séquence vide")
    total = 0.0
    for x in X:
        total += x
    return total / len(X)


def variance(X) -> float:
    """Q2. Variance d'une séquence : moyenne des carrés des écarts à la moyenne.

    On utilise la définition « écarts à la moyenne » plutôt que
    « moyenne des carrés - carré de la moyenne », numériquement plus stable
    (pas de soustraction de deux grands nombres voisins).
    """
    m = moyenne(X)
    return moyenne([(x - m) ** 2 for x in X])


def somme(M):
    """Q3. Somme de tous les nombres d'une séquence imbriquée quelconque (récursif)."""
    if isinstance(M, numbers.Real):
        return M
    total = 0
    for element in M:
        total += somme(element)
    return total


if __name__ == "__main__":
    print(moyenne([1, 2, 3, 4]))                     # 2.5
    print(variance([1, 2, 3, 4]))                    # 1.25
    print(somme([[[1, 2], [3, 4, 5]], 6, [7, 8], 9]))  # 45
