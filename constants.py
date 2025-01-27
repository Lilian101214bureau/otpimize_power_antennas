"""
Module constants
================

Contient l'ensemble des constantes physiques et des réglages généraux
utilisés dans le projet.
"""

import math

# ----------------------------
# Constantes physiques
# ----------------------------
freq = 868e6            # Fréquence de travail en Hz
c = 299792458           # Vitesse de la lumière en m/s
lam = c / freq          # Longueur d'onde en mètres

# ----------------------------
# Paramètres de discrétisation
# ----------------------------
segment_count_impair = 5
"""
Nombre de segments pour discrétiser chaque dipôle.
Doit être impair pour avoir un segment central défini.
La fiabilité des résultats NEC augmente avec le nombre de segments
(mais le temps de simulation est plus long).
"""

half_segment = int((segment_count_impair - 1) / 2) + 1
"""
Indice du segment central (exemple : si 5 segments => segment central = 3).
"""

position_half_segment = int((segment_count_impair - 1) / 2)
"""
Dans certains contextes, on indexe les segments à partir de 0,
ce qui rend cet offset utile.
"""

# ----------------------------
# Paramètres de la géométrie
# ----------------------------
radius = 1e-6   # Rayon des fils en mètres pour la modélisation NEC
"""
Habituellement on prend 1e-6 m dans des simulations 
pour approcher un rayon « très petit » par rapport à λ.
Plus petit n'est pas forcément meilleur (limites NEC). 
"""

# Exemple d'un demi-dipôle de λ/2 complet => half_length = λ/4
half_length = 0.25 * lam
"""
Demi-longueur d’un dipôle λ/2 = lam/2.
Pour un dipôle entier de lam/2 => la moitié vaut lam/4.
"""