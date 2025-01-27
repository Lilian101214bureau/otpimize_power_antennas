# geometry.py
"""
Module geometry
===============

Fonctions pour générer différentes géométries d'antennes :
- Manuelles (coordonnées définies à la main)
- Aléatoires dans une sphère
- Circulaires
- Carrées
- Linéaires (type Yagi)

Chaque fonction renvoie une liste de dictionnaires représentant
les dipôles, sous la forme :
[
  { 'coords': (x1, y1, z1, x2, y2, z2), 'type': 'emitter' },
  { 'coords': (...), 'type': 'receiver' },
  { 'coords': (...), 'type': 'reflector' },
  ...
]
"""

import numpy as np
from constants import lam, half_length

def generate_manual_antenna_coords(antennas_definition):
    """
    Crée la liste d'antennes à partir d'une liste de tuples manuels.

    Paramètres
    ----------
    antennas_definition : list of tuples
        Liste de tuples sous la forme:
        (xw1, yw1, zw1, xw2, yw2, zw2, type)

    Retour
    ------
    liste : list of dict
        Liste de dictionnaires {'coords': (...), 'type': ... }
    """
    antennes_coords_local = []
    for ant in antennas_definition:
        xw1, yw1, zw1, xw2, yw2, zw2, ant_type = ant
        antennes_coords_local.append({
            'coords': (xw1, yw1, zw1, xw2, yw2, zw2),
            'type': ant_type
        })
    return antennes_coords_local



    
def generate_random_antenna_coords_3D_sphere(num_antennas, lam, sphere_radius, max_attempts=1000):
    """
    Génère des coordonnées pour num_antennas dipôles dans une sphère de rayon sphere_radius.

    Hypothèses simplifiées :
    - 1 émetteur, 1 récepteur, le reste en réflecteurs
    - L'émetteur est placé à (20,0) (valeur en m), le récepteur à (0,0)
      (libre à vous de changer ce positionnement).
    - Les réflecteurs sont distribués aléatoirement dans la sphère.

    Paramètres
    ----------
    num_antennas : int
        Nombre total d'antennes (y compris émetteur et récepteur).
    lam : float
        Longueur d'onde.
    sphere_radius : float
        Rayon de la sphère en mètres.
    max_attempts : int
        Nombre maximal d'essais pour placer un réflecteur convenablement.

    Retour
    ------
    antennes_coords_local : list of dict
        Liste de dictionnaires {'coords': (...), 'type': 'emitter'|'receiver'|'reflector'}
    """
    antennes_coords_local = []
    antenna_types = ['emitter', 'receiver'] + ['reflector'] * (num_antennas - 2)

    emitter_coords = (20, 0, -half_length, 20, 0, half_length)
    receiver_coords = (0, 0, -half_length, 0, 0, half_length)

     # On exige une certaine distance minimale pour éviter intersection

    for antenna_type in antenna_types:
        if antenna_type == 'emitter':
            xw1, yw1, zw1, xw2, yw2, zw2 = emitter_coords
        elif antenna_type == 'receiver':
            xw1, yw1, zw1, xw2, yw2, zw2 = receiver_coords
        else:
             # Génération d'un point aléatoire dans la sphère
            phi = np.random.uniform(0, 2 * np.pi)
            cos_theta = np.random.uniform(-1, 1)
            sin_theta = np.sqrt(1 - cos_theta**2)
            direction = np.array([
                sin_theta * np.cos(phi),
                sin_theta * np.sin(phi),
                cos_theta
            ])
            
            u = np.random.uniform(0, 1)
            r = u ** (1 / 3) * sphere_radius
            center = r * direction
            
            # Orientation aléatoire pour le dipôle
            phi_o = np.random.uniform(0, 2 * np.pi)
            cos_theta_o = np.random.uniform(-1, 1)
            sin_theta_o = np.sqrt(1 - cos_theta_o**2)
            orientation = np.array([
                sin_theta_o * np.cos(phi_o),
                sin_theta_o * np.sin(phi_o),
                cos_theta_o
            ])

            # Calcul des deux extrémités du dipôle
            xw1, yw1, zw1 = center - half_length * orientation
            xw2, yw2, zw2 = center + half_length * orientation

        antennes_coords_local.append({
            'coords': (xw1, yw1, zw1, xw2, yw2, zw2),
            'type': antenna_type
        })

    return antennes_coords_local

def generate_circular_antenna_array(num_antennas, radius, lam=lam):
    """
    Génère un réseau circulaire dans le plan XY.
    Disposition :
    - 1 émetteur au centre
    - 1 récepteur sur le cercle (angle 0)
    - le reste des réflecteurs uniformément répartis.

    Paramètres
    ----------
    num_antennas : int
        Nombre total d'antennes.
    radius : float
        Rayon du cercle (en mètres).
    lam : float
        Longueur d'onde (utile si besoin).

    Retour
    ------
    antennes_coords : list of dict
        Liste de dipôles {'coords': (...), 'type': ...}
    """
    antennes_coords = []
    # Emetteur au centre
    emitter_coords = (0,0,-half_length,0,0,half_length)
    antennes_coords.append({'coords': emitter_coords, 'type': 'emitter'})

    # Récepteur sur l'axe x, angle = 0
    x_rec = radius
    y_rec = 0
    receiver_coords = (x_rec, y_rec, -half_length, x_rec, y_rec, half_length)
    antennes_coords.append({'coords': receiver_coords, 'type': 'receiver'})

    # Réflecteurs
    remaining = num_antennas - 2
    # Décalage d'angle pour éviter la superposition avec le récepteur (phi=0)
    phis = np.linspace(0, 2*np.pi, remaining, endpoint=False) + (2*np.pi/(2*remaining))
    for phi in phis:
        x_center = radius*np.cos(phi)
        y_center = radius*np.sin(phi)
        xw1 = x_center; yw1 = y_center; zw1 = -half_length
        xw2 = x_center; yw2 = y_center; zw2 = half_length
        antennes_coords.append({'coords': (xw1, yw1, zw1, xw2, yw2, zw2), 'type': 'reflector'})

    return antennes_coords

def generate_square_array(lam=lam, d=2*lam):
    """
    Génère un réseau carré avec 5 antennes :
    - 1 émetteur au centre (0,0)
    - 4 antennes (1 récepteur + 3 réflecteurs) disposées aux sommets du carré.

    Paramètres
    ----------
    lam : float
        Longueur d'onde.
    d : float
        Longueur du côté du carré.

    Retour
    ------
    antennes_coords_local : list of dict
    """
    Cx, Cy = lam, 0.0
    base_positions = [
        (Cx + d/2, Cy + d/2),
        (Cx - d/2, Cy + d/2),
        (Cx - d/2, Cy - d/2),
        (Cx + d/2, Cy - d/2),
    ]
    emitter_coords = (0, 0, -half_length, 0, 0, half_length)
    antennes_coords_local = [{'coords': emitter_coords, 'type': 'emitter'}]
    antenne_types = ['receiver', 'reflector', 'reflector', 'reflector']
    for i, (x, y) in enumerate(base_positions):
        coords_tag = (x, y, -half_length, x, y, half_length)
        antennes_coords_local.append({'coords': coords_tag, 'type': antenne_types[i]})
    return antennes_coords_local


def generate_linear_yagi_like_array(num_reflectors, lam, spacing=None, receiver_distance=5.0):
    """
    Génère une configuration linéaire de type Yagi :
    - 1 émetteur à x=0
    - 1 récepteur à x=receiver_distance * lam
    - Un certain nombre de réflecteurs régulièrement espacés entre Emetteur et Récepteur.

    Paramètres
    ----------
    num_reflectors : int
        Nombre de réflecteurs (entre émetteur et récepteur).
    lam : float
        Longueur d'onde.
    spacing : float or None
        Espacement entre réflecteurs. Par défaut lam/2 si None.
    receiver_distance : float
        Distance (en multiples de lam) entre émetteur et récepteur.

    Retour
    ------
    antennes_coords : list of dict
    """
    from constants import half_length  # Assurez-vous que half_length = 0.25 * lam

    if spacing is None:
        spacing = lam/2  # Espacement par défaut
    
    antennes_coords = []
    # Emetteur
    emitter_coords = (0, 0, -half_length, 0, 0, half_length)
    antennes_coords.append({'coords': emitter_coords, 'type': 'emitter'})

    # Récepteur
    rx_pos = receiver_distance * lam
    receiver_coords = (rx_pos, 0, -half_length, rx_pos, 0, half_length)
    antennes_coords.append({'coords': receiver_coords, 'type': 'receiver'})

    # Réflecteurs
    total_length = rx_pos
    needed_length = num_reflectors * spacing
    if needed_length > total_length - lam:
        spacing = (rx_pos - lam) / num_reflectors

    ref_positions = np.arange(spacing, spacing*(num_reflectors+1), spacing)
    ref_positions = ref_positions[ref_positions < rx_pos]

    for x_ref in ref_positions:
        xw1, yw1, zw1 = x_ref, 0, -half_length
        xw2, yw2, zw2 = x_ref, 0, half_length
        antennes_coords.append({'coords': (xw1, yw1, zw1, xw2, yw2, zw2), 'type': 'reflector'})

    return antennes_coords