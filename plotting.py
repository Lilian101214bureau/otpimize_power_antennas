# ============================================================================
# plotting.py
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from constants import lam

"""
Module plotting
===============

Fonctions pour tracer la disposition 3D des antennes
et visualiser les puissances (sous forme de barres).
"""

def plot_antennas(antennes_coords, lam=1.0, filename="geometry_default.png"):
    """
    Crée et enregistre automatiquement un plot 3D des dipôles,
    puis ferme la figure.

    Paramètres
    ----------
    antennes_coords : list of dict ou list of tuples
        Coordonnées des dipôles (x1,y1,z1, x2,y2,z2) + type
    lam : float
        Longueur d'onde (pour normaliser les coordonnées).
    filename : str
        Nom (ou chemin) du fichier de sortie où l'image sera sauvegardée.
        Par défaut, "geometry_default.png".

    Retour
    ------
    None
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Conversion en fraction de lam
    antennes_coords_lambda = []
    for antenna in antennes_coords:
        if isinstance(antenna, dict):
            (x1, y1, z1, x2, y2, z2) = antenna['coords']
            atype = antenna['type']
        else:
            (x1, y1, z1, x2, y2, z2, atype) = antenna
        antennes_coords_lambda.append({
            'coords': (x1/lam, y1/lam, z1/lam, x2/lam, y2/lam, z2/lam),
            'type': atype
        })

    # Détermination des limites
    all_coords = [c for a in antennes_coords_lambda for c in a['coords']]
    all_coords = np.array(all_coords).reshape(-1, 3)
    if len(all_coords) > 0:
        max_coord = np.max(all_coords, axis=0)
        min_coord = np.min(all_coords, axis=0)
        max_limit = max(abs(max_coord).max(), abs(min_coord).max()) * 1.1
    else:
        max_limit = 1.0

    ax.set_xlim([-max_limit, max_limit])
    ax.set_ylim([-max_limit, max_limit])
    ax.set_zlim([-max_limit, max_limit])
    ax.set_box_aspect([1, 1, 1])

    colors = {'emitter': 'red', 'receiver': 'green', 'reflector': 'blue'}
    labels = {'emitter': 'Émetteur', 'receiver': 'Récepteur', 'reflector': 'Réflecteur'}

    plotted_labels = set()
    for a in antennes_coords_lambda:
        (x1, y1, z1, x2, y2, z2) = a['coords']
        atype = a['type']
        c = colors.get(atype, 'black')
        lab = labels.get(atype, atype)
        if lab not in plotted_labels:
            ax.plot([x1, x2], [y1, y2], [z1, z2], color=c, marker='o', label=lab)
            plotted_labels.add(lab)
        else:
            ax.plot([x1, x2], [y1, y2], [z1, z2], color=c, marker='o')

    ax.set_xlabel('X (λ)')
    ax.set_ylabel('Y (λ)')
    ax.set_zlabel('Z (λ)')
    ax.set_title("Antennes générées (échelle en λ)")
    ax.legend()

    # Sauvegarde et fermeture automatique
    plt.savefig(filename, dpi=300)
    plt.close(fig)


def visualize_powers(
    power_data,           # ex. [array, array, array, array] => P_Voc, P_Vin_oc, P_in, P_L
    labels,               # ex. ["P_Voc", "P_Vin_oc", "P_in", "P_L"]
    title,
    dipole_numbers,       # ex. [0,1,2,...] ou range(num_antennas)
    antennes_coords,      # la liste d'antennes (pour l'étiquetage)
    filename=None,
    show=False
):
    """
    Trace un bar-plot comparant plusieurs types de puissances (ex. P_Voc, P_in, etc.)
    pour chaque dipôle.

    Paramètres
    ----------
    power_data : list of np.array
        ex. [ array_P_Voc, array_P_Vin_oc, array_P_in, array_P_L ]
        Chacune doit être de taille (N,) => 1D
    labels : list of str
    title : str
    dipole_numbers : array-like
        indices de dipôles (ex. range(num_antennas))
    antennes_coords : list
        liste des antennes pour l'étiquetage X (Émetteur/Récepteur/Réflecteur)
    filename : str or None
    show : bool

    Retour
    ------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    num_ant = len(dipole_numbers)

    # Préparation des étiquettes X => "Émetteur", "Récepteur", "Réflecteur n"
    xlabels = []
    em_idx = None
    rx_idx = None
    ref_count = 1
    for i, ant in enumerate(antennes_coords):
        if ant['type'] == 'emitter':
            xlabels.append("Émetteur")
            em_idx = i
        elif ant['type'] == 'receiver':
            xlabels.append("Récepteur")
            rx_idx = i
        else:
            xlabels.append(f"Réflecteur {ref_count}")
            ref_count += 1

    bar_width = 0.2
    offset = np.arange(len(power_data)) * bar_width

    # Tracé des barres
    # => On suppose ici que power_data[i] est 1D
    for i, data in enumerate(power_data):
        # S'il y a un risque que data soit shape (N,1), on peut faire data = data.flatten()
        data_1d = np.array(data).flatten()
        ax.bar(dipole_numbers + offset[i], data_1d, bar_width, label=labels[i])

    # Personnalisation des abscisses
    ax.set_xticks(dipole_numbers + bar_width * (len(power_data) - 1) / 2)
    ax.set_xticklabels(xlabels)

    # Affichage du P_Voc sur Émetteur et P_L sur Récepteur
    # => P_Voc = power_data[0], P_L = power_data[3]
    if em_idx is not None and len(power_data) >= 1:
        val_em = np.array(power_data[0]).flatten()[em_idx]  # flatten au cas où
        ax.text(em_idx + offset[0], val_em + 2,
                f"{val_em:.1f} uW (ém)",
                ha='center', va='bottom', color='blue')

    if rx_idx is not None and len(power_data) >= 4:
        val_rx = np.array(power_data[3]).flatten()[rx_idx]
        ax.text(rx_idx + offset[3], val_rx + 2,
                f"{val_rx:.1f} uW (rx)",
                ha='center', va='bottom', color='black')

    # Décoration
    ax.set_xlabel("Type d'antenne")
    ax.set_ylabel("Puissance (uW)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    if filename:
        plt.savefig(filename, dpi=300)
    if show:
        plt.show()

    return fig, ax
