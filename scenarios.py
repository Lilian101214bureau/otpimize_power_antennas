import numpy as np
import matplotlib.pyplot as plt
from constants import lam, half_length
from simulation import (calculate_impedances, tension_Oc_vector, impedance_matrix, 
                        calculate_Voc, calculate_powers)
from plotting import plot_antennas
from optimization import optimize_impedances

def run_square_scenario_separate_plots():
    R = 20.0
    d_values = [(2/3)*lam, 1*lam, (3/2)*lam, 2*lam, (5/2)*lam, 3*lam]
    N_phi = 360
    phi_values = np.linspace(0, 2*np.pi, N_phi)

    def create_antennas(d, phi):
        Cx, Cy = R, 0.0
        base_positions = [
            (Cx + d/2, Cy + d/2),
            (Cx - d/2, Cy + d/2),
            (Cx - d/2, Cy - d/2),
            (Cx + d/2, Cy - d/2),
        ]
        rotated_positions = []
        for (x,y) in base_positions:
            xr = Cx + (x - Cx)*np.cos(phi) - (y - Cy)*np.sin(phi)
            yr = Cy + (x - Cx)*np.sin(phi) + (y - Cy)*np.cos(phi)
            rotated_positions.append((xr, yr))
        emitter_coords = (0, 0, -half_length, 0, 0, half_length)
        antennes_coords_local = [{'coords': emitter_coords, 'type': 'emitter'}]
        antenne_types = ['receiver', 'reflector', 'reflector', 'reflector']
        for i, (xx, yy) in enumerate(rotated_positions):
            coords_tag = (xx, yy, -half_length, xx, yy, half_length)
            antennes_coords_local.append({'coords': coords_tag, 'type': antenne_types[i]})
        return antennes_coords_local

    def create_reference_antennas(d, phi):
        # Crée un scénario avec seulement l'émetteur et le Tag0 (le premier après l'émetteur)
        # On part de la géométrie complète et on ne garde que l'émetteur et le receiver
        full = create_antennas(d, phi)
        # Le premier après l'émetteur est le receiver, donc on prend indices [0,1]
        # Emetteur = 0, receiver = 1
        reference = full[0:2]  # garder juste émetteur et recepteur
        return reference

    def get_received_power(antennes_coords):
        num_antennas = len(antennes_coords)
        imp_sim = calculate_impedances(num_antennas, antennes_coords)
        exc = tension_Oc_vector(num_antennas, antennes_coords, reglage='auto')
        imp_auto = impedance_matrix(num_antennas, antennes_coords, reglage='auto')
        results, _ = calculate_Voc(imp_sim, num_antennas, exc, imp_auto, antennes_coords)
        powers = calculate_powers(results)
        P_L_receiver = powers['P_L'][1].real
        return P_L_receiver

    def get_received_power_optimized(antennes_coords):
        # Même calcul mais avec optimisation des impédances
        num_antennas = len(antennes_coords)
        imp_sim = calculate_impedances(num_antennas, antennes_coords)
        exc = tension_Oc_vector(num_antennas, antennes_coords, reglage='auto')
        imp_auto = impedance_matrix(num_antennas, antennes_coords, reglage='auto')
        # Résultats initiaux (pour rien, juste si on veut)
        # results_init, _ = calculate_Voc(imp_sim, num_antennas, exc, imp_auto, antennes_coords)
        # Optimisation
        optimized = optimize_impedances(antennes_coords)
        optimized_impedances = []
        for i in range(num_antennas):
            R = optimized[2*i]
            X = optimized[2*i+1]
            optimized_impedances.append(complex(R,X))
        imp_opt = impedance_matrix(num_antennas, antennes_coords, reglage='manuel', impedances_manuel=optimized_impedances)
        results_opt, _ = calculate_Voc(imp_sim, num_antennas, exc, imp_opt, antennes_coords)
        powers_opt = calculate_powers(results_opt)
        P_L_receiver_opt = powers_opt['P_L'][1].real
        return P_L_receiver_opt

    # Pour chaque d, on va tracer un graphe séparé
    for d in d_values:
        P_ref = []
        P_initial = []
        P_optimized = []
        for phi in phi_values:
            # Reference = Tag0 seul
            ref_coords = create_reference_antennas(d, phi)
            P_ref.append(get_received_power(ref_coords))
            # Scenario complet avant optimisation
            full_coords = create_antennas(d, phi)
            P_initial.append(get_received_power(full_coords))
            # Scenario complet après optimisation
            P_optimized.append(get_received_power_optimized(full_coords))

        # On crée une figure spécifique pour chaque d
        fig = plt.figure(figsize=(10,8))
        # Subplot 1: la géométrie à φ=0
        ax1 = fig.add_subplot(2, 1, 1, projection='3d')
        # Affichage de la géométrie pour φ=0
        full_coords_phi0 = create_antennas(d, 0.0)
        # on utilise plot_antennas qui crée une figure, on va l'adapter pour tracer sur ax1
        # On va ré-implémenter un petit code ici pour tracer sur ax1 sans créer une nouvelle figure
        antennes_coords_lambda = []
        for antenna in full_coords_phi0:
            xw1, yw1, zw1, xw2, yw2, zw2 = antenna['coords']
            antennes_coords_lambda.append({
                'coords': (
                    xw1 / lam, yw1 / lam, zw1 / lam,
                    xw2 / lam, yw2 / lam, zw2 / lam
                ),
                'type': antenna['type']
            })

        # Déterminer les limites
        all_coords = [coord for antenna in antennes_coords_lambda for coord in antenna['coords']]
        all_coords = np.array(all_coords).reshape(-1, 3)
        max_coord = np.max(all_coords, axis=0)
        min_coord = np.min(all_coords, axis=0)
        max_limit = max(np.max(max_coord), np.abs(np.min(min_coord))) * 1.1
        ax1.set_xlim([-max_limit, max_limit])
        ax1.set_ylim([-max_limit, max_limit])
        ax1.set_zlim([-max_limit, max_limit])
        ax1.set_box_aspect([1, 1, 1])

        colors = {'emitter': 'red', 'receiver': 'green', 'reflector': 'blue'}
        labels = {'emitter': 'Émetteur', 'receiver': 'Récepteur', 'reflector': 'Réflecteur'}

        plotted_labels = set()
        for antenna in antennes_coords_lambda:
            xw1, yw1, zw1, xw2, yw2, zw2 = antenna['coords']
            antenna_type = antenna['type']
            lbl = labels[antenna_type]
            if lbl not in plotted_labels:
                ax1.plot([xw1, xw2], [yw1, yw2], [zw1, zw2],
                         color=colors[antenna_type], marker='o', label=lbl)
                plotted_labels.add(lbl)
            else:
                ax1.plot([xw1, xw2], [yw1, yw2], [zw1, zw2],
                         color=colors[antenna_type], marker='o')
        ax1.set_xlabel('X (lamnda)')
        ax1.set_ylabel('Y (lamnda)')
        ax1.set_zlabel('Z (lamnda)')
        ax1.set_title(f"Géométrie carré, d={d/lam:.2f} lamnda, φ=0")
        ax1.legend()

        # Subplot 2: les trois courbes de puissance
        ax2 = fig.add_subplot(2,1,2)
        ax2.plot(phi_values, P_ref, 'r--', label="référence (Tag0 seul)")
        ax2.plot(phi_values, P_initial, 'b', label="Avec réflecteurs (avant opti)")
        ax2.plot(phi_values, P_optimized, 'g', label="Avec réflecteurs (après opti)")

        ax2.set_xlabel("Angle φ (radians)")
        ax2.set_ylabel("Puissance reçue par Tag0 (uW)")
        ax2.set_title(f"Puissance reçue par Tag0 en fonction de φ (d={d/lam:.2f} lamnda)")
        ax2.grid(True)
        ax2.legend()

        plt.tight_layout()
        plt.show()
