"""
--------------------------------------------------------------------------------
FICHIER : test_get_matrix.py

DESCRIPTION :
Ce script a pour objectif de générer et d’enregistrer les matrices d’impédance
de différents scénarios d’antennes (aléatoires et manuels), ainsi que de calculer
et sauvegarder diverses grandeurs électriques associées (tensions, courants,
puissances, etc.).

FONCTIONNEMENT GÉNÉRAL :
1) Paramétrage global (chemins d’accès, constantes, etc.).
2) SCÉNARIO(S) ALÉATOIRE(S) :
   - Génération de la géométrie (coordonnées des antennes) de façon aléatoire.
   - Calcul de la matrice d’impédance [Z].
   - Visualisation et enregistrement de la géométrie.
   - Sauvegarde de la matrice [Z] et des résultats (Voc, courants, puissances...).
   - Mesure et enregistrement du temps de simulation.
3) SCÉNARIO MANUEL :
   - Définition manuelle des antennes (coords, type, impédances de charge…).
   - Calcul de [Z], sauvegarde de la géométrie et des résultats,
     ainsi que le temps de calcul.    
4) Génération des différents fichiers CSV et PNG relatifs aux résultats
   (matrices, grandeurs calculées, temps de calcul).

UTILISATION :
- Adapter les paramètres (chemins de sauvegarde des données, nombre d'antennes, etc.) selon vos besoins.
- Lancer le script (python test_get_matrix.py) : il génère automatiquement
  les fichiers de résultats dans le dossier spécifié.

DÉPENDANCES :
- Python 3.x
- numpy, matplotlib, csv, os, time
- Modules internes : geometry.py, simulation.py, plotting.py, constants.py
  (fournissant les fonctions et constantes utilisées dans ce script)

 AUTEUR : Michalak Lilian / Équipe : Rhodes / lilianmichalak2002@gmail.com
--------------------------------------------------------------------------------
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import time

# --- Import de vos fonctions spécifiques (ex. geometry, simulation, plotting) ---
from geometry import (
    generate_random_antenna_coords_3D_sphere,
    generate_manual_antenna_coords  # <-- on importe aussi la fonction manuelle
)
from simulation import (
    calculate_impedances,
    tension_Oc_vector,
    calculate_Voc,
    calculate_powers
)
from plotting import plot_antennas
from constants import lam, half_length

def main():
    # 1) Paramètres globaux
    results_dir = r"C:\Documents\python-necpp\PyNEC\example\project\structurationned_code\full_wave_approach\result_matrix" #à modifier pour sauvegarder vos resultats dans vos dossiers
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # On veut un rayon = 5*lam pour la géométrie aléatoire
    radius = 5 * lam

    # 2) SCÉNARIO ALÉATOIRE
    # ---------------------
    # Liste des nombres d'antennes à tester en mode aléatoire
    antenna_counts = [5]  # par exemple
    # Si vous voulez tester davantage de cas :
    # antenna_counts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]s

    for n in antenna_counts:
        scenario_name = f"sphere_{n}ant_{2.0}lambda"
        print(f"\n=== [ALÉATOIRE] Génération de la géométrie pour {scenario_name} ===")

        # a) Génération de la géométrie (n antennes) aléatoires
        antennes_coords = generate_random_antenna_coords_3D_sphere(n, lam, radius)

        # b) Calcul de la matrice d'impédance [Z]
        start_time = time.time()
        Z = calculate_impedances(n, antennes_coords)
        end_time = time.time()
        simulation_time = end_time - start_time

        # c) Affichage / Enregistrement de la géométrie
        outgeo = os.path.join(results_dir, f"{scenario_name}_geometry.png")
        plot_antennas(antennes_coords, lam=lam, filename=outgeo)

        # d) Sauvegarde de la matrice [Z]
        csv_file = os.path.join(results_dir, f"{scenario_name}_Zmatrix.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f"# Scenario: {scenario_name}, rayon={radius:.2f}, antennes={n}"])
            writer.writerow([f"# Simulation time (s): {simulation_time:.4f}"])
            writer.writerow(["Re(Z_ij)", "Im(Z_ij)"])
            for row in Z:
                for z in row:
                    writer.writerow([z.real, z.imag])

        # e) Calcul de Voc/courants/puissances
        #    On met toutes les antennes en circuit ouvert (Z=0 -> en pratique ∞)
        impedance_loads = np.zeros((n, n), dtype=complex)
        voc_vector = tension_Oc_vector(n, antennes_coords, reglage='auto')
        results, _ = calculate_Voc(Z, n, voc_vector, impedance_loads, antennes_coords)
        power_dict = calculate_powers(results)

        # f) Sauvegarde de toutes les grandeurs calculées
        csv_file2 = os.path.join(results_dir, f"{scenario_name}_results.csv")
        with open(csv_file2, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [
                "Antenna_index", "Type",
                "Voc_Re", "Voc_Im",
                "Vin_Re", "Vin_Im",
                "Vl_Re",  "Vl_Im",
                "vin_Oc_Re", "vin_Oc_Im",
                "vbis_Re", "vbis_Im",
                "I_Re", "I_Im",
                "Zin_Re", "Zin_Im",
                "P_V_oc(uW)", "P_Vin_oc(uW)", "P_L(uW)", "P_in(uW)"
            ]
            writer.writerow(header)

            # Raccourcis vers les tableaux du dictionnaire "results"
            Voc   = results['Voc'].reshape(-1)
            Vin   = results['Vin'].reshape(-1)
            Vl    = results['Vl'].reshape(-1)
            vinOc = results['vin_Oc'].reshape(-1)
            vbis  = results['vbis'].reshape(-1)
            I     = results['currents_at_center'].reshape(-1)
            zin   = results['zin'].reshape(-1)

            # Raccourcis vers les puissances
            P_V_oc   = power_dict['P_V_oc'].reshape(-1)
            P_Vin_oc = power_dict['P_Vin_oc'].reshape(-1)
            P_L      = power_dict['P_L'].reshape(-1)
            P_in     = power_dict['P_in'].reshape(-1)

            # Export par antenne
            for i in range(n):
                antenna_type = antennes_coords[i]['type']
                row = [
                    i, antenna_type,
                    Voc[i].real,   Voc[i].imag,
                    Vin[i].real,   Vin[i].imag,
                    Vl[i].real,    Vl[i].imag,
                    vinOc[i].real, vinOc[i].imag,
                    vbis[i].real,  vbis[i].imag,
                    I[i].real,     I[i].imag,
                    zin[i].real,   zin[i].imag,
                    P_V_oc[i],     P_Vin_oc[i],
                    P_L[i],        P_in[i]
                ]
                writer.writerow(row)

        # g) Enregistrer le temps de simulation
        time_file = os.path.join(results_dir, f"{scenario_name}_time.csv")
        with open(time_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Scenario", "Nb_Antennes", "Simulation_Time_s"])
            writer.writerow([scenario_name, n, f"{simulation_time:.4f}"])

        print(f"-> [ALÉATOIRE] Fichiers enregistrés : {csv_file}, {csv_file2}")
        print(f"-> Temps de simulation : {simulation_time:.4f} s")
        print("--------------------------------------------------------")


    # 3) SCÉNARIO MANUEL
    # ------------------
    scenario_name_manual = "manual_0"
    print(f"\n=== [MANUEL] Génération de la géométrie pour {scenario_name_manual} ===")

    # a) Définition manuelle
    manual_defs_0 = [
        (0, 0, -half_length, 0, 0,  half_length,  'emitter'),
        (lam, 0, -half_length, lam, 0, half_length,  'receiver'),
        # Ajouter ici un réflecteur si besoin:
        # (0.3*lam, 0, -half_length, 0.3*lam, 0, half_length, 'reflector'),
    ]
    antennes_coords_man = generate_manual_antenna_coords(manual_defs_0)
    n_man = len(antennes_coords_man)

    # b) Calcul matrice d’impédances [Z]
    start_time_man = time.time()
    Z_man = calculate_impedances(n_man, antennes_coords_man)
    end_time_man = time.time()
    simulation_time_man = end_time_man - start_time_man

    # c) Visu & enregistrement de la géométrie
    outgeo_man = os.path.join(results_dir, f"{scenario_name_manual}_geometry.png")
    plot_antennas(antennes_coords_man, lam=lam, filename=outgeo_man)

    # d) Sauvegarde matrice Z
    csv_file_man = os.path.join(results_dir, f"{scenario_name_manual}_Zmatrix.csv")
    with open(csv_file_man, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"# Scenario: {scenario_name_manual}, antennes={n_man}"])
        writer.writerow([f"# Simulation time (s): {simulation_time_man:.4f}"])
        writer.writerow(["Re(Z_ij)", "Im(Z_ij)"])
        for row in Z_man:
            for z in row:
                writer.writerow([z.real, z.imag])

    # e) Calcul Voc/courants/puissances
    impedance_loads_man = np.zeros((n_man, n_man), dtype=complex)
    voc_vector_man = tension_Oc_vector(n_man, antennes_coords_man, reglage='auto')
    results_man, _ = calculate_Voc(Z_man, n_man, voc_vector_man, impedance_loads_man, antennes_coords_man)
    power_dict_man = calculate_powers(results_man)

    # f) Sauvegarde résultats
    csv_file2_man = os.path.join(results_dir, f"{scenario_name_manual}_results.csv")
    with open(csv_file2_man, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            "Antenna_index", "Type",
            "Voc_Re", "Voc_Im",
            "Vin_Re", "Vin_Im",
            "Vl_Re",  "Vl_Im",
            "vin_Oc_Re", "vin_Oc_Im",
            "vbis_Re", "vbis_Im",
            "I_Re", "I_Im",
            "Zin_Re", "Zin_Im",
            "P_V_oc(uW)", "P_Vin_oc(uW)", "P_L(uW)", "P_in(uW)"
        ]
        writer.writerow(header)

        Voc   = results_man['Voc'].reshape(-1)
        Vin   = results_man['Vin'].reshape(-1)
        Vl    = results_man['Vl'].reshape(-1)
        vinOc = results_man['vin_Oc'].reshape(-1)
        vbis  = results_man['vbis'].reshape(-1)
        I     = results_man['currents_at_center'].reshape(-1)
        zin   = results_man['zin'].reshape(-1)

        P_V_oc   = power_dict_man['P_V_oc'].reshape(-1)
        P_Vin_oc = power_dict_man['P_Vin_oc'].reshape(-1)
        P_L      = power_dict_man['P_L'].reshape(-1)
        P_in     = power_dict_man['P_in'].reshape(-1)

        for i in range(n_man):
            ant_type = antennes_coords_man[i]['type']
            row = [
                i, ant_type,
                Voc[i].real,   Voc[i].imag,
                Vin[i].real,   Vin[i].imag,
                Vl[i].real,    Vl[i].imag,
                vinOc[i].real, vinOc[i].imag,
                vbis[i].real,  vbis[i].imag,
                I[i].real,     I[i].imag,
                zin[i].real,   zin[i].imag,
                P_V_oc[i],     P_Vin_oc[i],
                P_L[i],        P_in[i]
            ]
            writer.writerow(row)

    # g) Temps de simulation manuel
    time_file_man = os.path.join(results_dir, f"{scenario_name_manual}_time.csv")
    with open(time_file_man, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Scenario", "Nb_Antennes", "Simulation_Time_s"])
        writer.writerow([scenario_name_manual, n_man, f"{simulation_time_man:.4f}"])

    print(f"-> [MANUEL] Fichiers enregistrés : {csv_file_man}, {csv_file2_man}")
    print(f"-> Temps de simulation : {simulation_time_man:.4f} s")
    print("--------------------------------------------------------")

    print("Toutes les simulations (aléatoire + manuel) sont terminées.")


if __name__ == "__main__":
    main()
