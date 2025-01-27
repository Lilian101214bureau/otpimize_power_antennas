"""
===========

Point d'entrée principal du programme :
- Gère la création de scenarii (positions d'antennes)
- Lance les différentes stratégies : 
    * GA global
    * "ReflectAll"
    * Reflecteurs en continu
    * Reflecteurs en discret (N)
- Sauvegarde des résultats (PNG, CSV, etc.)
"""
###############################################################################
# main.py (corrected) : GA global + reflect_all + reflectors_continu + reflectors_discret
###############################################################################
import os
import time
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # backend non-interactif (pour génération batch des figures)
import matplotlib.pyplot as plt
import math

from datetime import datetime

# Import des constantes
from constants import lam, half_length

# Import des fonctions pour générer la géométrie
from geometry import (
    generate_manual_antenna_coords,
    generate_random_antenna_coords_3D_sphere,
    generate_circular_antenna_array,
    generate_square_array,
    generate_linear_yagi_like_array,
)

# Import des fonctions de simulation
from simulation import (
    calculate_impedances,
    tension_Oc_vector,
    impedance_matrix,
    calculate_Voc,
    calculate_powers,
)

# Import de nos fonctions d'optimisation
from optimization import (
    optimize_impedances,
    optimize_reflector_only_continu,
    optimize_reflector_only_discrete,
    generate_discrete_reactances
)

# Import des fonctions de tracé
# ==> IMPORTANT : maintenant plot_antennas(...) enregistre et ferme directement la figure
from plotting import plot_antennas, visualize_powers


def scenario_reflect_all(antennes_coords):
    """
    Implémente le réglage "ReflectAll" :
    - Emetteur => -jIm(Zin_em)
    - Récepteur => conj(Zin_rx)
    - Reflecteur => -jIm(Zin_ref)
    """
    num_ant = len(antennes_coords)
    Zall = calculate_impedances(num_ant, antennes_coords)

    # On "ouvre" tout sauf emitter=0Ω et receiver=50Ω
    base_loads = []
    em_idx, rx_idx, ref_idx = None, None, None
    for i, ant in enumerate(antennes_coords):
        if ant['type'] == 'emitter':
            em_idx = i
            base_loads.append(complex(0,0))
        elif ant['type'] == 'receiver':
            rx_idx = i
            base_loads.append(complex(50,0))
        else:
            base_loads.append(complex(1e9,1e9))

    Voc_ = tension_Oc_vector(num_ant, antennes_coords,'auto')
    mat_ = impedance_matrix(num_ant, antennes_coords,'manuel', base_loads)
    resOpen, _ = calculate_Voc(Zall, num_ant, Voc_, mat_, antennes_coords)
    ZinAll = resOpen['zin']

    final_loads = base_loads[:]
    if em_idx is not None:
        Xem = -ZinAll[em_idx].imag
        final_loads[em_idx] = complex(0,Xem)
    if rx_idx is not None:
        final_loads[rx_idx] = np.conjugate(ZinAll[rx_idx])

    # Pour tous les reflectors
    for i, ant in enumerate(antennes_coords):
        if ant['type'] == 'reflector':
            Xrf = -ZinAll[i].imag
            final_loads[i] = complex(0, Xrf)

    matFin = impedance_matrix(num_ant, antennes_coords,'manuel', final_loads)
    resFin, _ = calculate_Voc(Zall, num_ant, Voc_, matFin, antennes_coords)
    return resFin


def run_scenario_reflect_all(antennes_coords, scenario_name, geometry_id, save_dir, plot_geometry=True):
    """
    Compare BEFORE vs AFTER pour le "ReflectAll".
    """
    num_ant = len(antennes_coords)
    print(f"\n=== [REFLECT_ALL] {scenario_name}, {geometry_id} ({num_ant} antennes) ===")

    if plot_geometry:
        outgeo = os.path.join(save_dir, f"{scenario_name}_{geometry_id}_reflectAll_geometry.png")
        # Appel direct => la figure est sauvegardée et fermée par plot_antennas
        plot_antennas(antennes_coords, lam=lam, filename=outgeo)

    # BEFORE
    Zall = calculate_impedances(num_ant, antennes_coords)
    exc  = tension_Oc_vector(num_ant, antennes_coords,'auto')
    matA = impedance_matrix(num_ant, antennes_coords,'auto')
    resA,_ = calculate_Voc(Zall, num_ant, exc, matA, antennes_coords)
    pA = calculate_powers(resA)

    # AFTER => scenario_reflect_all
    res_ref = scenario_reflect_all(antennes_coords)
    p_ref   = calculate_powers(res_ref)

    labels = ["$P_{V_{oc}}$", "$P_{V_{in_{oc}}}$","$P_{in}$","$P_{L}$"]
    arr_bef = [
       pA['P_V_oc'], pA['P_Vin_oc'], pA['P_in'], pA['P_L']
    ]
    arr_aft = [
       p_ref['P_V_oc'], p_ref['P_Vin_oc'], p_ref['P_in'], p_ref['P_L']
    ]

    fig_bef, _ = visualize_powers(
       arr_bef, labels,
       f"[REFLECT_ALL] {scenario_name}-{geometry_id} => BEFORE",
       np.arange(num_ant),
       antennes_coords,
       show=False
    )
    out_bef = os.path.join(save_dir, f"{scenario_name}_{geometry_id}_reflectAll_powers_before.png")
    plt.savefig(out_bef, dpi=300)
    plt.close(fig_bef)

    fig_aft, _ = visualize_powers(
       arr_aft, labels,
       f"[REFLECT_ALL] {scenario_name}-{geometry_id} => AFTER",
       np.arange(num_ant),
       antennes_coords,
       show=False
    )
    out_aft = os.path.join(save_dir, f"{scenario_name}_{geometry_id}_reflectAll_powers_after.png")
    plt.savefig(out_aft, dpi=300)
    plt.close(fig_aft)

    print(f"[REFLECT_ALL] Figures enregistrées dans {save_dir}.")


def run_scenario(antennes_coords, scenario_name, geometry_id, save_dir,
                 plot_geometry=True, optimize=True):
    """
    GA global
    """
    num_antennas = len(antennes_coords)
    print(f"\n=== Scénario: {scenario_name}, {geometry_id} ({num_antennas} antennes) ===")

    if plot_geometry:
        out_geo = os.path.join(save_dir, f"{scenario_name}_{geometry_id}_geometry.png")
        plot_antennas(antennes_coords, lam=lam, filename=out_geo)

    # Avant
    Zall = calculate_impedances(num_antennas, antennes_coords)
    exc  = tension_Oc_vector(num_antennas, antennes_coords, reglage='auto')
    matA = impedance_matrix(num_antennas, antennes_coords, reglage='auto')
    res_init,_= calculate_Voc(Zall, num_antennas, exc, matA, antennes_coords)
    p_init= calculate_powers(res_init)

    # Sauvegarde CSV before
    powers_before_csv = os.path.join(save_dir, f"{scenario_name}_{geometry_id}_powers_before.csv")
    with open(powers_before_csv,'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(["Dipole","P_V_oc(uW)","P_Vin_oc(uW)","P_in(uW)","P_L(uW)"])
        for i in range(num_antennas):
            w.writerow([i,
                        p_init['P_V_oc'][i],
                        p_init['P_Vin_oc'][i],
                        p_init['P_in'][i],
                        p_init['P_L'][i] ])

    # Plot before
    arr_bef = [
       p_init['P_V_oc'], p_init['P_Vin_oc'], p_init['P_in'], p_init['P_L']
    ]
    labels = ["$P_{V_{oc}}$","$P_{V_{in_{oc}}}$","$P_{in}$","$P_{L}$"]
    fig_bef,_ = visualize_powers(
        arr_bef, labels,
        f"Puissances avant opti ({scenario_name})",
        np.arange(num_antennas),
        antennes_coords,
        show=False
    )
    out_init = os.path.join(save_dir, f"{scenario_name}_{geometry_id}_powers_before.png")
    plt.savefig(out_init, dpi=300)
    plt.close(fig_bef)

    # Optimisation GA
    if optimize:
        start_opt = time.time()
        solution = optimize_impedances(antennes_coords)
        dt_ = time.time() - start_opt
        print(f"[GA global] Durée optimisation : {dt_:.2f}s")

        # Recontruit
        if solution is not None:
            final_loads = []
            for i in range(0, len(solution), 2):
                R_ = solution[i]
                X_ = solution[i+1]
                final_loads.append(complex(R_, X_))
        else:
            # par sécurité
            final_loads = [complex(73, 42.5)] * num_antennas

        matB = impedance_matrix(num_antennas, antennes_coords, 'manuel', final_loads)
        res_opt,_= calculate_Voc(Zall, num_antennas, exc, matB, antennes_coords)
        p_opt= calculate_powers(res_opt)

        # CSV after
        powers_after_csv = os.path.join(save_dir, f"{scenario_name}_{geometry_id}_powers_after.csv")
        with open(powers_after_csv,'w',newline='') as f:
            w=csv.writer(f)
            w.writerow(["Dipole","P_V_oc(uW)","P_Vin_oc(uW)","P_in(uW)","P_L(uW)"])
            for i in range(num_antennas):
                w.writerow([i,
                            p_opt['P_V_oc'][i],
                            p_opt['P_Vin_oc'][i],
                            p_opt['P_in'][i],
                            p_opt['P_L'][i] ])

        arr_aft = [
           p_opt['P_V_oc'], p_opt['P_Vin_oc'], p_opt['P_in'], p_opt['P_L']
        ]
        fig_aft, _ = visualize_powers(
           arr_aft, labels,
           f"Puissances après GA ({scenario_name})",
           np.arange(num_antennas),
           antennes_coords,
           show=False
        )
        out_after = os.path.join(save_dir, f"{scenario_name}_{geometry_id}_powers_after.png")
        plt.savefig(out_after, dpi=300)
        plt.close(fig_aft)

        # Impédances optimisées
        csv_imp = os.path.join(save_dir, f"{scenario_name}_{geometry_id}_impedances.csv")
        with open(csv_imp,'w',newline='') as f:
            w=csv.writer(f)
            w.writerow(["dipole_index","R","X","optim_time_s"])
            for i,z_ in enumerate(final_loads):
                w.writerow([i, z_.real, z_.imag, dt_])

        print("Optimisation terminée.")


def run_scenario_continu_reflectors(antennes_coords, scenario_name, geometry_id,
                                    save_dir, plot_geometry=True):
    """
    Emetteur => -jIm(Zin_em)
    Récepteur => conj(Zin_rx)
    Reflecteurs => GA continu
    """
    num_ant = len(antennes_coords)
    print(f"\n=== [CONTINU_reflectors] {scenario_name}, {geometry_id} ({num_ant} antennes) ===")

    if plot_geometry:
        out_geo = os.path.join(save_dir, f"{scenario_name}_{geometry_id}_continuRef_geometry.png")
        plot_antennas(antennes_coords, lam=lam, filename=out_geo)

    # before
    Zall = calculate_impedances(num_ant, antennes_coords)
    exc = tension_Oc_vector(num_ant, antennes_coords,'auto')
    matA= impedance_matrix(num_ant, antennes_coords,'auto')
    resA,_= calculate_Voc(Zall, num_ant, exc, matA, antennes_coords)
    pA= calculate_powers(resA)

    start_= time.time()
    final_loads, best_score= optimize_reflector_only_continu(antennes_coords)
    dt_ = time.time()-start_
    print(f"   => best_score={best_score:.3f}, dt={dt_:.2f}s")

    matB= impedance_matrix(num_ant, antennes_coords,'manuel', final_loads)
    resB,_= calculate_Voc(Zall, num_ant, exc, matB, antennes_coords)
    pB= calculate_powers(resB)

    labels = ["$P_{V_{oc}}$","$P_{V_{in_{oc}}}$","$P_{in}$","$P_{L}$"]
    arr_bef = [
        pA['P_V_oc'], pA['P_Vin_oc'], pA['P_in'], pA['P_L']
    ]
    arr_aft = [
        pB['P_V_oc'], pB['P_Vin_oc'], pB['P_in'], pB['P_L']
    ]

    fig_bef,_ = visualize_powers(
        arr_bef, labels,
        f"[CONTINU_Reflect] {scenario_name}-{geometry_id} BEFORE",
        np.arange(num_ant),
        antennes_coords,
        show=False
    )
    out_bef = os.path.join(save_dir, f"{scenario_name}_{geometry_id}_continuReflect_powers_before.png")
    plt.savefig(out_bef, dpi=300)
    plt.close(fig_bef)

    fig_aft,_ = visualize_powers(
        arr_aft, labels,
        f"[CONTINU_Reflect] {scenario_name}-{geometry_id} AFTER",
        np.arange(num_ant),
        antennes_coords,
        show=False
    )
    out_aft = os.path.join(save_dir, f"{scenario_name}_{geometry_id}_continuReflect_powers_after.png")
    plt.savefig(out_aft, dpi=300)
    plt.close(fig_aft)


def run_scenario_discret_reflectors(antennes_coords, scenario_name, geometry_id,
                                    N, save_dir, plot_geometry=True):
    """
    Reflecteurs => réactance prise dans liste discrète
    """
    num_ant = len(antennes_coords)
    print(f"\n=== [DISCRET_reflectors(N={N})] {scenario_name}, {geometry_id} ({num_ant} antennes) ===")

    if plot_geometry:
        out_geo = os.path.join(save_dir, f"{scenario_name}_{geometry_id}_discretRef_N{N}_geometry.png")
        plot_antennas(antennes_coords, lam=lam, filename=out_geo)

    Zall = calculate_impedances(num_ant, antennes_coords)
    exc  = tension_Oc_vector(num_ant, antennes_coords,'auto')
    matA = impedance_matrix(num_ant, antennes_coords,'auto')
    resA,_ = calculate_Voc(Zall, num_ant, exc, matA, antennes_coords)
    pA= calculate_powers(resA)

    reac_list= generate_discrete_reactances(N)
    start_= time.time()
    best_loads, bestP= optimize_reflector_only_discrete(antennes_coords, reac_list)
    dt_ = time.time()-start_
    if isinstance(bestP, np.ndarray):
        bestP_val = bestP.flat[0]
    else:
        bestP_val = float(bestP)

    print(f"   => bestP={bestP_val:.3f}, dt={dt_:.2f}s")

    matB= impedance_matrix(num_ant, antennes_coords,'manuel', best_loads)
    resB,_= calculate_Voc(Zall, num_ant, exc, matB, antennes_coords)
    pB= calculate_powers(resB)

    csv_imp = os.path.join(save_dir, f"{scenario_name}_{geometry_id}_discretRef_N{N}_impedances.csv")
    with open(csv_imp,'w',newline='') as ff:
        w=csv.writer(ff)
        w.writerow(["ant_id","ZL_opt_real","ZL_opt_imag","best_score","elapsed_time_s"])
        for i,zL in enumerate(best_loads):
            w.writerow([i, zL.real, zL.imag, bestP, dt_])

    labels= ["$P_{V_{oc}}$","$P_{V_{in_{oc}}}$","$P_{in}$","$P_{L}$"]
    arr_bef= [
        pA["P_V_oc"], pA["P_Vin_oc"], pA["P_in"], pA["P_L"]
    ]
    arr_aft= [
        pB["P_V_oc"], pB["P_Vin_oc"], pB["P_in"], pB["P_L"]
    ]

    fig_bef,_= visualize_powers(
        arr_bef, labels,
        f"[DISCRET_Reflect(N={N})] {scenario_name}_{geometry_id} => BEFORE",
        np.arange(num_ant),
        antennes_coords,
        show=False
    )
    out_bef = os.path.join(save_dir, f"{scenario_name}_{geometry_id}_discretRef_N{N}_powers_before.png")
    plt.savefig(out_bef, dpi=300)
    plt.close(fig_bef)

    fig_aft,_= visualize_powers(
        arr_aft, labels,
        f"[DISCRET_Reflect(N={N})] {scenario_name}_{geometry_id} => AFTER",
        np.arange(num_ant),
        antennes_coords,
        show=False
    )
    out_aft = os.path.join(save_dir, f"{scenario_name}_{geometry_id}_discretRef_N{N}_powers_after.png")
    plt.savefig(out_aft, dpi=300)
    plt.close(fig_aft)


###############################################################################
# MAIN
###############################################################################
def main():
    print("Début de l'exécution du programme principal...")

    save_dir = r"C:\Documents\python-necpp\PyNEC\example\project\structurationned_code\full_wave_approach\result"
    os.makedirs(save_dir, exist_ok=True)

    from constants import lam, half_length
  
    ###########################################################################
    # SCÉNARIOS "Classiques"
    ###########################################################################
    
    manual_defs_0 = [
        (0, 0, -half_length, 0, 0, half_length, 'emitter'),
        (9.5*lam, 0, -half_length, 9.5*lam, 0, half_length, 'receiver'),
        #(0.3*lam, 0, -half_length, 0.3*lam, 0, half_length, 'reflector'),
        
    ]

    # On génère la liste d'antennes
    oords_man0 = generate_manual_antenna_coords(manual_defs_0)

    # 1) GA global
    run_scenario(oords_man0, "manuel", "test_perf", save_dir,
                 plot_geometry=True, optimize=True)
    # 2) reflect_all
    run_scenario_reflect_all(oords_man0, "manuel", "test_perf", save_dir,
                             plot_geometry=True)
    # 3) reflectors => continu
    run_scenario_continu_reflectors(oords_man0, "manuel", "test_perf", save_dir,
                                    plot_geometry=True)
    # 4) reflectors => discret(N=3)
    run_scenario_discret_reflectors(oords_man0, "manuel", "test_perf", N=3,
                                    save_dir=save_dir, plot_geometry=True)
    # 5) reflectors => discret(N=5)
    run_scenario_discret_reflectors(oords_man0, "manuel", "test_perf", N=5,
                                    save_dir=save_dir, plot_geometry=True)
    """
    manual_defs_1 = [
        (0, 0, -half_length, 0, 0, half_length, 'emitter'),
        (10*lam, 0, -half_length, 10*lam, 0, half_length, 'receiver'),
        (2*lam, 0, -half_length, 2*lam, 0, half_length, 'reflector'),
        (2*lam, 1*lam, -half_length, 2*lam, 1*lam, half_length, 'reflector'),
        (2*lam, -3*lam, -half_length, 2*lam,  -3*lam, half_length, 'reflector'),
    ]

    # On génère la liste d'antennes
    oords_man1 = generate_manual_antenna_coords(manual_defs_1)

    # 1) GA global
    run_scenario(oords_man1, "manuel", "test_perf", save_dir,
                 plot_geometry=True, optimize=True)
    # 2) reflect_all
    run_scenario_reflect_all(oords_man1, "manuel", "test_perf", save_dir,
                             plot_geometry=True)
    # 3) reflectors => continu
    run_scenario_continu_reflectors(oords_man1, "manuel", "test_perf", save_dir,
                                    plot_geometry=True)
    # 4) reflectors => discret(N=3)
    run_scenario_discret_reflectors(oords_man1, "manuel", "test_perf", N=3,
                                    save_dir=save_dir, plot_geometry=True)
    # 5) reflectors => discret(N=5)
    run_scenario_discret_reflectors(oords_man1, "manuel", "test_perf", N=5,
                                    save_dir=save_dir, plot_geometry=True)
   """
    # SPHERE #1
    coords_sphere_1 = generate_random_antenna_coords_3D_sphere(5, lam, 2*lam)
    
    run_scenario(coords_sphere_1, "sphère", "sphere_5ant_2lam", save_dir, plot_geometry=True, optimize=True)
    run_scenario_reflect_all(coords_sphere_1, "sphère", "sphere_5ant_2lam", save_dir, plot_geometry=True)
    run_scenario_continu_reflectors(coords_sphere_1, "sphère", "sphere_5ant_2lam", save_dir, plot_geometry=True)
    run_scenario_discret_reflectors(coords_sphere_1, "sphère", "sphere_5ant_2lam", N=3, save_dir=save_dir, plot_geometry=True)
    run_scenario_discret_reflectors(coords_sphere_1, "sphère", "sphere_5ant_2lam", N=5, save_dir=save_dir, plot_geometry=True)
    
    
    # CIRCULAIRE #1
    coords_circ_1 = generate_circular_antenna_array(num_antennas=5, radius=2*lam)
    
    run_scenario(coords_circ_1, "circulaire", "circ_5ant_2lam", save_dir, plot_geometry=True, optimize=True)
    run_scenario_reflect_all(coords_circ_1, "circulaire", "circ_5ant_2lam", save_dir, plot_geometry=True)
    run_scenario_continu_reflectors(coords_circ_1, "circulaire", "circ_5ant_2lam", save_dir, plot_geometry=True)
    run_scenario_discret_reflectors(coords_circ_1, "circulaire", "circ_5ant_2lam", N=3, save_dir=save_dir, plot_geometry=True)
    run_scenario_discret_reflectors(coords_circ_1, "circulaire", "circ_5ant_2lam", N=5, save_dir=save_dir, plot_geometry=True)
    
    
    # CARRE #1
    coords_square_1 = generate_square_array(lam=lam, d=2*lam)
    run_scenario(coords_square_1, "carré", "carre_2lam_base", save_dir, plot_geometry=True, optimize=True)
    run_scenario_reflect_all(coords_square_1, "carré", "carre_2lam_base", save_dir, plot_geometry=True)
    run_scenario_continu_reflectors(coords_square_1, "carré", "carre_2lam_base", save_dir, plot_geometry=True)
    run_scenario_discret_reflectors(coords_square_1, "carré", "carre_2lam_base", N=3, save_dir=save_dir, plot_geometry=True)
    run_scenario_discret_reflectors(coords_square_1, "carré", "carre_2lam_base", N=5, save_dir=save_dir, plot_geometry=True)
    
    
    # YAGI #1
    coords_yagi_1 = generate_linear_yagi_like_array(num_reflectors=5, lam=lam,
                                                    spacing=lam/2, receiver_distance=5.0)
   
    run_scenario(coords_yagi_1, "yagi-like", "yagi_5ref_lambda2", save_dir, plot_geometry=True, optimize=True)
    run_scenario_reflect_all(coords_yagi_1, "yagi-like", "yagi_5ref_lambda2", save_dir, plot_geometry=True)
    run_scenario_continu_reflectors(coords_yagi_1, "yagi-like", "yagi_5ref_lambda2", save_dir, plot_geometry=True)
    run_scenario_discret_reflectors(coords_yagi_1, "yagi-like", "yagi_5ref_lambda2", N=3, save_dir=save_dir, plot_geometry=True)
    run_scenario_discret_reflectors(coords_yagi_1, "yagi-like", "yagi_5ref_lambda2", N=5, save_dir=save_dir, plot_geometry=True)
    
   
    
    ############################################################################
    # SCÉNARIO "square_separate" (FIG.7) => On applique AUSSI nos 5 méthodes
    ############################################################################
     # FIG.7 param
    R = 20.0
    #d_values_fig7 = [(2/3)*lam, 1*lam, (3/2)*lam, 2*lam, (5/2)*lam, 3*lam]
    #d_values_fig7 = [ 1*lam, (3/2)*lam, 2*lam, (5/2)*lam, 3*lam]
    d_values_fig7 = [(2/3)*lam]
    phi_values = np.linspace(0,2*math.pi,3)
    scenario_name = "square_separate"

    def create_antennas(d, phi):
        Cx, Cy = R, 0.0
        base_positions = [
            (Cx + d/2, Cy + d/2),
            (Cx - d/2, Cy + d/2),
            (Cx - d/2, Cy - d/2),
            (Cx + d/2, Cy - d/2),
        ]
        rotated_positions = []
        for (x0,y0) in base_positions:
            xR= Cx + (x0-Cx)*math.cos(phi) - (y0-Cy)*math.sin(phi)
            yR= Cy + (x0-Cx)*math.sin(phi) + (y0-Cy)*math.cos(phi)
            rotated_positions.append((xR,yR))

        emitter_coords= (0,0,-half_length,0,0,half_length)
        ants = [{'coords': emitter_coords, 'type':'emitter'}]
        ttypes= ['receiver','reflector','reflector','reflector']
        for i,(xx,yy) in enumerate(rotated_positions):
            c_=(xx,yy,-half_length,xx,yy,half_length)
            ants.append({'coords':c_,'type':ttypes[i]})
        return ants

    # Juste "référence sans reflectors"
    def create_reference_antennas(d, phi):
        # 2 antennes : emitter, receiver => pas de reflectors
        full= create_antennas(d, phi)
        # On coupe la liste => 2 antennes
        #   Emetteur => index=0,    Receiver => index=1
        return full[:2]

    def get_power_GA(antennes_coords, do_optimize=False):
    
        from optimization import optimize_impedances

        numA= len(antennes_coords)
        Zall= calculate_impedances(numA, antennes_coords)
        exc = tension_Oc_vector(numA, antennes_coords,'auto')
        matA= impedance_matrix(numA, antennes_coords,'auto')
        resA,_= calculate_Voc(Zall,numA,exc,matA,antennes_coords)
        pA = calculate_powers(resA)

        rx_idx=None
        for i,ant in enumerate(antennes_coords):
            if ant['type']=='receiver':
                rx_idx=i
                break
        if rx_idx is None:
            # Pas de receiver => bizarre => on renvoie None
            return None, None, 0.0, None

        p_before = pA["P_L"][rx_idx].real
        if not do_optimize:
            return p_before, None, 0.0, None

        # GA
        start_ = time.time()
        sol = optimize_impedances(antennes_coords)
        end_  = time.time()
        dt_   = end_-start_
        if sol is None:
            return p_before, None, dt_, None

        # Applique
        loaded=[]
        for i in range(0, len(sol), 2):
            R_ = sol[i]
            X_ = sol[i+1]
            loaded.append(complex(R_,X_))

        matB= impedance_matrix(numA, antennes_coords,'manuel', loaded)
        resB,_= calculate_Voc(Zall,numA,exc,matB,antennes_coords)
        pB= calculate_powers(resB)
        p_after = pB["P_L"][rx_idx].real
        return p_before, p_after, dt_, loaded

    ############################################################################
    # On fait la boucle "square_separate" => GA, ReflectAll, reflectors=>continu,
    #                                       reflectors=>discret(N=3 ou 5)
    ############################################################################

    for d_ in d_values_fig7:
        print(f"\n=== SCénario {scenario_name}, d={d_/lam:.2f} lambda ===")
        d_subdir = os.path.join(save_dir, f"{scenario_name}_d_{d_/lam:.2f}")
        os.makedirs(d_subdir, exist_ok=True)
        print(d_)
        phi_deg_values = np.degrees(phi_values)

        # On stocke 7 courbes:
        #   [0] => Sans reflectors (2 antennes)
        #   [1] => GA-bef
        #   [2] => GA-aft
        #   [3] => ReflectAll
        #   [4] => reflectors =>Continu
        #   [5] => reflectors =>Discret(N=3)
        #   [6] => reflectors =>Discret(N=5)
        arr_0 = []
        arr_1 = []
        arr_2 = []
        arr_3 = []
        arr_4 = []
        arr_5 = []
        arr_6 = []

        csvfile = os.path.join(d_subdir, f"{scenario_name}_d_{d_/lam:.2f}_impedances.csv")
        with open(csvfile,'w', newline='') as ff:
            w= csv.writer(ff)
            w.writerow(["phi(deg)","GA_optTime","GA_R[i]","GA_X[i]",
                        "ZL_reflectAll","ZL_reflectContinu",
                        "ZL_reflectDiscret_N3","ZL_reflectDiscret_N5"])

            for phi in phi_values:
                print(phi)
                # 0) "Sans reflectors" => 2 antennes
                ref2 = create_reference_antennas(d_, phi)
                n2= len(ref2)
                Z2= calculate_impedances(n2, ref2)
                exc2= tension_Oc_vector(n2, ref2,'auto')
                mat2= impedance_matrix(n2, ref2,'auto')
                r2,_= calculate_Voc(Z2,n2,exc2,mat2,ref2)
                p2= calculate_powers(r2)
                arr_0.append( p2["P_L"][1].real if len(p2["P_L"])>1 else np.nan )

                # 1/2) GA => full(4 antennes)
                full4 = create_antennas(d_, phi)
                p_bef, p_aft, dt_g, loads_g = get_power_GA(full4, do_optimize=True)
                arr_1.append( p_bef if p_bef is not None else np.nan )
                arr_2.append( p_aft if p_aft is not None else np.nan )

                # 3) reflectAll
                if loads_g is None:
                    loads_g = []
                coords4 = create_antennas(d_, phi)
                results_refA= scenario_reflect_all(coords4)
                p_refA= calculate_powers(results_refA)
                arr_3.append( p_refA["P_L"][1].real if len(p_refA["P_L"])>1 else np.nan )
                zlrA = results_refA.get("ZL_reflector", None)

                # 4) reflectors => Continu (version locale)
                from optimization import optimize_reflector_only_continu
                finalC, bestC= optimize_reflector_only_continu(coords4)
                matC= impedance_matrix(len(coords4), coords4,'manuel', finalC)
                Zc = calculate_impedances(len(coords4), coords4)
                VocC = tension_Oc_vector(len(coords4), coords4,'auto')
                resC,_= calculate_Voc(Zc, len(coords4), VocC, matC, coords4)
                pC= calculate_powers(resC)
                arr_4.append( pC["P_L"][1].real if len(pC["P_L"])>1 else np.nan )

                zlrC = None
                for i_,ant_ in enumerate(coords4):
                    if ant_['type']=='reflector':
                        zlrC = finalC[i_]
                        break

                # 5) reflectors => Discret N=3
                from optimization import optimize_reflector_only_discrete, generate_discrete_reactances
                reac3 = generate_discrete_reactances(3)
                bestLoad3, bestP3 = optimize_reflector_only_discrete(coords4, reac3)
                mat3= impedance_matrix(len(coords4),coords4,'manuel',bestLoad3)
                res3,_= calculate_Voc(Zc, len(coords4), VocC, mat3, coords4)
                p3= calculate_powers(res3)
                arr_5.append( p3["P_L"][1].real if len(p3["P_L"])>1 else np.nan )

                zlr3 = None
                for i_,ant_ in enumerate(coords4):
                    if ant_['type']=='reflector':
                        zlr3 = bestLoad3[i_]
                        break

                # 6) reflectors => Discret N=5
                reac5 = generate_discrete_reactances(5)
                bestLoad5, bestP5 = optimize_reflector_only_discrete(coords4, reac5)
                mat5= impedance_matrix(len(coords4),coords4,'manuel',bestLoad5)
                res5,_= calculate_Voc(Zc, len(coords4), VocC, mat5, coords4)
                p5= calculate_powers(res5)
                arr_6.append( p5["P_L"][1].real if len(p5["P_L"])>1 else np.nan )

                zlr5 = None
                for i_,ant_ in enumerate(coords4):
                    if ant_['type']=='reflector':
                        zlr5 = bestLoad5[i_]
                        break

                # Ecriture CSV
                phi_deg = np.degrees(phi)
                ga_reflect_str = ""
                if loads_g:
                    for i_,ant_ in enumerate(coords4):
                        if ant_['type']=='reflector' and i_< len(loads_g):
                            ga_reflect_str += f"({loads_g[i_].real:.1f}+j{loads_g[i_].imag:.1f})"

                w.writerow([
                   f"{phi_deg:.2f}",
                   f"{dt_g:.4f}",
                   ga_reflect_str,
                   "",
                   zlrA,
                   zlrC,
                   zlr3,
                   zlr5
                ])

        plt.figure(figsize=(8,6))
        plt.plot(phi_deg_values, arr_0, '--', color='red', label="Sans réflecteurs")
        plt.plot(phi_deg_values, arr_1, color='blue',  label="Avec réflecteurs (GA-bef)")
        plt.plot(phi_deg_values, arr_2, color='green', label="Avec réflecteurs (GA-aft)")
        plt.plot(phi_deg_values, arr_3, color='magenta', label="ReflectAll")
        plt.plot(phi_deg_values, arr_4, color='orange', label="Reflecteurs(continu)")
        plt.plot(phi_deg_values, arr_5, color='brown', label="Reflecteurs(discret N=3)")
        plt.plot(phi_deg_values, arr_6, color='gray', label="Reflecteurs(discret N=5)")

        plt.xlabel("Angle φ (degrés)")
        plt.ylabel("Puissance reçue (Tag#1) [uW?]")
        plt.title(f"Puissance vs φ, d={d_/lam:.2f} lambda\n[GA global, ReflectAll, cont, discret(3/5)]")
        plt.grid(True)
        plt.legend()

        outfig = os.path.join(d_subdir, f"{scenario_name}_d_{d_/lam:.2f}_allOptim.png")
        plt.savefig(outfig, dpi=300)
        plt.close()

    print("Programme terminé.")


if __name__=="__main__":
    main()
