########################################
# test_3_antennas_varying_distance.py
########################################

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

from constants import lam, half_length
from geometry import generate_manual_antenna_coords
from simulation import (
    calculate_impedances,
    tension_Oc_vector,
    impedance_matrix,
    calculate_Voc,
    calculate_powers
)
# Algorithmes d’optimisation (anciens + nouveaux)
from optimization import (
    # Ancien GA (optimise tout)
    optimize_reflector_only_continu,
    generate_discrete_reactances,
    optimize_reflector_only_discrete # GA : n’optimise que le réflecteur
)

from plotting import plot_antennas


###############################################
# SCÉNARIOS
###############################################

def scenarioA_imZin(dist_em_rx=20.0):
    """
    Scénario A : 2 dipôles (ém + rx) SANS réflecteur.
    Emetteur => -j Im(Zin_em), Récepteur => conj(Zin_rx).
    """
    ants_def = [
       (0.0, 0.0, -half_length,
        0.0, 0.0, +half_length, 'emitter'),
       (dist_em_rx, 0.0, -half_length,
        dist_em_rx, 0.0, +half_length, 'receiver')
    ]
    coords = generate_manual_antenna_coords(ants_def)
    n = len(coords)

    Zall = calculate_impedances(n, coords)
    Voc_ = tension_Oc_vector(n, coords, reglage='auto')

    # Pass 1 => lire Zin_em
    em_idx, rx_idx = 0, 1
    loadsP1 = [complex(0,0), complex(1e9,1e9)]
    matP1 = impedance_matrix(n, coords, 'manuel', loadsP1)
    resP1, _ = calculate_Voc(Zall, n, Voc_, matP1, coords)
    Zin_em = resP1['zin'][em_idx]

    # Pass 2 => lire Zin_rx
    loadsP2 = [complex(1e9,1e9), complex(0,0)]
    tensions2 = np.zeros(n, dtype=complex)
    tensions2[rx_idx] = 1.0
    matP2 = impedance_matrix(n, coords, 'manuel', loadsP2)
    resP2, _ = calculate_Voc(Zall, n, tensions2.reshape(-1,1), matP2, coords)
    Zin_rx = resP2['zin'][rx_idx]

    # Final
    final_loads = [complex(0, -Zin_em.imag), np.conjugate(Zin_rx)]
    matFin = impedance_matrix(n, coords, 'manuel', final_loads)
    VocFin = tension_Oc_vector(n, coords, 'auto')
    resFin, _ = calculate_Voc(Zall, n, VocFin, matFin, coords)
    return coords, resFin


def scenarioB_ga_reflector_only(dist_em_rx=20.0, d_lam=1.0, angle_deg=90.0):
    """
    Scenario B2: 3 dipôles (ém, rx, réflecteur).
      Emetteur => -jIm(Zin_em),
      Récepteur => conj(Zin_rx),
      Réflecteur => GA (continu)
    
    IMPORTANT :
    - On place le réflecteur du côté *Récepteur*.
      => xRef = dist_em_rx + r_ref*cos(angle_rad)
      => yRef = 0          + r_ref*sin(angle_rad)
    """
    angle_rad= math.radians(angle_deg)
    ants_def= [
      (0.0, 0.0,       -half_length,
       0.0, 0.0,       +half_length, 'emitter'),
      (dist_em_rx,0.0, -half_length,
       dist_em_rx,0.0, +half_length, 'receiver')
    ]
    r_ref= d_lam*lam
    x_rf= dist_em_rx + r_ref*math.cos(angle_rad)
    y_rf= 0.0        + r_ref*math.sin(angle_rad)
    ants_def.append((
      x_rf,y_rf,-half_length,
      x_rf,y_rf,+half_length,'reflector'
    ))
    coords= generate_manual_antenna_coords(ants_def)
    n= len(coords)

    final_loads, _ = optimize_reflector_only_continu(coords)
    n= len(coords)
    Zall= calculate_impedances(n, coords)
    Voc_= tension_Oc_vector(n, coords,'auto')
    mat= impedance_matrix(n, coords,'manuel', final_loads)
    results,_= calculate_Voc(Zall,n,Voc_,mat,coords)

    # On stocke la charge du réflecteur
    refl_idx = None
    for i,a in enumerate(coords):
        if a['type']=='reflector':
            refl_idx= i
            break
    if refl_idx is not None:
        results["ZL_reflector"] = final_loads[refl_idx]

    return coords, results

def scenarioC_discrete_reflector_only(dist_em_rx=20.0, d_lam=1.0, angle_deg=90.0, N=4):
    """
    Scenario C2: 3 dipôles (ém, rx, réflecteur).
      Emetteur => -jIm(Zin_em),
      Récepteur => conj(Zin_rx),
      Réflecteur => discret

    On place le réflecteur à partir du récepteur :
       xRef = dist_em_rx + r_ref*cos(angle_deg)
       yRef = 0          + r_ref*sin(angle_deg)
    """
    angle_rad= math.radians(angle_deg)
    ants_def= [
      (0.0, 0.0,       -half_length,
       0.0, 0.0,       +half_length, 'emitter'),
      (dist_em_rx,0.0, -half_length,
       dist_em_rx,0.0, +half_length, 'receiver')
    ]
    r_ref= d_lam*lam
    x_rf= dist_em_rx + r_ref*math.cos(angle_rad)
    y_rf= 0.0        + r_ref*math.sin(angle_rad)
    ants_def.append((
      x_rf,y_rf,-half_length,
      x_rf,y_rf,+half_length,'reflector'
    ))
    coords= generate_manual_antenna_coords(ants_def)
    n= len(coords)

    # Liste de réactances discrètes
    reac_list= generate_discrete_reactances(N)
    best_loads, _= optimize_reflector_only_discrete(coords, reac_list)

    n= len(coords)
    Zall= calculate_impedances(n, coords)
    Voc_= tension_Oc_vector(n, coords,'auto')
    mat= impedance_matrix(n, coords,'manuel', best_loads)
    results,_= calculate_Voc(Zall,n,Voc_,mat,coords)

    # Stocker la charge du réflecteur
    refl_idx= None
    for i,a in enumerate(coords):
        if a['type']=='reflector':
            refl_idx= i
            break
    if refl_idx is not None:
        results["ZL_reflector"] = best_loads[refl_idx]

    return coords, results

def plot_zin_3antennas_for_scenario(scn_data, scenario_name, d_values, figfile):
    """
    Trace, sur 1 figure, Re(Zin) et Im(Zin) pour les 3 antennes (ém, rx, réflecteur).
    On suppose scn_data["Zin"] est un array de shape (Npoints, 3),
    c’est-à-dire Npoints = len(d_values) et 3 = nb d’antennes.
    """
    # scn_data["Zin"] doit exister
    Zin_array = scn_data["Zin"]  # shape (Npoints, 3)
    if Zin_array.shape[1] < 3:
        print(f"Attention: Le scénario {scenario_name} n’a pas 3 antennes. (shape={Zin_array.shape})")
        # On continue quand même, mais il n’y aura que 2 ou 1 antenne(s).

    # Création figure
    fig, (ax_re, ax_im) = plt.subplots(1, 2, figsize=(10,4))
    fig.suptitle(f"Zin pour {scenario_name} : 3 antennes", fontsize=12)

    # Couleurs ou styles pour antennes
    color_list = ['red','green','blue']
    label_list = ['Émetteur','Récepteur','Réflecteur']
    print ("bb")
    print (Zin_array)
    print ("bb")
    # On boucle sur i_antenne = 0..2
    nb_ants = Zin_array.shape[1]  # si =3, on aura 3 boucles
    for ant_i in range(nb_ants):
        Re_ = Zin_array[:, ant_i].real
        Im_ = Zin_array[:, ant_i].imag
        # Couleur
        c_ = color_list[ant_i] if ant_i<len(color_list) else 'black'
        lbl_ = label_list[ant_i] if ant_i<len(label_list) else f"Antenne#{ant_i+1}"
        
        ax_re.plot(d_values, Re_, color=c_, marker='o', ms=0.01,
                   lw=0.8, label=lbl_)
        ax_im.plot(d_values, Im_, color=c_, marker='o', ms=0.01,
                   lw=0.8, label=lbl_)

    ax_re.set_title("Re(Zin)", fontsize=10)
    ax_re.set_xlabel("d / λ", fontsize=9)
    ax_re.set_ylabel("Ohms", fontsize=9)
    ax_re.grid(True, linestyle=':')
    ax_re.legend(fontsize=8)

    ax_im.set_title("Im(Zin)", fontsize=10)
    ax_im.set_xlabel("d / λ", fontsize=9)
    ax_im.set_ylabel("Ohms", fontsize=9)
    ax_im.grid(True, linestyle=':')
    ax_im.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(figfile, dpi=300)
    plt.close(fig)

def scenarioReflectAll(dist_em_rx=20.0, d_lam=1.0, angle_deg=90.0):
    """
    Scenario: 3 dipôles (ém, rx, ref).
      Emetteur => -jIm(Zin_em),
      Rx => conj(Zin_rx),
      Reflecteur => -jIm(Zin_ref)
    
    Placement du réflecteur autour du récepteur.
    """
    angle_rad= math.radians(angle_deg)
    ants_def= [
      (0.0, 0.0,       -half_length,
       0.0, 0.0,       +half_length, 'emitter'),
      (dist_em_rx,0.0, -half_length,
       dist_em_rx,0.0, +half_length, 'receiver')
    ]
    r_ref= d_lam*lam
    x_rf= dist_em_rx + r_ref*math.cos(angle_rad)
    y_rf= 0.0        + r_ref*math.sin(angle_rad)
    ants_def.append((
      x_rf,y_rf,-half_length,
      x_rf,y_rf,+half_length,'reflector'
    ))
    coords= generate_manual_antenna_coords(ants_def)
    n= len(coords)

    n= len(coords)
    Zall= calculate_impedances(n, coords)

    # On "ouvre" le reflecteur
    base_loads=[]
    em_idx= None
    rx_idx= None
    ref_idx= None
    for i,a in enumerate(coords):
        if a['type']=='emitter':
            em_idx= i
            base_loads.append(complex(0,0))
        elif a['type']=='receiver':
            rx_idx= i
            base_loads.append(complex(50,0))
        else:
            ref_idx= i
            base_loads.append(complex(1e9,1e9))

    Voc_= tension_Oc_vector(n, coords,'auto')
    mat_= impedance_matrix(n, coords,'manuel', base_loads)
    resOpen,_= calculate_Voc(Zall,n,Voc_,mat_,coords)
    ZinAll= resOpen['zin']

    Zin_em= ZinAll[em_idx]
    Zin_rx= ZinAll[rx_idx]
    Zin_ref=ZinAll[ref_idx]

    final_loads= base_loads[:]
    final_loads[em_idx] = complex(0, -Zin_em.imag)
    final_loads[rx_idx] = np.conjugate(Zin_rx)
    final_loads[ref_idx]= complex(0, -Zin_ref.imag)

    matFin= impedance_matrix(n, coords,'manuel', final_loads)
    resultsFin,_= calculate_Voc(Zall,n,Voc_,matFin,coords)

    if ref_idx is not None:
        resultsFin["ZL_reflector"]= final_loads[ref_idx]

    return coords, resultsFin
def scenarioC_discrete_imagOnly(dist_em_rx=20.0, d_lam=1.0, angle_deg=90.0, N=4):
    """
    3 dipôles: (ém, rx, réflecteur).
      Emetteur => -jIm(Zin_em), Rx => conj(Zin_rx),
      Réflecteur => X pure imag [0..-400].
    
    Réflecteur placé autour du Rx.
    """
    angle_rad= math.radians(angle_deg)
    ants_def= [
      (0.0, 0.0,       -half_length,
       0.0, 0.0,       +half_length, 'emitter'),
      (dist_em_rx,0.0, -half_length,
       dist_em_rx,0.0, +half_length, 'receiver')
    ]
    r_ref= d_lam*lam
    x_rf= dist_em_rx + r_ref*math.cos(angle_rad)
    y_rf= 0.0        + r_ref*math.sin(angle_rad)
    ants_def.append((
      x_rf,y_rf,-half_length,
      x_rf,y_rf,+half_length,'reflector'
    ))
    coords= generate_manual_antenna_coords(ants_def)
    n= len(coords)

    Zall= calculate_impedances(n, coords)
    base_loads=[]
    em_idx=None
    rx_idx=None
    reflectors_idx=[]
    for i,a in enumerate(coords):
        if a['type']=='emitter':
            em_idx= i
            base_loads.append(complex(0,0))
        elif a['type']=='receiver':
            rx_idx= i
            base_loads.append(complex(50,0))
        else:
            reflectors_idx.append(i)
            base_loads.append(complex(1e9,1e9))

    Voc_= tension_Oc_vector(n, coords,'auto')
    mat_= impedance_matrix(n, coords,'manuel', base_loads)
    resOpen,_= calculate_Voc(Zall,n,Voc_,mat_,coords)
    Zin_= resOpen['zin']

    Zin_em= Zin_[em_idx]
    Zin_rx= Zin_[rx_idx]

    import itertools
    step = 100 / (N - 1)
    Xvals= [-50+(i*step) for i in range(N)]

    bestP= -9999
    bestLoads=None
    for combo in itertools.product(Xvals, repeat=len(reflectors_idx)):
        candidate= base_loads[:]
        candidate[em_idx]= complex(0, -Zin_em.imag)
        candidate[rx_idx]= np.conjugate(Zin_rx)
        idx_=0
        for iRef in reflectors_idx:
            X_= combo[idx_]
            candidate[iRef]= complex(0, X_)
            idx_+=1

        mat2= impedance_matrix(n, coords,'manuel', candidate)
        r2,_= calculate_Voc(Zall,n,Voc_,mat2,coords)
        p2= calculate_powers(r2)
        val= p2['P_L'][rx_idx].real
        if val> bestP:
            bestP= val
            bestLoads= candidate[:]

    matFin= impedance_matrix(n, coords,'manuel', bestLoads)
    resultsFin,_= calculate_Voc(Zall,n,Voc_,matFin,coords)

    if len(reflectors_idx)==1:
        resultsFin["ZL_reflector"]= bestLoads[ reflectors_idx[0] ]
    elif len(reflectors_idx)>1:
        resultsFin["ZL_reflector"]= [ bestLoads[i_] for i_ in reflectors_idx ]

    return coords, resultsFin

###############################################
# EXTRACTION + CSV
###############################################

def extract_4powers_per_antenna(results):
    p = calculate_powers(results)
    voc = p['P_V_oc'].ravel()
    vin = p['P_Vin_oc'].ravel()
    pin = p['P_in'].ravel()
    pl  = p['P_L'].ravel()
    return (voc, vin, pin, pl)


def append_results_to_csv(csvfile, scenario_name, d_val, results):
    keys = [
      "Voc","Vl","Vin","vin_Oc","V","vbis",
      "currents_at_center","zin","impedance_loads","impedance_matrix"
    ]
    ZL_refl = results.get("ZL_reflector", None)
    if ZL_refl is not None:
        keys.append("ZL_reflector")

    with open(csvfile,'a',newline='') as ff:
        w= csv.writer(ff)
        for k_ in keys:
            if k_=="ZL_reflector":
                Zval = results["ZL_reflector"]
                if isinstance(Zval, list):
                    for i, val in enumerate(Zval):
                        w.writerow([scenario_name, d_val, "ZL_reflector", 900+i, val.real, val.imag])
                else:
                    w.writerow([scenario_name, d_val, "ZL_reflector", 999, Zval.real, Zval.imag])
                continue

            arr= results.get(k_, None)
            if arr is None:
                continue
            arr= np.array(arr)
            if arr.ndim==0:
                w.writerow([scenario_name, d_val, k_,1, arr.real, arr.imag])
            elif arr.ndim==1:
                for i,val in enumerate(arr):
                    w.writerow([scenario_name, d_val, k_, i+1, val.real, val.imag])
            elif arr.ndim==2:
                nr, nc = arr.shape
                for r_ in range(nr):
                    for c_ in range(nc):
                        val = arr[r_, c_]
                        w.writerow([
                          scenario_name, d_val,
                          f"{k_}[{r_},{c_}]",
                          r_*1000 + (c_+1),
                          val.real, val.imag
                        ])
            else:
                for i,val in enumerate(arr.flatten()):
                    w.writerow([scenario_name, d_val, k_, i+1, val.real, val.imag])


###############################################
# MAIN
###############################################

def main():
    out_dir = r"C:\Documents\python-necpp\PyNEC\example\project\structurationned_code\full_wave_approach\result_test_3_antennas"
    os.makedirs(out_dir, exist_ok=True)

    # CSV
    csvReflectAll = os.path.join(out_dir, "scenarioReflectAll.csv")
    csvImagSweep  = os.path.join(out_dir, "scenarioC_discreteImag.csv")
    csvA  = os.path.join(out_dir, "scenarioA_imZin.csv")
    csvB2 = os.path.join(out_dir, "scenarioB2_gaReflectOnly.csv")
    csvC2 = os.path.join(out_dir, "scenarioC2_discreteReflectOnly.csv")

    all_csv_files = [csvReflectAll, csvImagSweep, csvA, csvB2, csvC2]
    for c_ in all_csv_files:
        with open(c_,'w', newline='') as ff:
            csv.writer(ff).writerow(["scenario","d(lam)","param","antID","Re","Im"])

    d_values= np.arange(0.01,6.01,0.01)
    dist_em_rx= 20.0
    angle_deg= 90.0

    ############################
    # SCENARIO A
    ############################
    scenario_dataA = {"P_V_oc":[], "P_Vin_oc":[], "P_in":[], "P_L":[], "Zin":[]}
    for d_ in d_values:
        coordsA, resA = scenarioA_imZin(dist_em_rx=dist_em_rx)
        append_results_to_csv(csvA, "A_imZin", d_, resA)
        voc_,vin_,pin_,pl_ = extract_4powers_per_antenna(resA)
        scenario_dataA["P_V_oc"].append(voc_)
        scenario_dataA["P_Vin_oc"].append(vin_)
        scenario_dataA["P_in"].append(pin_)
        scenario_dataA["P_L"].append(pl_)
        scenario_dataA["Zin"].append(resA["zin"])

    for k_ in scenario_dataA:
        scenario_dataA[k_] = np.vstack(scenario_dataA[k_])

    ############################
    # SCENARIO B2
    ############################
    scenario_dataB2 = {"P_V_oc":[], "P_Vin_oc":[], "P_in":[], "P_L":[], "Zin":[], "ZL_reflector":[]}
    for d_ in d_values:
        coordsB2, resB2 = scenarioB_ga_reflector_only(dist_em_rx, d_, angle_deg)
        append_results_to_csv(csvB2, "B2_gaReflectOnly", d_, resB2)
        voc_,vin_,pin_,pl_ = extract_4powers_per_antenna(resB2)
        scenario_dataB2["P_V_oc"].append(voc_)
        scenario_dataB2["P_Vin_oc"].append(vin_)
        scenario_dataB2["P_in"].append(pin_)
        scenario_dataB2["P_L"].append(pl_)

        scenario_dataB2["Zin"].append(resB2["zin"])

        zlr = resB2.get("ZL_reflector", None)
        scenario_dataB2["ZL_reflector"].append(zlr if zlr is not None else complex(0,0))

        figgeom, axgeom = plot_antennas(coordsB2, lam=lam, show=False)
        axgeom.set_title(f"Géom B2, d={d_:.2f}λ", fontsize=9)
        outgeo = os.path.join(out_dir, f"geom_B2_{d_:.2f}.png")
        plt.savefig(outgeo, dpi=150)
        plt.close(figgeom)

    for k_ in scenario_dataB2:
        scenario_dataB2[k_] = np.vstack(scenario_dataB2[k_])

    ############################
    # SCENARIO C2
    ############################
    scenario_dataC2 = {"P_V_oc":[], "P_Vin_oc":[], "P_in":[], "P_L":[], "Zin":[], "ZL_reflector":[]}
    for d_ in d_values:
        coordsC2, resC2 = scenarioC_discrete_reflector_only(dist_em_rx, d_, angle_deg, N=5)
        append_results_to_csv(csvC2, "C2_discreteReflectOnly", d_, resC2)
        voc_,vin_,pin_,pl_ = extract_4powers_per_antenna(resC2)
        scenario_dataC2["P_V_oc"].append(voc_)
        scenario_dataC2["P_Vin_oc"].append(vin_)
        scenario_dataC2["P_in"].append(pin_)
        scenario_dataC2["P_L"].append(pl_)
        scenario_dataC2["Zin"].append(resC2["zin"])

        zlr = resC2.get("ZL_reflector", None)
        scenario_dataC2["ZL_reflector"].append(zlr if zlr is not None else complex(0,0))

        figgeom, axgeom = plot_antennas(coordsC2, lam=lam, show=False)
        axgeom.set_title(f"Géom C2, d={d_:.2f}λ", fontsize=9)
        outgeo = os.path.join(out_dir, f"geom_C2_{d_:.2f}.png")
        plt.savefig(outgeo, dpi=150)
        plt.close(figgeom)

    print(scenario_dataC2["Zin"])
    for k_ in scenario_dataC2:
        scenario_dataC2[k_] = np.vstack(scenario_dataC2[k_])

    ############################
    # SCENARIO ReflectAll
    ############################
    scenario_dataReflect = {"P_V_oc":[], "P_Vin_oc":[], "P_in":[], "P_L":[], "Zin":[], "ZL_reflector":[]}
    for d_ in d_values:
        coordsRef, resRef = scenarioReflectAll(dist_em_rx, d_, angle_deg)
        append_results_to_csv(csvReflectAll, "REFLECT_ALL", d_, resRef)
        voc_, vin_, pin_, pl_ = extract_4powers_per_antenna(resRef)
        scenario_dataReflect["P_V_oc"].append(voc_)
        scenario_dataReflect["P_Vin_oc"].append(vin_)
        scenario_dataReflect["P_in"].append(pin_)
        scenario_dataReflect["P_L"].append(pl_)
        scenario_dataReflect["Zin"].append(resRef["zin"])

        zlr = resRef.get("ZL_reflector", None)
        scenario_dataReflect["ZL_reflector"].append(zlr if zlr is not None else complex(0,0))

        figgeom, axgeom = plot_antennas(coordsRef, lam=lam, show=False)
        axgeom.set_title(f"Géométrie ReflectAll, d={d_:.2f}λ", fontsize=9)
        outgeo = os.path.join(out_dir, f"geom_refAll_{d_:.2f}.png")
        plt.savefig(outgeo, dpi=150)
        plt.close(figgeom)
    print("ccccccccc")
    print(scenario_dataReflect["Zin"])
    print ("cccccccccccc")
    for k_ in scenario_dataReflect:
        scenario_dataReflect[k_] = np.vstack(scenario_dataReflect[k_])
    print("dddd")
    print(scenario_dataReflect["Zin"])
    print ("ddd")
    ############################
    # scenarioC_discrete_imagOnly (N=2..12)
    ############################
    #N_values= [3,4,5,8,11]
    N_values= [4]
    scenario_dataImagN= {}
    for N_ in N_values:
        scenario_dataImagN[N_] = {
           "P_V_oc":[], "P_Vin_oc":[],
           "P_in":[],   "P_L":[],
           "Zin":[],    "ZL_reflector":[]
        }

    for d_ in d_values:
        for N_ in N_values:
            coordsN, resN= scenarioC_discrete_imagOnly(dist_em_rx, d_, angle_deg, N=N_)
            scenario_name= f"Cimag_N{N_}"
            append_results_to_csv(csvImagSweep, scenario_name, d_, resN)
            voc_, vin_, pin_, pl_ = extract_4powers_per_antenna(resN)
            scenario_dataImagN[N_]["P_V_oc"].append(voc_)
            scenario_dataImagN[N_]["P_Vin_oc"].append(vin_)
            scenario_dataImagN[N_]["P_in"].append(pin_)
            scenario_dataImagN[N_]["P_L"].append(pl_)
            scenario_dataImagN[N_]["Zin"].append(resN["zin"])

            zlr = resN.get("ZL_reflector", None)
            scenario_dataImagN[N_]["ZL_reflector"].append(zlr if zlr is not None else complex(0,0))

    for N_ in N_values:
        for k_ in scenario_dataImagN[N_]:
            scenario_dataImagN[N_][k_] = np.vstack(scenario_dataImagN[N_][k_])
    # Exemple d'appel après avoir rempli scenario_dataB2
# scenario_dataB2["Zin"] a une forme (Npoints, 3).
    outfig_b2 = os.path.join(out_dir, "Zin_B2_3antennas.png")
    plot_zin_3antennas_for_scenario(
        scenario_dataB2, "B2_gaReflectOnly",
        d_values, outfig_b2
    )

    # Idem pour scenario_dataC2, scenario_dataReflect, etc.
    outfig_c2 = os.path.join(out_dir, "Zin_C2_3antennas.png")
    plot_zin_3antennas_for_scenario(
        scenario_dataC2, "C2_discreteReflectOnly",
        d_values, outfig_c2
    )

    outfig_refl = os.path.join(out_dir, "Zin_ReflectAll_3antennas.png")
    plot_zin_3antennas_for_scenario(
        scenario_dataReflect, "REFLECT_ALL",
        d_values, outfig_refl
    )

    #################################
    # TRACÉ de PUISSANCE : toutes antennes vs distance
    #################################
    pnames= ["P_V_oc","P_Vin_oc","P_in","P_L"]
    ptitle={
      "P_V_oc":"P_V_oc(uW)",
      "P_Vin_oc":"P_Vin_oc(uW)",
      "P_in":"P_in(uW)",
      "P_L":"P_L(uW)"
    }
    def antenna_label(ant_idx):
        if ant_idx==0: return "Émetteur"
        elif ant_idx==1: return "Récepteur"
        else: return "Réflecteur"

    scenario_labels = {
        'A': scenario_dataA,
        'B2': scenario_dataB2,
        'C2': scenario_dataC2,
        'ReflectAll': scenario_dataReflect
    }
    color_map= {
        'A':'red',
        'B2':'green',
        'C2':'blue',
        'ReflectAll':'purple'
    }

    def plot_all_scenarios_for_antenna(ant_idx, figfile):
        fig, axes= plt.subplots(2,2, figsize=(10,7))
        axes= axes.ravel()
        fig.suptitle(f"Toutes puissances sur Ant#{ant_idx+1} ({antenna_label(ant_idx)})", fontsize=12)
        for i, pnm in enumerate(pnames):
            ax= axes[i]
            ax.set_title(ptitle[pnm], fontsize=10)
            for scnID, scnData in scenario_labels.items():
                arr_= scnData[pnm]
                if ant_idx< arr_.shape[1]:
                    yvals= arr_[:,ant_idx]
                    if scnID=='A':
                        label_ = "A_imZin (2 dipôles)"
                    elif scnID=='B2':
                        label_ = "B2: GA (réflecteur seul)"
                    elif scnID=='C2':
                        label_ = "C2: discret (réflecteur seul)"
                    else:
                        label_ = "ReflectAll"
                    ax.plot(d_values, yvals,
                            color=color_map[scnID],
                            marker='o', ms=0.01,
                            lw=0.8,
                            label=label_)
            ax.set_xlabel("d / λ", fontsize=9)
            ax.set_ylabel("Puissance (uW)", fontsize=9)
            ax.grid(True, alpha=0.8, linestyle=':')
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(figfile, dpi=300)
        plt.close(fig)

    for ant_i in range(3):
        outfig_ = os.path.join(out_dir, f"compare_all_scenarios_ant{ant_i+1}.png")
        plot_all_scenarios_for_antenna(ant_i, outfig_)

    
    #################################
    # TRACÉ de la CHARGE RÉFLECTEUR vs d (B2, C2, ReflectAll)
    #################################
    scenario_labels_refl = {
        'B2': scenario_dataB2,
        'C2': scenario_dataC2,
        'ReflectAll': scenario_dataReflect
    }
    color_map_refl= {
        'B2':'green',
        'C2':'blue',
        'ReflectAll':'purple'
    }

    def plot_reflector_load_vs_d(figfile):
        """
        Compare Re/Im(ZL_reflector) vs d, pour B2, C2, ReflectAll
        """
        fig, (ax1,ax2)= plt.subplots(1,2, figsize=(10,4))
        fig.suptitle("Impédance de charge (réflecteur) vs d", fontsize=12)
        for scnID, scnData in scenario_labels_refl.items():
            Zlr= scnData["ZL_reflector"]  # shape(Npoints,) => 1 reflecteur
            Re_= Zlr.real
            Im_= Zlr.imag
            if scnID=='B2':
                label_ = "B2: GA-réfl. only"
            elif scnID=='C2':
                label_ = "C2: Discret-réfl. only"
            else:
                label_ = "ReflectAll"
            ax1.plot(d_values, Re_,
                     color=color_map_refl[scnID],
                     marker='o', ms=0.01, lw=1,
                     label=label_)
            ax2.plot(d_values, Im_,
                     color=color_map_refl[scnID],
                     marker='o', ms=0.01, lw=1,
                     label=label_)
        ax1.set_title("Re(ZL_reflector)", fontsize=10)
        ax1.set_xlabel("d / λ", fontsize=9)
        ax1.set_ylabel("Ohms", fontsize=9)
        ax1.grid(True, linestyle=':')
        ax1.legend(fontsize=8)

        ax2.set_title("Im(ZL_reflector)", fontsize=10)
        ax2.set_xlabel("d / λ", fontsize=9)
        ax2.set_ylabel("Ohms", fontsize=9)
        ax2.grid(True, linestyle=':')
        ax2.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(figfile, dpi=300)
        plt.close(fig)

    outfigZL= os.path.join(out_dir, "ZL_reflector_B2_C2_refAll.png")
    plot_reflector_load_vs_d(outfigZL)

    #################################
    # scenarioC_discrete_imagOnly => compare N
    #################################
    scenario_dataImagN_pnames= ["P_V_oc","P_Vin_oc","P_in","P_L"]
    rx_index=1

    def plot_rx_for_imagN(pkey, figfile):
        fig, ax = plt.subplots(figsize=(7,5))
        colorlist= ['red','green','blue','magenta','orange','cyan']
        for idx,N_ in enumerate(N_values):
            arr_ = scenario_dataImagN[N_][pkey]
            if rx_index< arr_.shape[1]:
                yvals= arr_[:,rx_index]
                ax.plot(d_values, yvals,
                        color=colorlist[idx], marker='o',
                        ms=0.01, lw=0.8,
                        label=f"N={N_} états imag")
        ax.set_title(f"C_discrete_imagOnly: {pkey} sur Rx vs. N", fontsize=11)
        ax.set_xlabel("d / λ", fontsize=9)
        ax.set_ylabel("Puissance (uW)", fontsize=9)
        ax.grid(True, linestyle=':')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(figfile, dpi=300)
        plt.close(fig)

    for pnm in scenario_dataImagN_pnames:
        outN_ = os.path.join(out_dir, f"compare_imagN_{pnm}.png")
        plot_rx_for_imagN(pnm, outN_)

    def plot_reflector_load_imagN(figfile):
        """
        Compare Re/Im(ZL_reflector) vs d, pour N=2..12 (C_discrete_imagOnly).
        """
        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,4))
        fig.suptitle("Charge reflecteur (C_discrete_imagOnly) vs d, selon N", fontsize=12)
        colorlist= ['red','green','blue','magenta','orange','cyan']
        for idx, N_ in enumerate(N_values):
            Zlr= scenario_dataImagN[N_]["ZL_reflector"]
            Re_= Zlr.real
            Im_= Zlr.imag
            ax1.plot(d_values, Re_,
                     color=colorlist[idx], marker='o',
                     ms=0.01, lw=1, label=f"N={N_}")
            ax2.plot(d_values, Im_,
                     color=colorlist[idx], marker='o',
                     ms=0.01, lw=1, label=f"N={N_}")
        ax1.set_title("Re(ZL_reflector)", fontsize=10)
        ax1.set_xlabel("d / λ", fontsize=9)
        ax1.set_ylabel("Ohms", fontsize=9)
        ax1.grid(True, linestyle=':')
        ax1.legend(fontsize=8)

        ax2.set_title("Im(ZL_reflector)", fontsize=10)
        ax2.set_xlabel("d / λ", fontsize=9)
        ax2.set_ylabel("Ohms", fontsize=9)
        ax2.grid(True, linestyle=':')
        ax2.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(figfile,dpi=300)
        plt.close(fig)

    outfigZL2= os.path.join(out_dir, "ZL_reflector_imagOnly_varN.png")
    plot_reflector_load_imagN(outfigZL2)

    print("Terminé. Regardez .CSV et .PNG dans:", out_dir)


if __name__=="__main__":
    main()
