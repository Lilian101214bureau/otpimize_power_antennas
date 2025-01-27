# optimization.py
"""
Module optimization
===================

Implémente différentes stratégies d'optimisation :
- GA global (optimize_impedances)
- Optim. partielle "reflectors only" en continu (optimize_reflector_only_continu)
- Optim. partielle "reflectors only" en discret (optimize_reflector_only_discrete)
- Génération d'une liste de réactances discrètes (generate_discrete_reactances)

"""
import numpy as np
from scipy.optimize import differential_evolution
from simulation import impedance_matrix, tension_Oc_vector, calculate_Voc, calculate_powers, calculate_impedances
from constants import freq

def objective_function(params, antennes_coords, impedances_simulation):
    """
    Fonction objectif pour le GA global :
    - On assigne (R,X) = params[2*i], params[2*i+1] pour chaque antenne i.
    - On calcule la puissance reçue sur l'antenne de type 'receiver' => on veut la maximiser.
    - On renvoie -P_L_receiver (car le solver minimise).
    - P_L_receiver étant la puissance au niveau de la charge sur le recepteur du tag "recepteur" ( du tag cible sur lequel on veut optimsier la puissance)
    """
    num_antennas = len(antennes_coords)
    impedances = []
    param_index = 0
    for antenna in antennes_coords:
        R = params[param_index]
        X = params[param_index + 1]
        impedances.append(complex(R, X))
        param_index += 2

    impedance_loads = impedance_matrix(num_antennas, antennes_coords, reglage='manuel', impedances_manuel=impedances)
    excitation_voltages = tension_Oc_vector(num_antennas, antennes_coords, reglage='auto')
    results, _  = calculate_Voc(impedances_simulation, num_antennas, excitation_voltages, impedance_loads, antennes_coords)
    powers = calculate_powers(results)

    receiver_index = next((idx for idx, ant in enumerate(antennes_coords) if ant['type'] == 'receiver'), None)
    if receiver_index is None:
        raise ValueError("Aucune antenne de type 'receiver' n'a été trouvée.")

    P_L_receiver = powers['P_L'][receiver_index].real
    return -P_L_receiver



def optimize_impedances(antennes_coords):
    """
    GA global :
    On optimise R + jX pour chaque antenne (y compris émetteur, récepteur, réflecteurs).Afin que la fonction objectif soit minimisé 
    Si j'ai N antenne j'ai donc 2N paramètres à optimiser.  Chaque couple (Ri,Xi).
    L'algorithme utilisé est génératif, c'est le differential_evolution.
    Les bornes permettent de chercher les solutions avec R appartenant (0.0, 400.0) et X appartenant à (-400.0, 400.0) 
    On obtimise totues les impédances de charge sur ( émetteur, récepteur (tag cible), et réflecteur tel que la fonction objectif soit minimale,
    (i.e : -P_L_receiver soit minimale donc P_L_receiver soit maximal)

    Retour
    ------
    solution : np.array
       [R0, X0, R1, X1, ..., R_{N-1}, X_{N-1}]
    """
    num_antennas = len(antennes_coords)
    impedances_simulation = calculate_impedances(num_antennas, antennes_coords)

    # Définir les bornes pour les impédances (R et X pour chaque antenne)
    bounds = [(0.0, 400.0), (-400.0, 400.0)] * num_antennas
    
    # Paramètres ajustés pour un compromis entre qualité de la solution et temps d'exécution
    result = differential_evolution(
        objective_function,
        bounds,
        args=(antennes_coords, impedances_simulation),
        strategy='best1bin',   # Stratégie raisonnable pour converger vers une bonne solution
        maxiter=2000,          # Augmenter un peu le nombre d'itérations par rapport à 1000
        popsize=20,            # Légèrement plus grand que 15 pour une meilleure exploration
        tol=1e-4,              # Tolérance plus stricte que 0.01 sans être trop extrême
        mutation=(0.6, 1.0),   # Mutation un peu plus élevée pour explorer davantage
        recombination=0.8,     # Recombinaison plus élevée que 0.7 pour favoriser le mélange des solutions
        disp=True               # Afficher l’évolution pour surveiller la convergence
    )
    """
    bounds = [(0.0, 1000.0), (-1000.0, 1000.0)] * num_antennas

    result = differential_evolution(
        objective_function,
        bounds,
        args=(antennes_coords, impedances_simulation),
        strategy='best1bin',
        maxiter=1000,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=True
    )
    """
    return result.x
def optimize_reflector_only_continu(antennes_coords):
    """
    Optimise uniquement les 'reflectors' en continu (R + jX),
    en figeant l'impédance de l'émetteur et du récepteur.

    Description
    -----------
    - Dans un premier temps, on "ouvre" les réflecteurs (très grande impédance) pour lire
      le Zin de l'émetteur et du récepteur dans ce contexte (Zin_em, Zin_rx).
    - On fixe alors :
        ZL(emitter)  = -j * Im(Zin_em)
        ZL(receiver) = conj(Zin_rx)
      Ces valeurs sont *a priori* proches du Zin optimal pour émetteur et récepteur.
      *Note* : Au fur et à mesure qu'on optimise les réflecteurs, Zin_em/rx pourrait
      théoriquement changer, mais ici on les fige à cette valeur initiale.
    - On lance ensuite une optimisation (differential_evolution) uniquement sur les
      réflecteurs (chacun ayant une R et X libres). La fonction objectif reste
      la puissance au récepteur, qu'on veut maximiser.

    But
    ---
    Cela permet d'observer l'impact de l'optimisation sur les réflecteurs seuls,
    pour un émetteur et un récepteur qui ne sont *pas* ré-optimisés à chaque itération.
    On veut ainsi "uniformiser" la méthode pour comparer avec d'autres approches quels impacts ont les méthodes d'optimsiation 
    sur les impédances des rélfecteurs pour un émission commune de puissance commune. Et un facteur de transmission de puissance au niveau
    du récepteur commun.

    Retour
    ------
    final_loads : list of complex
        La liste d'impédances (R + jX) pour chaque antenne, incluant le réglage
        imposé à l'émetteur et au récepteur.
    best_score : float
        Valeur de P_L (récepteur) maximum trouvée.
    """
    # 1) local function => calc Z_in
    n = len(antennes_coords)
    Zall = calculate_impedances(n, antennes_coords)
    # On "open" tous les reflectors => emitter=0, rx=50
    base_loads=[]
    em_idx=None
    rx_idx=None
    for i,a in enumerate(antennes_coords):
        if a['type']=='emitter':
            em_idx=i
            base_loads.append(complex(0,0))
        elif a['type']=='receiver':
            rx_idx=i
            base_loads.append(complex(50,0))
        else:
            base_loads.append(complex(1e9,1e9))
    Voc_ = tension_Oc_vector(n, antennes_coords, 'auto')
    mat_ = impedance_matrix(n, antennes_coords, 'manuel', base_loads)
    results,_=calculate_Voc(Zall,n,Voc_, mat_, antennes_coords)
    Zin = results['zin']
    Zin_em = Zin[em_idx]  # => impose => -jIm(Zin_em)
    Zin_rx = Zin[rx_idx]  # => impose => conj(Zin_rx)

    # on construit un "mask" reflectors
    reflectors_idx=[]
    for i,a in enumerate(antennes_coords):
        if a['type']=='reflector':
            reflectors_idx.append(i)
    num_ref = len(reflectors_idx)
    if num_ref==0:
        # pas de reflecteur => on fait rien
        return base_loads, 0.0

    # def bounds
    # ex. R=0..1000, X=-1000..1000
    bnds=[]
    for iref in range(num_ref):
        bnds.append((0.0,400.0))
        bnds.append((-400.0,400.0))

    def objective_function2(params):
        # on reconstruit
        candidate_loads=base_loads[:]
        idxp=0
        for i_ in reflectors_idx:
            R_=params[idxp]
            X_=params[idxp+1]
            candidate_loads[i_] = complex(R_,X_)
            idxp+=2

        # emitter => -j Im(Zin_em)
        # receiver => conj(Zin_rx)
        # c’est déjà dans base_loads ? => Non, fixons-le explicitement
        Xem = -Zin_em.imag
        candidate_loads[em_idx] = complex(0.0,Xem)
        candidate_loads[rx_idx] = np.conjugate(Zin_rx)

        mat2 = impedance_matrix(n, antennes_coords,'manuel',candidate_loads)
        Voc2 = tension_Oc_vector(n, antennes_coords,'auto')
        r2,_= calculate_Voc(Zall,n,Voc2,mat2,antennes_coords)
        p2= calculate_powers(r2)
        # index rx => rx_idx
        val= p2['P_L'][rx_idx].real
        return -val
    
    result = differential_evolution(
            objective_function2,
            bnds,
            strategy='best1bin',
            maxiter=2000,         # Était 300 => on augmente
            popsize=20,          # Était 15 => on augmente
            tol=1e-4,            # Était 1e-3 => un peu plus strict
            mutation=(0.6,1.0),  # Était (0.5,1) => exploration un peu plus large
            recombination=0.8,   # Était 0.7 => favorise plus de mixing
            disp=True
        )
    best = result.x
    print (best)
    best_score = -result.fun
    
    
    
    # reconstruit la solution finale
    final_loads= base_loads[:]
    idxp=0
    for i_ in reflectors_idx:
        R_= best[idxp]
        X_= best[idxp+1]
        final_loads[i_] = complex(R_,X_)
        idxp+=2

    # fix em,rx
    final_loads[em_idx] = complex(0,-Zin_em.imag)
    final_loads[rx_idx] = np.conjugate(Zin_rx)

    return final_loads, best_score


def generate_discrete_reactances(N):
    """
    Génère N valeurs de réactance (strictement imaginaires) uniformément espacées
    entre 0 et -400j, par exemple.
    Ex effectif ici. si N=5 => [-400j, -200j, 0j, 200j, 400j]
    Paramètres
    ----------
    N : int
        Nombre de points (>=2)

    Retour
    ------
    reactances : list of complex
    Ajustez en fonction de votre besoin (intervalle, etc.).
    """
    if N < 2:
        raise ValueError("N doit être >= 2")
    step = 800.0 / (N - 1) #répartition uniforme sur l'intervalle
    reactances = []
    for i in range(N):
        X = - 400+(i * step)    #répartition uniforme sur l'intervalle 
        reactances.append(complex(0.0, X))
    return reactances


def optimize_reflector_only_discrete(antennes_coords, reactances_list):
    """
    Optimisation "reflectors only" mais en discret :
    ------------------------------------------------
    On balaie toutes les réactances possibles (dans reactances_list) pour
    chaque réflecteur, en fixant l'émetteur et le récepteur comme dans
    optimize_reflector_only_continu :

        - Emetteur => -j(Im(Zin(emitter)))
        - Récepteur => conj(Zin(receiver))

    On lit ces Zin(emitter) et Zin(receiver) dans un contexte "réflecteurs ouverts"
    (ou grands), puis on applique ce réglage fixe pendant tout le balayage.

    Paramètres
    ----------
    antennes_coords : list of dict
        Informations sur les antennes.
    reactances_list : list of complex
        Les valeurs discrètes  à tester.

    Retour
    ------
    best_loads : list of complex
        Impédances finales pour chaque antenne (émetteur + récepteur fixés,
        et réflecteurs choisis parmi reactances_list).
    bestP : float
        Puissance maximale atteinte au récepteur trouver par test des chacunes des possibilités.
    """
    num_ant = len(antennes_coords)
    Zall = calculate_impedances(num_ant, antennes_coords)

    emitter_idx = None
    rx_idx = None
    reflectors_idx=[]
    for i,a in enumerate(antennes_coords):
        if a['type']=='emitter':
            emitter_idx = i
        elif a['type']=='receiver':
            rx_idx = i
        else:
            reflectors_idx.append(i)

    # open reflectors => on lit Zin(em), Zin(rx)
    base_loads = []
    for i,a in enumerate(antennes_coords):
        if i==emitter_idx:
            base_loads.append(complex(0,0))
        elif i==rx_idx:
            base_loads.append(complex(50,0))
        else:
            base_loads.append(complex(1e9,1e9))

    from simulation import calculate_Voc
    Voc_ = tension_Oc_vector(num_ant, antennes_coords, 'auto')
    load_mat = impedance_matrix(num_ant, antennes_coords, 'manuel', base_loads)
    ropen,_ = calculate_Voc(Zall, num_ant, Voc_, load_mat, antennes_coords)
    Zin_em = ropen['zin'][emitter_idx]
    Zin_rx = ropen['zin'][rx_idx]

    def fix_em_rx():
        arr_=[None]*num_ant
        for i,a in enumerate(antennes_coords):
            if i==emitter_idx:
                # - jIm(Zin_em)
                arr_[i] = complex(0, -np.imag(Zin_em))
            elif i==rx_idx:
                arr_[i] = np.conjugate(Zin_rx)
            else:
                arr_[i] = None
        return arr_

    bestP = -9999
    best_loads = None

    import itertools
    combos = itertools.product(reactances_list, repeat=len(reflectors_idx))
    for combo in combos:
        c_ = fix_em_rx()
        idx_ = 0
        for iref in reflectors_idx:
            c_[iref] = combo[idx_]
            idx_+=1
        # calc
        mat_ = impedance_matrix(num_ant, antennes_coords, 'manuel', c_)
        r2,_ = calculate_Voc(Zall, num_ant, Voc_, mat_, antennes_coords)
        p2 = calculate_powers(r2)
        val = p2['P_L'][rx_idx].real
        if val>bestP:
            bestP=val
            best_loads=c_[:]

    return best_loads, bestP

