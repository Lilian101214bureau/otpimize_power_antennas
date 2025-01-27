"""
Module simulation
=================

Gère la création du contexte NEC (PyNEC), la configuration de la géométrie,
le calcul des impédances mutuelles et des tensions / courants.

Fonctionnalités principales :
- create_context
- setup_geometry
- calculate_impedances
- tension_Oc_vector
- impedance_matrix
- calculate_Voc (pour calculer tensions & courants)
- calculate_powers
"""


# simulation.py
from PyNEC import nec_context
import numpy as np
import math
from constants import freq, lam, segment_count_impair, half_segment, position_half_segment, radius, half_length

def setup_geometry(context, antennes_coords):
    """
    Construit la géométrie (les fils) dans le contexte NEC.
    """
    geo = context.get_geometry()
    for idx, antenna in enumerate(antennes_coords):
        xw1, yw1, zw1, xw2, yw2, zw2 = antenna['coords']
        geo.wire(
            tag_id=idx + 1,
            segment_count=segment_count_impair,
            xw1=xw1, yw1=yw1, zw1=zw1,
            xw2=xw2, yw2=yw2, zw2=zw2,
            rad=radius, rdel=1, rrad=1
        )
        # Indique à NEC qu'on a terminé la construction de la géométrie.
    context.geometry_complete(0)

def create_context(freq, antennes_coords):
    """
    Crée un contexte NEC, construit la géométrie
    et fixe la fréquence.

    Paramètres
    ----------
    freq_hz : float
        Fréquence en Hz.
    antennes_coords : list of dict
        Liste d'antennes.

    Retour
    ------
    context : nec_context
        Contexte PyNEC configuré.
    """
    context = nec_context()
    setup_geometry(context, antennes_coords)
    context.fr_card(0, 1, freq / 1e6, 0)  # Fréquence en MHz pour ex_card
    return context

def get_currents_per_segment(sc):
    currents_per_segment = []
    if hasattr(sc, 'get_n') and hasattr(sc, 'get_current'):
        n = sc.get_n()
        currents = sc.get_current()
        segment_numbers = sc.get_current_segment_number()
        tags = sc.get_current_segment_tag()
        for i in range(n):
            current = currents[i]
            amplitude = abs(current)
            phase = np.angle(current, deg=True)
            segment_number = segment_numbers[i]
            tag = tags[i]
            currents_per_segment.append({
                'segment_number': segment_number,
                'tag': tag,
                'amplitude': amplitude,
                'phase': phase
            })
    else:
        print("Les méthodes get_n() ou get_current() ne sont pas disponibles.")
    return currents_per_segment

def calculate_self_impedances(num_antennas, antennes_coords):
    """
    Calcule la matrice d'impédances propre (self )
    entre tous les dipôles, en utilisant PyNEC.
    Retourne une matrice (num_antennas x num_antennas) de complex dont la diagonale représente les impédances propres.  Les impédances mutuelles ne sont pas calculées dans cette fonction.

    Hypothèse :
    - On excite un dipôle à la fois
    - Les autres sont mis en charge "quasi-ouverture" (impédance ~ très grande)

    Remarque :
    - Nécessaire de calculer les impédances propre avant les impédances mutuelles 
    """
    # Step 1 : calcul des self-impedances (diagonale)
    self_impedances = np.zeros((num_antennas, num_antennas), dtype=complex)
    for i in range(num_antennas):
        context = create_context(freq, antennes_coords)
        context.ex_card(0, i + 1, half_segment, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for k in range(num_antennas):
            if k != i:
                context.ld_card(4, k + 1, 0, segment_count_impair - 1, 1e50, 1e50, 0.0)
        context.xq_card(0)
        sc = context.get_structure_currents(0)
        ipt = context.get_input_parameters(0)
        voltages = ipt.get_voltage()
        currents = sc.get_current()
        self_impedances[i, i] = voltages[0] / currents[i * segment_count_impair + position_half_segment]
    return self_impedances

def calculate_impedances(num_antennas, antennes_coords):
    """
    Calcule la matrice d'impédances (self + mutuelles)
    entre tous les dipôles, en utilisant PyNEC.
    Retourne une matrice (num_antennas x num_antennas) de complex.
    A chaque itération on ne considère que deux antennes. Les autres sont placés en circuit ouvert (impédances infinies) permettant de dire que 
    (1)    V1 = Z11 I1 + Z12 I2,
    (2)    V2 = Z21 I1 + Z22 I2, 
    où V2/I2= − ZL = 0 
    V1 est la tension aplliquée connue, Z11 calculé dans calculate_self_impedances, I1 et I2 peuvent être simulé
.   Z12= (V1- z11 I2 )/ I2
    Hypothèse :
    - On excite un dipôle à la fois
    - Les autres sont mis en charge "quasi-ouverture" (impédance ~ très grande)
    - pour avoir les (N^2 - N) impédances mutuelles il faut faire (N^2 - N) simulations simulations
    - Les courants et tensions utilisées sont ceux des segments centraux de chaques antennes

    
    """
    self_imped = calculate_self_impedances(num_antennas, antennes_coords)
    mutual_impedances = np.zeros((num_antennas, num_antennas), dtype=complex)
    for i in range(num_antennas):
        for j in range(num_antennas):
            if i != j:
                context = create_context(freq, antennes_coords)
                context.ex_card(0, i + 1, half_segment, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                for k in range(num_antennas):
                    if k != j and k != i:
                        context.ld_card(4, k + 1, 0, segment_count_impair - 1, 1e50, 1e50, 0.0)
                context.xq_card(0)
                sc = context.get_structure_currents(0)
                ipt = context.get_input_parameters(0)
                voltages = ipt.get_voltage()
                currents = sc.get_current()
                mutual_impedances[i, j] = (voltages[0] - currents[i * segment_count_impair + position_half_segment] * self_imped[i, i]) /  currents[(j) * segment_count_impair + position_half_segment]
                  
                
                
    return self_imped + mutual_impedances

def impedance_matrix(num_antennas, antennes_coords, reglage='auto', impedances_manuel=None):
    """
    Construit une matrice diagonale d'impédances (charges),
    pour chaque dipôle.

    Paramètres
    ----------
    num_antennas : int
    antennes_coords : list
        Non utilisé ici, juste par cohérence.
    reglage : str
        'auto' => tous identiques (ex. 73 + j42.5) - purement illustratif
        'manuel' => utilise impedances_manuel.
    impedances_manuel : list of complex
        Si 'manuel' => liste d'impédances complexes.

    Retour
    ------
    np.diag(...) : ndarray
    """
    if reglage == 'manuel':
        if impedances_manuel is None or len(impedances_manuel) != num_antennas:
            raise ValueError("Pour 'manuel', fournissez une liste 'impedances_manuel' de longueur égale à 'num_antennas'.")
        impedances = impedances_manuel
    elif reglage == 'auto':
        impedances = [complex(73, 42.5)] * num_antennas
    else:
        raise ValueError("Le paramètre 'reglage' doit être 'auto' ou 'manuel'.")
    return np.diag(impedances)

def tension_Oc_vector(n_antennes, antennes_coords, reglage='auto', tensions_manuel=None):
    """
    Génère le vecteur des tensions à vide (Voc) pour chaque dipôle.
    - reglage='auto' => On met 12.4 V sur l'antenne de type 'emitter', 0V sur les autres. 
    - si ZLoad_emitter= -Im(Zin_emitter) , la puissance diffusée dans l'espace libre sera de 1W
    - reglage='manuel' => On utilise tensions_manuel.

    Retour
    ------
    Voc : np.array (n_antennes, 1)
    """
    if reglage == 'manuel':
        if tensions_manuel is None or len(tensions_manuel) != n_antennes:
            raise ValueError("Pour 'manuel', fournissez une liste tensions_manuel de longueur égale à n_antennes.")
        tensions = tensions_manuel
    elif reglage == 'auto':
        tensions = []
        for antenna in antennes_coords:
            if antenna['type'] == 'emitter':
                tensions.append(complex(12.40, 0))
            else:
                tensions.append(complex(0, 0))
    else:
        raise ValueError("Le paramètre 'reglage' doit être 'auto' ou 'manuel'.")

    return np.array(tensions).reshape(-1, 1)

def calculate_Voc(impedance_matrix_data, num_antennas, excitation_voltages, impedance_loads, antennes_coords):
    """
    Calcule les tensions/courants en chaque dipôle,
    en tenant compte de la matrice d'impédances mutuelles,
    des tensions appliquées et des charges appliquées (impedance_loads).
    Voir le modèle équivalent dans le pdf pour comprendre les correspondances, les conventions utilisées.
    Retour
    ------
    (results_dict, contextNEC)

    results_dict : dict contenant :
       Voc, Vl, Vin, vin_Oc, V, vbis, currents_at_center, zin, ...
    """
    context = create_context(freq, antennes_coords)
    # Appliquer excitations
    for i in range(num_antennas):
        V = excitation_voltages[i][0]
        if V != 0:
            context.ex_card(0, i+1, half_segment, half_segment, V.real, V.imag, 0,0,0,0)

    # Appliquer loads
    diag_load = np.diag(impedance_loads)
    for i in range(num_antennas):
        ZL = diag_load[i]
        if ZL != 0:
            R = ZL.real
            X = ZL.imag
            L = X/(2*math.pi*freq)
            context.ld_card(0, i+1, half_segment, half_segment, R, L, 0.0)

    context.xq_card(0)
    sc = context.get_structure_currents(0)
    ipt = context.get_input_parameters(0)
    currents = sc.get_current()
    # voltages d'entrée:
    voltages_in = ipt.get_voltage()
    
    currents_at_center = np.zeros((num_antennas,1), dtype=complex)
    
    
    
    for i in range(num_antennas):
        idx_center = i*segment_count_impair+ position_half_segment
        currents_at_center[i,0] = currents[idx_center]
    
    
    
    zin = np.zeros(num_antennas, dtype=complex)

    for i in range(num_antennas):
        # Création de ZL_mod, une matrice diagonale avec ZL sauf à la position (i,i) où c'est 0
        ZL_mod = np.zeros((num_antennas, num_antennas), dtype=complex)
        for j in range(num_antennas):
            if j != i:
                ZL_mod[j, j] = impedance_loads[j, j]
            # Sinon, ZL_mod[i, i] reste à 0

        # Calcul de Zmod
        Zmod = impedance_matrix_data + ZL_mod

        # Calcul de l'inverse de Zmod
        try:
            Zmod_inv = np.linalg.inv(Zmod)
        except np.linalg.LinAlgError:
            
            Zmod_inv = np.zeros((num_antennas, num_antennas), dtype=complex)

        # Calcul de zin[i] en prenant l'inverse du scalaire
        zin_scalar = Zmod_inv[i, i]
        if zin_scalar != 0:
            zin[i] = 1 / zin_scalar
        else:
            zin[i] = np.inf  # Ou une valeur appropriée si le scalaire est nul

    # Calcul de Vin = zin * I
    Vin = zin.reshape(-1, 1) * currents_at_center
    Vl = diag_load[:,None]*currents_at_center
    vin_Oc = excitation_voltages - Vin - Vl
    V = excitation_voltages - Vl
    Vbis = vin_Oc + Vin

    results = {
        'Voc': excitation_voltages,
        'Vl': Vl,
        'Vin': Vin,
        'vin_Oc': vin_Oc,
        'V': V,
        'vbis': Vbis,
        'currents_at_center': currents_at_center,
        'zin': zin,
        'impedance_loads': diag_load,
        'impedance_matrix': impedance_matrix_data
    }
    return results,context

def calculate_powers_dbm(results):
    """
    Calcule différentes puissances à partir des tensions et courants, et les renvoie en dbm :
      - P_V_oc    : puissance liée à Voc  (V_oc * conj(I))
      - P_Vin_oc  : puissance liée à vin_Oc (vin_Oc * - conj(I))
      - P_L       : puissance liée à Vl   (V_l  * conj(I))
      - P_in      : puissance liée à Vin  (V_in * conj(I))

    Les formules utilisées sont du type :
        P(mW) = 1000 * [0.5 * Re(V * conj(I))]
    Puis conversion en uW :
        P(dbm) = 10 * log10(P(mW))

    ATTENTION :
    -----------
    - Dans certains scénarios, la partie réelle peut être négative ou très faible,
      conduisant à des puissances nulles ou négatives. La conversion en uW n'étant
      pas définie pour P ≤ 0, on remplace ces cas par np.nan et on enregistre leur indice.
    - Si vous souhaitez manipuler la puissance en échelle linéaire (uW) plutôt qu'en uW,
      vous pouvez récupérer directement les variables val_P_V_oc, val_P_Vin_oc, etc.
      avant la conversion. Et donc utiliser la fonction calculate_powers

    Paramètres
    ----------
    results : dict
        Doit contenir :
        - 'Voc'               : tension à vide (np.array)
        - 'Vl'                : tension aux bornes de la charge (np.array)
        - 'vin_Oc'            : tension d'entrée en circuit ouvert pour l'émetteur (np.array)
        - 'Vin'               : tension d'entrée en charge (np.array)
        - 'currents_at_center': courant au centre (np.array)

    Retour
    ------
    dict
        {
            'P_V_oc'   : np.array de puissances en dbm,
            'P_Vin_oc' : np.array de puissances en dbm,
            'P_L'      : np.array de puissances en dbm,
            'P_in'     : np.array de puissances en dbm
        }

    Notes
    -----
    - Les puissances retournées sont en uW, et valent NaN si la puissance linéaire
      était négative ou nulle.
    - Pour tracer ou inspecter les valeurs négatives / nulles, référez-vous
      au contenu de 'error_logs' (dans la fonction safe_log10).
    """

    Voc = results['Voc']
    Vl = results['Vl']
    vin_Oc = results['vin_Oc']
    Vin = results['Vin']
    currents = results['currents_at_center']

    
    val_P_V_oc   = 1000 * (0.5 * np.real(Voc   * np.conj(currents)))
    val_P_Vin_oc = 1000 * (0.5 * np.real(vin_Oc * -np.conj(currents)))
    val_P_L      = 1000 * (0.5 * np.real(Vl    * np.conj(currents)))
    val_P_in     = 1000 * (0.5 * np.real(Vin   * np.conj(currents)))

    error_logs = []

    def safe_log10(values, name):
        """
        Applique log10 à 'values' si celles-ci sont strictement positives.
        Remplace toute valeur <= 0 par NaN et enregistre un log d'erreur.
        """

        # Indices des puissances négatives ou nulles
        problem_indices = np.where(values <= 0)[0]
        if problem_indices.size > 0:
            error_logs.append({
                "name": name,
                "indices": problem_indices.tolist(),
                "values": values[problem_indices].tolist()
            })
        # Copie pour ne pas modifier l'original
        safe_values = values.copy()
        safe_values[safe_values <= 0] = np.nan  # Evite l'erreur log10
        return np.log10(safe_values)


    # Conversion en uW : 10 * log10( P[dbm] )
    P_V_oc   = 10 * safe_log10(val_P_V_oc,   "P_V_oc")
    P_Vin_oc = 10 * safe_log10(val_P_Vin_oc, "P_Vin_oc")
    P_L      = 10 * safe_log10(val_P_L,      "P_L")
    P_in     = 10 * safe_log10(val_P_in,     "P_in")
    
    powers = {
        'P_V_oc'  : P_V_oc,
        'P_Vin_oc': P_Vin_oc,
        'P_L'     : P_L,
        'P_in'    : P_in,
    }


    return powers
def calculate_powers(results):
    """
    Calcule différentes puissances à partir des tensions et courants, et les renvoie en uW. :
      - P_V_oc    : puissance liée à Voc  (V_oc * conj(I))
      - P_Vin_oc  : puissance liée à vin_Oc (vin_Oc * - conj(I))
      - P_L       : puissance liée à Vl   (V_l  * conj(I))
      - P_in      : puissance liée à Vin  (V_in * conj(I))

    Les formules utilisées sont du type :
        P(mW) = 1000000 * [0.5 * Re(V * conj(I))]
    
    ATTENTION :
    
    Paramètres
    ----------
    results : dict
        Doit contenir :
        - 'Voc'               : tension à vide (np.array)
        - 'Vl'                : tension aux bornes de la charge (np.array)
        - 'vin_Oc'            : tension d'entrée en circuit ouvert pour l'émetteur (np.array)
        - 'Vin'               : tension d'entrée en charge (np.array)
        - 'currents_at_center': courant au centre (np.array)

    Retour
    ------
    dict
        {
            'P_V_oc'   : np.array de puissances en uW.,
            'P_Vin_oc' : np.array de puissances en uW.,
            'P_L'      : np.array de puissances en uW.,
            'P_in'     : np.array de puissances en uW.
        }

    Notes
    -----
    - Les puissances retournées sont en uW, 
    """

    Voc = results['Voc']
    Vl = results['Vl']
    vin_Oc = results['vin_Oc']
    Vin = results['Vin']
    currents = results['currents_at_center']

    # Calcul en échelle linéaire (mW)
    # Calcul en échelle linéaire (mW)
    val_P_V_oc   = 1000000 * (0.5 * np.real(Voc   * np.conj(currents)))
    val_P_Vin_oc = 1000000 * (0.5 * np.real(vin_Oc * -np.conj(currents)))
    val_P_L      = 1000000 * (0.5 * np.real(Vl    * np.conj(currents)))
    val_P_in     = 1000000 * (0.5 * np.real(Vin   * np.conj(currents)))

    
    powers = {
        'P_V_oc'  : val_P_V_oc,
        'P_Vin_oc': val_P_Vin_oc,
        'P_L'     : val_P_L,
        'P_in'    : val_P_in,
    }


    return powers

def afficher_resultats(resultats, format_polaire=False):
    print("\n--- Résultats des calculs ---\n")
    def afficher_element(element, indent=4):
        espace=" "*indent
        if np.iscomplexobj(element):
            if format_polaire:
                module=np.abs(element)
                phase=np.angle(element,deg=True)
                print(f"{espace}Module = {module:.6f}, Phase = {phase:.2f}°")
            else:
                print(f"{espace}{element.real:.6f} + {element.imag:.6f}j")
        elif isinstance(element,(int,float)):
            print(f"{espace}{element:.6f}")
        else:
            print(f"{espace}{element}")

    def afficher_structure(valeur,indent=4):
        espace=" "*indent
        if isinstance(valeur,np.ndarray):
            if valeur.ndim==1:
                for i,element in enumerate(valeur):
                    print(f"{espace}[{i}] :",end="")
                    afficher_element(element,indent=0)
            elif valeur.ndim==2:
                for i,ligne in enumerate(valeur):
                    print(f"{espace}Ligne {i}:")
                    for j,element in enumerate(ligne):
                        print(f"{espace}  [{i},{j}] :",end="")
                        afficher_element(element,indent=0)
            else:
                for i,sous_tableau in enumerate(valeur):
                    print(f"{espace}Sous-tableau {i}:")
                    afficher_structure(sous_tableau,indent+4)
        elif isinstance(valeur,(list,tuple)):
            for i,element in enumerate(valeur):
                print(f"{espace}[{i}] :")
                afficher_structure(element,indent+4)
        elif isinstance(valeur,dict):
            for sous_cle,sous_valeur in valeur.items():
                print(f"{espace}{sous_cle} :")
                afficher_structure(sous_valeur,indent+4)
        else:
            afficher_element(valeur,indent=indent)

    for cle,valeur in resultats.items():
        print(f"{cle} :")
        afficher_structure(valeur,indent=4)
        print()

