
# README ENGLISH

# README FRENCH PLUS BAS 




# README - Antenna Modeling Project with PyNEC

This repository provides a set of Python scripts to:

- **Model** different antennas (dipole, Yagi, circular arrays, spherical, etc.).
- **Compute** the impedance matrix (self and mutual) using the [PyNEC](https://pypi.org/project/PyNEC/) library.
- **Optimize** impedance matching (either a **global GA approach**—which finds the best \(R + jX\) load for *every* antenna simultaneously—or a **“reflectors only”** strategy, in **continuous** or **discrete** mode, to fine-tune only reflector antennas). All these approaches aim to maximize the power received by a target antenna (e.g., the “receiver” in an RFID application).
- **Plot** various graphs (powers, 3D geometries) and save data (CSV, PNG).

This work is based on a **full wave** approach (segmenting the antennas) to access currents, voltages, and impedances, including mutual electromagnetic interactions within the antenna network.  
**PyNEC** is the Python wrapper library used by the scripts `main.py`, `test_3_antennas_varying_distance.py`, and `test_get_matrix.py`. It exploits EZNEC source code (full wave methods) translated into simple Python functions.

---

## 1. Context and Disclaimer

This project was carried out as part of an **internship** to address issues of **improving the received power** by a targeted RFID tag through impedance optimization in a network of reconfigurable tags.  
For more technical/theoretical details (conventions, definitions of parameters, etc.), please refer to the PDF:  
**_“Amélioration de la puissance reçue par un tag RFID cible via l’optimisation d’impédances dans un réseau de tags reconfigurables.pdf”_** (in French).

**Note**: This repository **is not a final version**. It may evolve if necessary to incorporate improvements or corrections.

---

## 2. Main Structure of the Project

- **`constants.py`**  
  Defines global constants (frequency `freq`, wavelength `lam`, number of segments, etc.).

- **`geometry.py`**  
  Functions to generate various **antenna geometries** (3D random, linear Yagi, circular, square, etc.).

- **`simulation.py`**  
  **PyNEC** interface (creating a `nec_context`, building wire geometry, computing impedance matrices, etc.).

- **`optimization.py`**  
  Implements **several optimization strategies**:
  1. **Global GA**: finds the best \((R + jX)\) load for *every* antenna (including emitter, receiver, reflectors).
  2. **Reflectors-only (continuous)**: emitter and receiver impedances are fixed (e.g., \(-j\Im(Z_{\text{em}})\), \(\overline{Z_{\text{rx}}}\)), while reflectors are optimized continuously for \((R + jX)\).
  3. **Reflectors-only (discrete)**: similar to above, but reflectors must be chosen from a **discrete** set of reactances.
  4. **Generation of discrete reactances** (`generate_discrete_reactances`).

- **`plotting.py`**  
  **Visualization** functions (3D for dipole positions, bar plots for power, etc.).

- **`main.py`**  
  Main entry point: combines several **test scenarios** (square arrays, spheres, Yagi…) and produces results (PNG, CSV).

- **`test_3_antennas_varying_distance.py`** (advanced example)  
  Illustrates variation of distance between 2–3 antennas, testing different impedances.

- **`test_get_matrix.py`**  
  Shows how to compute an **impedance matrix** for a certain number of antennas and export the results to CSV.

---

## 3. About PyNEC and Copyright

- The **PyNEC** package is distributed via PyPI ([PyNEC on PyPI](https://pypi.org/project/PyNEC/)) and is **not** developed in this repository.  
- The copyright and licensing of PyNEC belong to its original author/publisher.  
- The use of PyNEC in this project is **as a dependency**. Please make sure to respect PyNEC’s license and usage terms.

**Author of the scripts in this repository**: Michalak Lilian (<lilianmichalak2002@gmail.com>).

---

## 4. Prerequisites and Installation

### 4.0 Getting the Source Code

Two main methods:

1. **Clone via Git:**
   
   cd <Your_desired_location>
   git clone https://github.com/Lilian101214bureau/otpimize_power_antennas.git
   cd otpimize_power_antennas

2. **Download the .zip archive from GitHub:**
  Go to https://github.com/Lilian101214bureau/otpimize_power_antennas
  Click Code > Download ZIP.
  Unzip to the desired location.
  In a terminal, navigate to the extracted folder
  cd <Your_desired_location>/otpimize_power_antennas

## 4.1 Python Version

The project was primarily tested with Python 3.10.11 (stable) along with a compatible wheel for PyNEC.

Other versions (>=3.8) may work, but Python 3.10.11 is recommended to ensure compatibility with PyNEC==1.7.3.4 and avoid potential wheel conflicts.

Beforehand, check your Python version and update your tools if needed:

python --version

# Make sure you are in a location/terminal where Python/PIP is accessible.

cd <Your_code_extraction_directory>
pip install --upgrade pip setuptools wheel

## 4.2 Required Python Libraries

- [PyNEC==1.7.3.4](https://pypi.org/project/PyNEC/1.7.3.4/)
- [NumPy](https://pypi.org/project/numpy/)
- [Matplotlib](https://pypi.org/project/matplotlib/)
- [SciPy](https://pypi.org/project/scipy/) (for `differential_evolution`)

## 4.3 Quick Installation (via pip)

Make sure you have Python >= 3.8 (preferably 3.10.11).
cd <Your_code_extraction_directory>
# Update pip, setuptools, and wheel (if not already done)
pip install --upgrade pip setuptools wheel

# Install the main dependencies
pip install numpy
pip install matplotlib
pip install scipy

# Install PyNEC (version 1.7.3.4)
pip install PyNEC==1.7.3.4

## 4.4 Repository Location and Execution
Ensure you are in the folder containing the code, for example:

cd C:\Documents\python-necpp\PyNEC\example\project\structurationned_code\full_wave_approach

Then you can run one of the scripts, for instance:

python main.py
python test_3_antennas_varying_distance.py
python test_get_matrix.py

The results (CSV, PNG, etc.) will be generated in the directory specified in the code.

### 5. Final Remarks

As mentioned, this project is not final and may still evolve.
Output paths, activated scenarios, or simulation parameters can be configured directly in the scripts.
For more details on the theory and conventions used, please refer to the attached PDF:
“Amélioration de la puissance reçue par un tag RFID cible via l’optimisation d’impédances dans un réseau de tags reconfigurables.pdf” (in French).

Carried out as part of an internship at CITI LAB in the RHODES team (guillaume.villemaud@insa-lyon.fr).

For any inquiry, please contact the author: Michalak Lilian (lilianmichalak2002@gmail.com).



# README - Projet de Modélisation d’Antennes avec PyNEC

Ce dépôt propose un ensemble de scripts Python permettant de :

- **Modéliser** différentes antennes (dipôles, Yagi, réseaux circulaires, sphériques, etc.).
- **Calculer** la matrice d’impédances (self et mutuelle) via la bibliothèque [PyNEC](https://pypi.org/project/PyNEC/).
- **Optimiser** l’adaptation d’impédances (soit par une **approche GA globale**—qui détermine les meilleures charges \((R + jX)\) pour *toutes* les antennes simultanément—ou par une stratégie **« reflectors only »** en **continu** ou **discret**, se concentrant uniquement sur les réflecteurs). L’objectif est de trouver les couples \(\mathrm{R}_i + j\,\mathrm{X}_i\) maximisant la puissance reçue par une antenne cible (par ex. un tag RFID).
- **Tracer** des graphiques (puissances, géométries 3D) et sauvegarder des données (CSV, PNG).

L’ensemble repose sur l’approche **full wave** (discrétisation en segments des antennes) afin d’accéder aux courants, tensions et impédances, y compris les interactions électromagnétiques mutuelles entre les antennes du réseau. 
**PyNEC** est la librairie wrap utilisée par les scrip main.py, test_3_antennas_varying_distance.py et test_get_matrix.py de ce code. PyNEC permet d'utiliser le code source de EZNEC, utilisant des méthodes **full wave** pour le traduire en des fonctions simples python;

---

## 1. Contexte et Avertissements

Ce projet a été réalisé dans le cadre d’un **stage** afin de répondre à des problématiques d’**amélioration de la puissance reçue** par un tag RFID cible via l’optimisation d’impédances dans un réseau de tags reconfigurables.  
Pour plus d’informations techniques et théoriques (conventions, définitions des grandeurs, etc.), se référer au document PDF :  
**_« Amélioration de la puissance reçue par un tag RFID cible via l’optimisation d’impédances dans un réseau de tags reconfigurables.pdf »_**.

**Note** : Ce dépôt **n’est pas une version finale**. Il est susceptible d’évoluer si nécessaire, afin d’y apporter des corrections et améliorations ultérieures.

---

## 2. Structure principale du projet

- **`constants.py`**  
  Définit les constantes globales (fréquence `freq`, longueur d’onde `lam`, nombre de segments, etc.).

- **`geometry.py`**  
  Fonctions pour générer diverses **géométries** d’antennes (aléatoire 3D, linéaire Yagi, circulaire, carré, etc.).

- **`simulation.py`**  
  Interface avec **PyNEC** (création d’un contexte `nec_context`, construction de la géométrie filaire, calcul de matrices d’impédance…).

- **`optimization.py`**  
  Implémente **plusieurs stratégies d’optimisation** :
  1. **GA global** : détermine \((R + jX)\) optimal pour *toutes* les antennes (émetteur, récepteur, réflecteurs).
  2. **Reflectors only (continu)** : impédances de l’émetteur et du récepteur fixées (ex. \(-j\,\Im(Z_{\text{em}})\) et \(\overline{Z_{\text{rx}}}\)), tandis que les réflecteurs sont optimisés continûment sur \((R + jX)\).
  3. **Reflectors only (discret)** : similaire à la version « continu », mais la réactance des réflecteurs doit être choisie parmi une **liste discrète** de valeurs.
  4. **Génération de réactances discrètes** (`generate_discrete_reactances`).

- **`plotting.py`**  
  Fonctions de **visualisation** (3D pour la position des dipôles, bar plots pour la puissance, etc.).

- **`main.py`**  
  Point d’entrée principal : rassemble plusieurs **scénarios de test** (réseaux carrés, sphère, Yagi…), génère les résultats (PNG, CSV).

- **`test_3_antennas_varying_distance.py`** (exemple avancé)  
  Simule la variation de distance entre 2–3 antennes, en testant différentes impédances.

- **`test_get_matrix.py`**  
  Extrait un exemple de calcul de **matrice d’impédance** pour un certain nombre d’antennes, écrit les résultats en CSV, etc.

---

## 3. À propos de PyNEC et droits d’auteur

- Le package **PyNEC** est distribué via PyPI (voir [PyNEC sur PyPI](https://pypi.org/project/PyNEC/)) et **n’est pas** développé dans ce dépôt.  
- Les droits d’auteur ou de licence relatifs à PyNEC appartiennent à son auteur/responsable de publication.  
- L’utilisation de PyNEC dans ce projet se fait **sous forme de dépendance** ; merci de respecter la licence de PyNEC et les conditions d’utilisation éventuelles.

**Auteur des scripts du présent dépôt** : Michalak Lilian, lilianmichalak2002@gmail.com

---

## 4. Prérequis et Installation

### 4.0 Récupération du code source

Plusieurs méthodes sont possibles :

1. **Cloner via Git :**
   
   cd <Votre_emplacement_souhaité>
   git clone https://github.com/Lilian101214bureau/otpimize_power_antennas.git
   cd otpimize_power_antennas
2. **élécharger l’archive .zip depuis GitHub**

Rendez-vous sur : https://github.com/Lilian101214bureau/otpimize_power_antennas
Cliquez sur Code > Download ZIP.
Décompressez le fichier à l’emplacement souhaité.
Dans un terminal, placez-vous dans le dossier extrait :

cd <Votre_emplacement_souhaité>/otpimize_power_antennas

### 4.1 Version de Python

Le projet a été principalement testé avec une version **Python 3.10.11** (stable) accompagnée d’une **wheel** compatible pour PyNEC.  
- D’autres versions (>=3.8) peuvent fonctionner, mais **Python 3.10.11** est recommandée pour assurer la compatibilité avec `PyNEC==1.7.3.4` et éviter d’éventuels problèmes liés aux wheels.


# Avant tout, vérifiez votre version de Python et mettez à jour vos outils :

python --version

# Assurez-vous de vous trouver dans l'emplacement où vous souhaitez travailler ou dans un terminal où Python/PIP est accessible.
cd <Votre_emplacement_de_décompréssion_du_code >
pip install --upgrade pip setuptools wheel

### 4.2 Bibliothèques Python nécessaires

- [PyNEC==1.7.3.4](https://pypi.org/project/PyNEC/1.7.3.4/)
- [NumPy](https://pypi.org/project/numpy/)
- [Matplotlib](https://pypi.org/project/matplotlib/)
- [SciPy](https://pypi.org/project/scipy/) (pour `differential_evolution`)

### 4.3 Installation rapide (via pip)


# Assurez-vous d’abord de posséder une version Python >= 3.8 (idéalement 3.10.11).
python --version

# Assurez-vous de vous trouver dans l'emplacement où vous souhaitez travailler ou dans un terminal où Python/PIP est accessible.
cd <Votre_emplacement_de_décompression_du_code>

# Mettre à jour pip, setuptools et wheel
pip install --upgrade pip setuptools wheel

# Installation des principales dépendances
pip install numpy
pip install matplotlib
pip install scipy

# Installation de PyNEC (version 1.7.3.4)
pip install PyNEC==1.7.3.4

### 4.4 Emplacement du dépôt et exécution

Assurez-vous d’être dans le dossier contenant le code, par exemple :

cd C:\Documents\python-necpp\PyNEC\example\project\structurationned_code\full_wave_approach

Ensuite, vous pouvez lancer un des scripts, par exemple :

python main.py

python test_3_antennas_varying_distance.py

python test_get_matrix;PY

Les résultats (CSV, PNG, etc.) se créeront alors dans le répertoire spécifié dans le code.

### 5. Remarques finales
Comme indiqué plus haut, ce projet n’est pas final : il peut encore être amené à évoluer.
Les paramètres (localisation du répertoire de sortie, scénarios à activer, etc.) sont configurables dans les fichiers de script.
Pour tout détail complémentaire sur la théorie et les conventions utilisées, veuillez consulter le PDF associé :
« Amélioration de la puissance reçue par un tag RFID cible via l’optimisation d’impédances dans un réseau de tags reconfigurables.pdf ».


Réalisé dans le cadre d’un stage au [CITI LAB](https://www.citi-lab.fr/) dans l'équipe RHODES (guillaume.villemaud@insa-lyon.fr) . 

Pour toute question, merci de contacter l’auteur : Michalak Lilian, lilianmichalak2002@gmail.com