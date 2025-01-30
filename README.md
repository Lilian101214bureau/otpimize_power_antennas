# README - Projet de Modélisation d’Antennes avec PyNEC

Ce dépôt propose un ensemble de scripts Python permettant de :
- **Modéliser** différentes antennes (dipôles, Yagi, réseaux circulaires, sphériques, etc.).
- **Calculer** la matrice d’impédances (self et mutuelle) via la bibliothèque `PyNEC`.
- **Optimiser** l’adaptation d’impédances (approches GA global, ou encore « reflectors only » en continu ou discret).
- **Tracer** des graphiques (puissances, géométries 3D) et sauvegarder des données (CSV, PNG).

Tout le dépot se base sur la structure interne de l'outil eznec, on retrouvera donc les logique de l'approche "ful wave" discrétisant les antennes en en nombre fini de segments. Ceci permettant de récupérer les courants, tensions, impédances ,...., le long de chaque antenne et incluant les interactions EM entre les antennes de notre réseau.
---

## 1. Structure principale du projet


- **constants.py**  
  Définit les constantes globales (fréquence `freq`, longueur d’onde `lam`, nombre de segments `segment_count_impair`, etc.).

- **geometry.py**  
  Fonctions pour générer diverses **géométries** d’antennes (aléatoire 3D, linéaire Yagi, circulaire, carré, etc.).

- **simulation.py**  
  Gère l’**interface** avec PyNEC (création d’un contexte `nec_context`, construction de la géométrie filaire, calcul de matrices d’impédance, etc.).

- **optimization.py**  
  Contient des méthodes d’**optimisation** (ex. GA global : `optimize_impedances`; optim. partielle des réflecteurs en discret/continu).

- **plotting.py**  
  Fonctions de **visualisation** (3D pour la position des dipôles, bar plots pour la puissance, etc.).

- **main.py**  
  Script principal regroupant plusieurs **scénarios de test** (réseaux carrés, sphère, Yagi…) avec génération de résultats (PNG, CSV).

- **test_3_antennas_varying_distance.py** (exemple)  
  Illustrations ou tests plus spécifiques (ex. variation de distance entre 2–3 antennes).

- **test_get_matrix.py**  
  Extrait un exemple de calcul de **matrice d’impédance** pour un certain nombre d’antennes, écrit les résultats en CSV, etc.

---

## 2. Prérequis

- **Python 3.8** ou version plus récente
- Bibliothèques Python :
  - [PyNEC==1.7.3.4](https://pypi.org/project/PyNEC/1.7.3.4/)  
  - [numpy](https://pypi.org/project/numpy/)  
  - [matplotlib](https://pypi.org/project/matplotlib/)  
  - [scipy](https://pypi.org/project/scipy/) (pour `differential_evolution`)

### Installation rapide (via pip)

pip install numpy
pip install matplotlib
pip install scipy

### Se déplaces au dossier contenant le code full_wave_approach EXEMPLE A ADAPTER EN FONCTION DE L'EMPLACEMENT DU GIT CLONE 
 cd C:\Documents\python-necpp\PyNEC\example\project\structurationned_code\full_wave_approach

### Mettre à jour pip, setuptools et wheel 
pip install --upgrade pip setuptools wheel

### Dans un terminal  
pip install PyNEC==1.7.3.4

