# Projet de PFE

## Apprentissage Fédéré pour la Gestion Intelligente de la Recharge des Véhicules Électriques

---

# 1. Contexte et Objectif du Projet

Ce projet propose un prototype de système intelligent d’aide à la décision pour la recharge de véhicules électriques (VE), fondé sur :

* des données réelles de mobilité (dataset GeoLife),
* un modèle énergétique simulé,
* un environnement multi-véhicules avec congestion réelle,
* un apprentissage par renforcement,
* un apprentissage fédéré pour traiter des données naturellement hétérogènes.

L’objectif principal est de comparer différentes stratégies d’apprentissage dans un environnement réaliste où plusieurs véhicules partagent un nombre limité de bornes :

* Apprentissage centralisé (baseline théorique).
* Apprentissage fédéré classique (FedAvg).
* Apprentissage fédéré dynamique (FedDyn).
* Heuristiques simples (règles déterministes).

Le projet vise à analyser les compromis entre :

* coût de recharge,
* temps d’attente,
* détours,
* risque de batterie critique,
* fréquence des demandes de recharge.

---

# 2. Installation et Lancement

## 2.1 Prérequis

* Python ≥ 3.10 recommandé
* Installation des dépendances :

```bash
pip install -r requirements.txt
```

---

## 2.2 Organisation du projet

Structure générale :

```text
PFE/
├── data/
│   ├── geolife_raw/        # Dataset GeoLife (non versionné)
│   ├── processed/          # Fichiers intermédiaires (parquet)
├── src/                    # Code source
├── outputs/                # Résultats des expériences
├── README.md
└── requirements.txt
```

---

## 2.3 Lancer l’expérience complète

Depuis la racine du projet :

```bash
python -m src.run_experiment
```

Ce script exécute automatiquement :

1. Le parsing et le nettoyage des données GeoLife
2. La construction des épisodes de mobilité
3. La génération des stations de recharge
4. L’entraînement des modèles :

   * Centralisé
   * FedAvg
   * FedDyn
5. L’évaluation multi-véhicules
6. Trois exécutions indépendantes (seeds différentes)
7. L’export des résultats (moyenne et écart-type)

---

# 3. Données de Mobilité : GeoLife

## 3.1 Présentation du Dataset

Le projet repose sur le dataset :

**Microsoft Research GeoLife GPS Trajectory Dataset**

Il contient :

* 182 utilisateurs
* Plus de 17 000 trajectoires
* Données GPS collectées sur plusieurs années
* Informations : latitude, longitude, altitude, date et heure

Important :

* Il ne contient aucune information énergétique.
* Il ne contient aucune information sur des bornes de recharge.
* Il ne contient aucune information sur des véhicules électriques.

Les aspects énergétiques et d’infrastructure sont simulés par-dessus les trajectoires de mobilité.

---

## 3.2 Structure interne du dataset

Organisation des fichiers :

```text
Geolife Trajectories 1.3/
└── Data/
    ├── 000/
    │   └── Trajectory/
    │       ├── 20081023025304.plt
    │       ├── ...
    ├── 001/
    ├── ...
```

Chaque dossier correspond à un utilisateur.

Chaque fichier `.plt` correspond à une trajectoire continue.

---

## 3.3 Pipeline de transformation des données

Le pipeline comporte deux grandes étapes :

### 1. Nettoyage et préparation (geolife_prepare.py)

* Lecture des fichiers `.plt`
* Nettoyage des points aberrants
* Conversion des timestamps
* Agrégation dans un fichier parquet

Objectif : obtenir une base exploitable et structurée.

---

### 2. Construction des épisodes (geolife_load.py)

Transformation des points GPS en épisodes discrets :

* Discrétisation spatiale en grille (zi, zj)
* Ré-échantillonnage temporel (ex. : pas de 10 minutes)
* Calcul des distances entre pas successifs
* Construction d’épisodes par utilisateur

Chaque épisode contient :

| Colonne | Description                  |
| ------- | ---------------------------- |
| zi, zj  | Position discrète sur grille |
| dist_km | Distance parcourue           |
| dt      | Timestamp                    |

Chaque utilisateur possède plusieurs épisodes indépendants.

---

# 4. Architecture détaillée du code

Le dossier `src/` est structuré par modules fonctionnels.

---

## 4.1 Orchestration principale

### run_experiment.py

Rôle :

* Script principal du projet.
* Lance le pipeline complet.
* Gère les seeds.
* Entraîne les modèles.
* Lance les évaluations.
* Exporte les résultats.

C’est le point d’entrée unique.

---

## 4.2 Gestion des données

### geolife_prepare.py

* Parsing des fichiers `.plt`
* Nettoyage et conversion
* Sauvegarde au format parquet

### geolife_load.py

* Construction des épisodes
* Discrétisation spatiale
* Resampling temporel

---

## 4.3 Modélisation environnementale

### env_ev.py

Environnement véhicule-centré.

Responsabilités :

* Gestion d’un épisode
* Mise à jour du SoC
* Calcul des récompenses
* Interaction avec les stations
* Construction du vecteur d’observation

C’est le cœur de la simulation.

---

### vehicles.py

Modèle énergétique simplifié :

* Capacité batterie
* Consommation par km
* Mise à jour du SoC
* Recharge selon puissance station

---

### stations.py

Modélisation des bornes :

* Nombre de ports
* File d’attente FIFO
* Sessions de recharge
* Estimation réaliste du temps d’attente

La congestion est réellement simulée (partage entre véhicules).

---

## 4.4 Apprentissage par renforcement

### models.py

Définit le réseau de neurones (Policy Network) :

* MLP simple
* Entrée : observation complète
* Sortie : probabilité d’action (station ou attendre)

---

### rl_train.py

Implémentation de l’algorithme REINFORCE :

* Policy gradient
* Mise à jour des poids
* Entraînement local

Choix volontairement simple pour la lisibilité et la stabilité.

---

## 4.5 Apprentissage fédéré

### federated.py

Implémente :

* FedAvg : moyenne simple des paramètres locaux
* FedDyn : ajout d’un terme dynamique pour limiter la dérive locale

Motivation :
Les trajectoires sont naturellement non-IID.
Chaque utilisateur a un comportement différent.

---

## 4.6 Évaluation

### evaluate_multi.py

Évaluation des modèles RL :

* Simulation multi-véhicules
* Stations partagées
* Congestion réelle

---

### evaluate_heuristics.py

Évaluation des politiques heuristiques dans le même cadre.

---

### baselines.py

Contient les règles heuristiques :

* Recharge à la station la plus proche
* Recharge selon attente + prix
* Règle seuil SoC

---

### metrics.py

Agrégation des métriques :

* Coût total
* Attente totale
* Détour total
* Temps sous seuil critique
* Nombre de demandes de recharge

---

### utils_geo.py

* Gestion de la grille spatiale
* Calcul de distance Manhattan

---

# 5. Modélisation du système

## 5.1 Observation

Chaque agent observe :

* SoC
* Encodage horaire sin/cos
* Distance aux stations
* Temps d’attente estimé
* Prix
* Occupation actuelle

---

## 5.2 Action

L’agent choisit :

* Une station
* Ou attendre (ne pas recharger)

---

## 5.3 Récompense

Composée de :

* Pénalité de détour
* Pénalité d’attente
* Coût énergétique
* Pénalité si SoC critique
* Bonus d’efficacité (faible congestion)

---

# 6. Évaluation et Résultats

Simulation :

* 20 véhicules simultanés
* Stations partagées
* 3 runs indépendants

Métriques :

| Métrique          | Description               |
| ----------------- | ------------------------- |
| cost_mean         | Coût moyen                |
| wait_mean         | Temps d’attente moyen     |
| detour_mean       | Détour moyen              |
| soc_critical_rate | Ratio sous seuil critique |
| charge_requests   | Nombre de demandes        |

Export :

```
outputs/
├── results_runs.csv
├── results_mean.csv
├── results_std.csv
```

---

# 7. Choix de conception

1. Stations générées selon densité de mobilité
2. Batterie simulée (pas de données EV réelles)
3. REINFORCE pour simplicité
4. 3 seeds pour robustesse statistique
5. Évaluation multi-véhicules pour reproduire congestion réelle

---

# 8. Limites

* Modèle énergétique simplifié
* Pas de données stations réelles
* Pas d’intégration réseau électrique
* RL volontairement simple

---

# 9. Perspectives

* Intégration données EV réelles
* Stations géographiques réelles
* Smart Grid
* Multi-agent RL
* Personnalisation fédérée
* Algorithmes plus avancés (PPO, SAC)

