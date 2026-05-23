# Projet de PFE : Apprentissage Fédéré pour la Gestion Intelligente de la Recharge des Véhicules Électriques

# 1. Présentation du projet

Ce projet de fin d’études a pour objectif de développer un prototype permettant d’optimiser les décisions de recharge de véhicules électriques (VE) dans un environnement multi-véhicules et congestionné.

Le système combine :

- des données de mobilité réelles ;
- un environnement de simulation de recharge ;
- de l’apprentissage par renforcement (Reinforcement Learning) ;
- de l’apprentissage fédéré (Federated Learning).

L’objectif principal est de comparer différentes stratégies d’apprentissage afin de déterminer comment les véhicules peuvent choisir intelligemment :

- quand se recharger ;
- où se recharger ;
- tout en minimisant :
  - le temps d’attente ;
  - le coût ;
  - les détours ;
  - les situations critiques de batterie.

---

# 2. Installation

## 2.1 Cloner le projet

```bash
git clone https://github.com/Iyad-78/PFE_federated_EV_charging.git
```

---

## 2.2 Installer les dépendances

```bash
pip install -r requirements.txt
```

---

# 3. Lancement du projet

Depuis la racine du projet :

```bash
python -m src.run_experiment
```

Le script lance automatiquement :

- le parsing GeoLife ;
- le prétraitement ;
- la génération des épisodes ;
- l’entraînement RL ;
- l’apprentissage fédéré ;
- l’évaluation multi-véhicules ;
- l’export des résultats.

Les résultats sont ensuite sauvegardés dans le dossier :

```text
outputs/
```

---

# 4. Objectifs du projet

Le projet compare plusieurs approches :

- apprentissage centralisé ;
- Federated Learning avec FedAvg ;
- Federated Learning avec FedDyn ;
- heuristiques simples de décision.

Le système prend en compte :

- la congestion des stations ;
- les files d’attente ;
- les contraintes énergétiques ;
- les habitudes de mobilité des utilisateurs.

---

# 5. Dataset utilisé : GeoLife

## 5.1 Présentation du dataset

Le projet utilise le dataset :

**Microsoft Research GeoLife GPS Trajectory Dataset**

Lien Kaggle :

```text
https://www.kaggle.com/datasets/arashnic/microsoft-geolife-gps-trajectory-dataset
```

Le dataset contient :

- 182 utilisateurs ;
- plus de 17 000 trajectoires GPS ;
- plusieurs millions de points GPS.

Chaque point contient notamment :

- latitude ;
- longitude ;
- timestamp ;
- altitude.

Le dataset ne contient pas directement :

- d’informations sur les véhicules électriques ;
- de données énergétiques ;
- de stations de recharge.

Ces éléments ont donc été simulés dans le cadre du projet.

---

## 5.2 Prétraitement des données

Les données GeoLife sont transformées en plusieurs étapes.

### Étape 1 : Parsing des fichiers `.plt`

Les fichiers GPS bruts sont convertis en DataFrame pandas puis sauvegardés au format Parquet.

Fichier concerné :

```text
src/geolife_prepare.py
```

---

### Étape 2 : Génération des épisodes

Les trajectoires sont ensuite :

- discrétisées spatialement sur une grille ;
- ré-échantillonnées temporellement ;
- converties en épisodes exploitables par l’environnement RL.

Fichier concerné :

```text
src/geolife_load.py
```

---

## 5.3 Colonnes principales utilisées

| Colonne | Description |
|---|---|
| `zi` | Position latitude discrétisée |
| `zj` | Position longitude discrétisée |
| `dist_km` | Distance parcourue |
| `dt` | Timestamp |
| `traj_id` | Identifiant de trajectoire |

---

# 6. Architecture générale du projet

Le pipeline complet du projet est le suivant :

```text
GeoLife brut
→ Prétraitement
→ Génération des épisodes
→ Simulation VE
→ Reinforcement Learning
→ Federated Learning
→ Évaluation multi-véhicules
→ Analyse des résultats
```

---

# 7. Structure du projet

```text
PFE/
├── data/
│   ├── geolife_raw/
│   └── processed/
│
├── outputs/
│   ├── results_runs.csv
│   ├── results_mean.csv
│   └── results_std.csv
│
├── src/
│   ├── baselines.py
│   ├── config.py
│   ├── env_ev.py
│   ├── evaluate_heuristics.py
│   ├── evaluate_multi.py
│   ├── federated.py
│   ├── geolife_load.py
│   ├── geolife_prepare.py
│   ├── metrics.py
│   ├── models.py
│   ├── rl_train.py
│   ├── run_experiment.py
│   ├── stations.py
│   ├── utils_geo.py
│   └── vehicles.py
│
├── requirements.txt
└── README.md
```

---

# 8. Description des principaux fichiers

## `run_experiment.py`

Fichier principal du projet.

Il orchestre :

- le chargement des données ;
- l’entraînement ;
- le Federated Learning ;
- l’évaluation ;
- l’export des résultats.

---

## `env_ev.py`

Implémente l’environnement de simulation.

Il gère :

- les déplacements ;
- le SoC (State of Charge) ;
- les récompenses RL ;
- les interactions avec les stations.

---

## `vehicles.py`

Définit le comportement des véhicules :

- batterie ;
- consommation ;
- recharge ;
- déplacement.

---

## `stations.py`

Gère les stations de recharge :

- occupation ;
- files d’attente ;
- sessions de recharge ;
- logique FIFO.

---

## `rl_train.py`

Implémente l’apprentissage par renforcement.

Le projet utilise une version simplifiée de :

- REINFORCE (Policy Gradient).

---

## `models.py`

Définit le réseau de neurones utilisé comme policy.

---

## `federated.py`

Implémente :

- FedAvg ;
- FedDyn ;
- l’agrégation des modèles locaux.

---

## `evaluate_multi.py`

Évaluation multi-véhicules avec congestion réelle.

Les véhicules partagent les mêmes stations.

---

## `evaluate_heuristics.py`

Évaluation des heuristiques classiques.

---

## `baselines.py`

Contient plusieurs stratégies simples :

- station la plus proche ;
- coût minimal ;
- règles fixes.

---

## `metrics.py`

Calcule les métriques finales :

- coût ;
- attente ;
- détour ;
- SoC critique ;
- score d’utilité.

---

## `config.py`

Contient tous les hyperparamètres :

- nombre de clients ;
- nombre de stations ;
- learning rate ;
- paramètres RL ;
- paramètres FL.

---

# 9. Modélisation du problème

## 9.1 Observation de l’agent

Chaque véhicule observe :

- son niveau de batterie ;
- l’heure ;
- la distance aux stations ;
- le temps d’attente estimé ;
- le prix de l’électricité ;
- l’occupation des stations.

---

## 9.2 Actions possibles

Le véhicule peut :

- choisir une station ;
- attendre sans recharger.

---

## 9.3 Fonction de récompense

La récompense prend en compte :

- le coût ;
- l’attente ;
- le détour ;
- le risque de batterie critique.

L’objectif est de maximiser l’utilité globale.

---

# 10. Federated Learning

## 10.1 Pourquoi utiliser le FL ?

Les données sont naturellement non-IID :

- chaque utilisateur possède ses propres habitudes ;
- certains roulent davantage ;
- certains utilisent des zones différentes.

Le FL permet :

- d’éviter de centraliser les données ;
- d’améliorer la confidentialité ;
- d’exploiter l’hétérogénéité des comportements.

---

## 10.2 Méthodes implémentées

### FedAvg

Moyenne simple des poids locaux.

### FedDyn

Ajout d’une régularisation dynamique afin de limiter la dérive des modèles locaux.

---

# 11. Évaluation

L’évaluation finale simule :

- plusieurs véhicules simultanés ;
- des stations partagées ;
- des situations de congestion.

---

## 11.1 Métriques utilisées

| Métrique | Description |
|---|---|
| `cost_mean` | Coût moyen |
| `wait_mean` | Temps d’attente moyen |
| `detour_mean` | Détour moyen |
| `soc_critical_rate` | Temps passé sous le seuil critique |
| `charge_requests` | Nombre de recharges |
| `utility_score` | Score synthétique global |

---

# 12. Résultats générés

Les résultats sont exportés automatiquement dans :

```text
outputs/
```

Fichiers générés :

| Fichier | Description |
|---|---|
| `results_runs.csv` | Résultats détaillés |
| `results_mean.csv` | Moyennes |
| `results_std.csv` | Écarts-types |

---

# 13. Choix techniques réalisés

Plusieurs choix ont été faits afin de conserver :

- un projet reproductible ;
- un temps d’exécution raisonnable ;
- une architecture simple à comprendre.

Exemples :

- stations synthétiques ;
- modèle énergétique simplifié ;
- REINFORCE au lieu d’algorithmes RL plus complexes ;
- implémentation FL maison sans framework externe.

---

# 14. Limites du projet

Le projet reste une preuve de concept expérimentale.

Certaines limites existent :

- pas de données EV réelles ;
- stations simulées ;
- réseau électrique non modélisé ;
- modèle RL volontairement simple ;
- absence de Smart Grid réel.

---

# 15. Perspectives

Améliorations possibles :

- intégration de données EV réelles ;
- utilisation de PPO ou SAC ;
- intégration Smart Grid ;
- réservation de bornes ;
- multi-agent RL ;
- personnalisation des modèles fédérés.

---

