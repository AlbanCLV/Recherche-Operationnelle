# 🚚 Projet Mobilité Multimodale Intelligente – Optimisation des tournées de livraison

## 🌍 Introduction

Depuis les années 1990, la nécessité de réduire la consommation énergétique et les émissions de gaz à effet de serre est devenue une priorité mondiale. Les politiques publiques ont initié de nombreuses actions, comme le protocole de Kyoto ou les objectifs nationaux de réduction des émissions. Cependant, atteindre ces objectifs nécessite une transformation profonde des comportements, en particulier dans les domaines du transport et de la logistique.

## 📦 Sujet du projet

Dans le cadre d’un appel à manifestation d’intérêt lancé par l’ADEME, notre structure **CesiCDP** a été mandatée pour proposer des solutions innovantes en matière de **mobilité intelligente**. L’objectif est de concevoir et tester des algorithmes d’optimisation de tournées de livraison qui soient économes en énergie, tout en répondant aux contraintes logistiques modernes.

### 🎯 Objectif principal

> Calculer une tournée optimale sur un réseau routier reliant un ensemble de villes, en minimisant la durée ou le coût total du trajet, tout en respectant diverses contraintes.

---

## 🧠 Contenu de l’étude

### ✔️ Modélisation du problème

- Formulation du problème de tournée comme une **variation du problème du voyageur de commerce (TSP)**.
- Analyse de **la complexité algorithmique**.
- Prise en compte de contraintes réalistes :
  - Fenêtres temporelles
  - Coûts différenciés ou interdictions de certaines routes
  - Utilisation de plusieurs véhicules avec capacité limitée
  - Équilibrage des tournées

### ⚙️ Implémentation technique (Python)

- Génération d’instances aléatoires (graphes avec paramètres ajustables).
- Implémentation d'au moins **deux méthodes de résolution** :
  - Méthode exacte (ex: Branch & Bound, programmation linéaire)
  - Méthode heuristique (ex: algorithmes gloutons, génétiques, colonies de fourmis)
- Étude expérimentale :
  - Tests sur différentes tailles d’instances
  - Analyse comparative des performances
  - Propositions d’améliorations

---
## 🛠️ Outils & technologies

- **Langage** : Python 3
- **Librairies principales** : NetworkX, NumPy, Matplotlib, Pandas
- **Environnement** : Jupyter Notebook
- **Normes de code** : Respect des conventions [PEP8](https://peps.python.org/pep-0008/)

---

## 📊 Évaluation

| Note cible | Critères attendus |
|------------|--------------------|
| A          | Programme fonctionnel + complexité + analyse statistique |
| B          | Fonctionnel + analyse statistique |
| C          | Fonctionnel uniquement |
| D          | Non fonctionnel |

---

## 🙌 Remerciements

Ce projet est réalisé dans le cadre du cursus ingénieur à **CESI**
