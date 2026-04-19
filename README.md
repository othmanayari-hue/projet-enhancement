# 🔬 WAGF-Fusion: Wavelet Guided Attention Module for Skin Cancer Classification

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![DeepLearning](https://img.shields.io/badge/AI-Deep%20Learning-F37626.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **Un système avancé de diagnostic assisté par ordinateur (CAD) pour la classification des lésions cutanées cancéreuses, exploitant l'apprentissage profond et les transformées en ondelettes.**

---

## 📋 Table des Matières
1. [À propos du Projet](#-à-propos-du-projet)
2. [Caractéristiques Principales](#-caractéristiques-principales)
3. [Architecture du Modèle](#-architecture-du-modèle)
4. [Jeux de Données](#-jeux-de-données)
5. [Résultats](#-résultats)
6. [Technologies Utilisées](#-technologies-utilisées)
7. [Installation et Utilisation](#-installation-et-utilisation)
8. [Auteurs & Remerciements](#-auteurs--remerciements)

---

## 📖 À propos du Projet

Le cancer de la peau est l'une des formes de cancer les plus menaçantes au monde. Un diagnostic précoce et précis est crucial pour l'efficacité du traitement et la survie des patients. 

Ce projet propose d'améliorer les systèmes de diagnostic assisté par ordinateur (CAD) afin d'aider les dermatologues en automatisant l'identification des lésions cutanées malignes à partir d'images dermoscopiques à haute résolution. 

---

## ✨ Caractéristiques Principales

Notre solution se démarque par l'intégration de plusieurs mécanismes de pointe :
* **Backbone DenseNet-121** : Utilise une connectivité dense pour une réutilisation optimale des caractéristiques et l'atténuation du problème de disparition du gradient.
* **Transformées en Ondelettes Discrètes (DWT)** : Extraction des caractéristiques multi-résolution pour capturer des motifs de texture fins et des limites structurelles nettes (Ondelettes de Haar).
* **Module SaFA (Symmetry-aware Feature Attention)** : Un mécanisme d'attention novateur intégrant des couches LSTM pour capturer la symétrie spatiale de la lésion et ses variations sémantiques.
* **Fusion basée sur le Gradient** : Fusion dynamique et adaptative des caractéristiques des ondelettes et de l'attention sans augmenter la complexité des paramètres.

---

## 🧠 Architecture du Modèle

Notre architecture hybride combine l'extraction profonde de caractéristiques avec des modules d'attention spécialisés :
1. Les images prétraitées (256x256x3) passent dans **DenseNet-121** pour extraire une carte de caractéristiques globale.
2. Le modèle applique la transformée de Haar pour récupérer les informations spatiales et fréquentielles.
3. Le module **SaFA** analyse les variations sémantiques en largeur et en hauteur à l'aide de blocs F-DaB et SaB.
4. Une **fusion adaptative (Gradient-based Feature Fusion)** combine ces éléments de manière ciblée juste avant la classification finale.

---

## 📊 Jeux de Données

Le modèle a été entraîné et évalué sur deux bases de données de référence :
* **HAM10000** : Environ 10 015 images réparties en 7 catégories de diagnostic.
* **ISIC 2019** : Intégré pour enrichir le contexte, augmenter la diversité des données et pallier le déséquilibre des classes.

Grâce à des techniques d'augmentation de données, le dataset final d'entraînement compte plus de 35 346 images.

---

## 📈 Résultats

Après un entraînement efficace sur seulement 25 epochs avec des données augmentées, le modèle a démontré des performances très compétitives :
* **Précision globale (Accuracy)** : `90.87%`
* **Score F1** : `91.62%`

*(N'hésitez pas à insérer ici une capture d'écran de votre matrice de confusion ou du graphique de la Heatmap d'attention SaFA de la page 22 du rapport)*

---

## 💻 Technologies Utilisées

* **Langage de Programmation** : Python 3
* **Bibliothèque de Deep Learning** : TensorFlow
* **Environnements** : Google Colab (Support GPU) / Machine Virtuelle (Intel Xeon) pour l'entraînement

---

## 🚀 Installation et Utilisation

### 1. Cloner le dépôt
```bash
git clone [https://github.com/votre-nom-d-utilisateur/WAGF-Fusion.git](https://github.com/votre-nom-d-utilisateur/WAGF-Fusion.git)
cd WAGF-Fusion
