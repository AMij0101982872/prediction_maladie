# 🧠 Projet de Machine Learning – Prédiction de Maladie

## 📌 Objectif du projet

Ce projet vise à **prédire la présence d’une maladie (classe 1)** chez des patients à partir de données cliniques. Le principal défi est le **déséquilibre des classes**, la classe malade étant minoritaire.

L’objectif métier prioritaire est donc :
👉 **Maximiser le rappel et le F1-score de la classe 1**, afin de détecter le plus de patients malades possible.

## ⚙️ Prétraitement des données

Les données contiennent des **variables numériques et catégorielles**.

### 🔹 Étapes de preprocessing

* Imputation des valeurs manquantes

  * Numériques : moyenne
  * Catégorielles : modalité la plus fréquente
* Encodage des variables catégorielles (OneHotEncoder)
* Standardisation des variables numériques (StandardScaler)

Ces étapes sont regroupées dans un **ColumnTransformer**, intégré directement dans les pipelines.

---

##  Modèles testés

### 1️ Logistic Regression – Baseline

* Pipeline : Preprocessing + LogisticRegression
* Sans gestion du déséquilibre

📉 Résultat :

* Bonne performance sur la classe 0
* Mauvais rappel sur la classe 1

---

### 2️ Logistic Regression avec ACP (PCA 95%)

* Réduction de dimension après preprocessing

📉 Résultat :

* Baisse globale des performances
* Forte dégradation du rappel de la classe 1

 **Conclusion** : l’ACP a supprimé des variables discriminantes importantes.

---

### 3️ Logistic Regression équilibrée + GridSearchCV ✅

* `class_weight="balanced"`
* Optimisation des hyperparamètres via GridSearchCV

📈 Résultat :

* Amélioration significative du rappel de la classe 1
* Meilleur compromis biais / variance
* Modèle interprétable

---

### 4️ KNN + SMOTE + PCA + GridSearchCV

* Pipeline Imbalanced-learn (`ImbPipeline`)
* Sur-échantillonnage avec SMOTE
* PCA pour réduction de dimension
* Optimisation de `n_neighbors`

📈 Résultat :

* Bon rappel de la classe 1
* Performance globale correcte
* Modèle sensible au bruit et moins interprétable

---

## 📊 Comparaison des modèles Logistic Regression

| Modèle                  | Accuracy | Rappel classe 1 | F1 classe 1 | Commentaire                  |
| ----------------------- | -------- | --------------- | ----------- | ---------------------------- |
| Baseline                | ≈ 0.71   | ≈ 0.36          | ≈ 0.46      | Détecte mal les malades      |
| Avec PCA (95%)          | ≈ 0.67   | ≈ 0.26          | ≈ 0.36      | Perte d’information critique |
| Balanced + GridSearchCV | ≈ 0.70   | ≈ 0.64          | ≈ 0.60      | ⭐ Meilleur compromis         |

---

##  Modèle final retenu

 **Logistic Regression avec `class_weight='balanced'` et hyperparamètres optimisés**

### Pourquoi ce choix ?

* Très bon rappel sur la classe malade
* F1-score satisfaisant
* Robuste sur validation croisée
* Interprétable (important en contexte médical)

---

## 🚀 Entraînement final

Le modèle final est **entraîné sur l’ensemble des données disponibles** afin de maximiser l’apprentissage avant déploiement.

```python
best_model.fit(X, y)
```

---

##  Améliorations possibles

* Tester XGBoost / LightGBM avec gestion du déséquilibre
* Ajuster le seuil de décision (0.5 → 0.3)
* Analyse SHAP pour interprétabilité avancée
* Validation externe sur nouvelles données

---

## 🛠️ Technologies utilisées

* Python
* scikit-learn
* imbalanced-learn
* pandas / numpy
* matplotlib / seaborn

---

## 👤 Auteur

**Mobio Ivan Junior Ake**
Machine Learning & Data Science



 *Projet prêt pour GitHub /
 https://predictionmaladie-chd.streamlit.app/
