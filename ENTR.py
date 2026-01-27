from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib

# Pipeline KNN
pipe_knn = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=123)),
    ("pca", PCA(n_components=0.95)),
    ("knn", KNeighborsClassifier())
])

# Paramètres à tester pour KNN
param_grid = {
    "knn__n_neighbors": [3, 5, 7, 9],   # nombre de voisins
    "knn__weights": ["uniform", "distance"]  # pondération
}

# GridSearch sur pipe_knn
grid = GridSearchCV(
    pipe_knn,
    param_grid,
    cv=5,
    scoring="recall",
    n_jobs=-1
)

# Entraînement
grid.fit(X_train, y_train)

# Prédiction
y_pred_knn = grid.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred_knn))

# Réentraîner sur toutes les données avec le meilleur modèle
best_knn = grid.best_estimator_
best_knn.fit(X, y)

# Sauvegarde
joblib.dump(best_knn, "Model1.pkl")
print("✅ Modèle sauvegardé avec succès sous le nom Model1.pkl !")
