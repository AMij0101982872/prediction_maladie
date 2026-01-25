import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Charger le dataset
data = pd.read_csv("CHD.csv")
X = data.drop("chd", axis=1)
y = data["chd"]

# Définir les colonnes numériques et catégorielles
numeric_features = ["sbp", "tobacco", "ldl", "adiposity", "typea", "obesity", "alcohol", "age"]
categorical_features = ["famhist"]

# Créer les transformateurs
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

# Préprocesseur
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Créer le pipeline complet
pipe_knn = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=123)),
    ("pca", PCA(n_components=0.95)),
    ("knn", KNeighborsClassifier())
])

# Entraîner le pipeline sur toutes les données
pipe_knn.fit(X, y)

# Sauvegarder le pipeline entraîné
joblib.dump(pipe_knn, "model.pkl")
print("Modèle sauvegardé avec succès sous le nom Model.pkl !")
