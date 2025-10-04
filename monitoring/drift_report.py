import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# === 1. Cargar datasets crudos ===
X_train = pd.read_csv("C:\Documentos\heart-desease-mlops\data\processed\X_train.csv")
X_test = pd.read_csv("C:\Documentos\heart-desease-mlops\data\processed\X_test.csv")

# === 2. Definir columnas ===
cat_features = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
num_features = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]

# === 3. Crear preprocessor (igual al del entrenamiento) ===
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

# === 4. Ajustar y transformar ===
preprocessor.fit(X_train)

feature_names = preprocessor.get_feature_names_out()

X_train_transformed = pd.DataFrame(
    preprocessor.transform(X_train),
    columns=feature_names
)

X_test_transformed = pd.DataFrame(
    preprocessor.transform(X_test),
    columns=feature_names
)

# === 5. Generar Evidently report ===
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=X_train_transformed, current_data=X_test_transformed)
report.save_html("drift_report.html")

print("✅ Drift report generado con Evidently (con preprocesamiento interno)")