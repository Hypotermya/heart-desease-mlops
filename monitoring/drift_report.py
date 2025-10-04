import pandas as pd
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric

# Cargar los datasets ya procesados (OneHot, escalados, etc.)
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

# Crear el reporte de drift
report = Report(metrics=[
    DatasetDriftMetric()
])

report.run(reference_data=X_train, current_data=X_test)

# Guardar el reporte
report.save_html("drift_report.html")

print("✅ Drift report generado con Evidently 0.7.14")