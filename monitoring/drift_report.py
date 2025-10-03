import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Carga tus datasets
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

# Generar reporte
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=X_train, current_data=X_test)
report.save_html("drift_report.html")

print("✅ Reporte generado en drift_report.html")
