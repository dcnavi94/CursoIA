import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Datos simulados: [ppm_CO2, temperatura, humedad]
X = np.array([
    [250, 24, 40],
    [400, 27, 45],
    [550, 29, 50],
    [700, 30, 55],
    [900, 32, 60],
    [1100, 34, 65],
    [1250, 35, 70],
    [1350, 36, 75],
    [300, 23, 42],
    [650, 31, 57],
    [1200, 33, 69]
])

# Etiquetas: 0 = buena, 1 = moderada, 2 = mala
y = np.array([0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])

# 2. División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Modelo XGBoost
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# 4. Predicción
y_pred = model.predict(X_test)

# 5. Resultados
print("📊 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\n📋 Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=["Buena", "Moderada", "Mala"]))

# 6. Simulación: nuevo dato en tiempo real desde Arduino
nueva_muestra = np.array([[800, 31, 60]])  # Ejemplo de sensores
prediccion = model.predict(nueva_muestra)[0]
clase = ["Buena", "Moderada", "Mala"][prediccion]
print(f"\n🧪 Predicción en tiempo real → Calidad del aire: {clase}")
