import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# 1. Datos: [edad, presión, colesterol, frecuencia cardiaca, IMC, glucosa, estrés]
X = np.array([
    [50, 140, 230, 80, 28.5, 95, 1],
    [45, 130, 210, 76, 27.0, 85, 0],
    [60, 160, 250, 85, 30.1, 110, 1],
    [35, 120, 190, 72, 25.0, 80, 0],
    [55, 145, 240, 83, 29.0, 105, 1],
    [48, 135, 220, 78, 26.5, 92, 0],
    [65, 170, 270, 88, 31.2, 115, 1],
    [42, 125, 200, 74, 26.0, 88, 0],
    [58, 150, 235, 82, 28.9, 102, 1],
    [38, 122, 195, 75, 24.5, 84, 0]
])

# Diagnóstico: 1 = enfermedad cardíaca, 0 = no
y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

# 2. Crear modelo logístico con L1 (Lasso)
modelo_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
modelo_l1.fit(X, y)

# 3. Nuevo paciente
paciente_nuevo = np.array([[53, 142, 225, 81, 28.0, 100, 1]])
prob_enfermedad = modelo_l1.predict_proba(paciente_nuevo)[0][1]
diagnostico = modelo_l1.predict(paciente_nuevo)[0]

# 4. Resultados
print("🩺 Evaluación de paciente: 53 años, presión 142, colesterol 225, etc.")
print(f"📈 Probabilidad estimada de enfermedad cardíaca: {prob_enfermedad:.2%}")
print(f"✅ Diagnóstico final: {'POSITIVO (1)' if diagnostico == 1 else 'NEGATIVO (0)'}")

# 5. Mostrar importancia de variables
coef = modelo_l1.coef_[0]
variables = ["Edad", "Presión", "Colesterol", "F. Cardiaca", "IMC", "Glucosa", "Estrés"]

plt.figure(figsize=(9, 5))
plt.bar(variables, coef, color='darkorange')
plt.title("Importancia de variables (Clasificación con L1)")
plt.ylabel("Coeficiente")
plt.axhline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.grid(True)
plt.show()
