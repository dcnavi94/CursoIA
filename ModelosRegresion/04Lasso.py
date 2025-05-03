import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# 1. Datos simulados: [edad, presión, colesterol, frecuencia cardiaca, IMC, glucosa, estrés]
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

# Riesgo de enfermedad cardíaca (valor entre 0 y 100%)
y = np.array([70, 60, 90, 40, 80, 65, 95, 55, 85, 45])

# 2. Crear modelo Lasso
modelo_lasso = Lasso(alpha=5)
modelo_lasso.fit(X, y)

# 3. Nuevo paciente a evaluar
paciente_nuevo = np.array([[53, 142, 225, 81, 28.0, 100, 1]])
riesgo_predicho = modelo_lasso.predict(paciente_nuevo)[0]

# 4. Imprimir resultados
print("📋 Paciente nuevo: 53 años, presión 142, colesterol 225, frecuencia 81, IMC 28, glucosa 100, estrés: sí")
print(f"❤️ Riesgo estimado de enfermedad cardíaca: {riesgo_predicho:.1f}%")
print("\n📊 Importancia de variables (coeficientes):")
print(modelo_lasso.coef_)

# 5. Visualización de importancia de variables
nombres = ["Edad", "Presión", "Colesterol", "F. Cardiaca", "IMC", "Glucosa", "Estrés"]

plt.figure(figsize=(9, 5))
plt.bar(nombres, modelo_lasso.coef_, color='crimson')
plt.title("Importancia de variables en predicción médica (Lasso)")
plt.ylabel("Coeficiente")
plt.axhline(0, color='gray', linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.show()
