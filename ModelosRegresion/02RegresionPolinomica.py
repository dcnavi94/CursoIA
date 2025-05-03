import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 1. Datos simulados: ciclos de carga vs % de capacidad restante
X = np.array([[0], [100], [200], [300], [400], [500], [600], [700], [800], [900], [1000]])
y = np.array([100, 98, 95, 91, 86, 80, 72, 60, 45, 30, 15])  # Capacidad en %

# 2. Crear modelo polinómico de grado 3
modelo = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
modelo.fit(X, y)

# 3. Rango de valores para graficar
X_plot = np.linspace(0, 1000, 300).reshape(-1, 1)
y_pred = modelo.predict(X_plot)

# 4. Predicción para 750 ciclos
ciclos_nuevos = np.array([[750]])
capacidad_predicha = modelo.predict(ciclos_nuevos)[0]

# 5. Visualización
plt.figure(figsize=(9, 5))
plt.scatter(X, y, color='blue', label="Datos reales")
plt.plot(X_plot, y_pred, color='red', label="Modelo polinómico (grado 3)")
plt.scatter(750, capacidad_predicha, color='purple', s=100, label=f"Predicción 750 ciclos: {capacidad_predicha:.1f}%")
plt.xlabel("Ciclos de carga")
plt.ylabel("Capacidad restante (%)")
plt.title("Degradación de batería vs Ciclos de carga (Regresión Polinómica grado 3)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. Evaluación
mse = mean_squared_error(y, modelo.predict(X))
print(f"⚠️ Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"🔋 Capacidad estimada para 750 ciclos: {capacidad_predicha:.2f}%")
