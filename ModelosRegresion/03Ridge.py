import numpy as np
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt

# 1. Datos de entrenamiento: [km, años, potencia]
X = np.array([
    [50000, 3, 120],
    [70000, 4, 110],
    [30000, 2, 130],
    [90000, 6, 100],
    [60000, 4, 115],
    [120000, 7, 95],
    [80000, 5, 105],
    [40000, 2, 125],
    [100000, 6, 98],
    [35000, 1, 135]
])
y = np.array([18, 16, 22, 12, 17, 10, 14, 21, 11, 23])  # Precio real en miles de USD

# 2. Vehículo nuevo a analizar
vehiculo_nuevo = np.array([[75000, 4, 110]])

# 3. Definir los valores de alpha a probar
alphas = [0.01, 0.1, 1, 10, 100]

# 4. Crear y entrenar el modelo con selección automática de alpha
modelo_cv = RidgeCV(alphas=alphas, store_cv_values=True)
modelo_cv.fit(X, y)

# 5. Obtener la mejor alpha y hacer predicción
mejor_alpha = modelo_cv.alpha_
prediccion_nueva = modelo_cv.predict(vehiculo_nuevo)[0]

# 6. Mostrar resultados
print(f"✅ Mejor valor de alpha seleccionado automáticamente: {mejor_alpha}")
print(f"💰 Precio estimado del vehículo (75000 km, 4 años, 110 HP): ${prediccion_nueva:.2f} mil USD")

# 7. Graficar error de validación para cada alpha
cv_errors = modelo_cv.cv_values_.mean(axis=0)

plt.figure(figsize=(8, 5))
plt.plot(alphas, cv_errors, marker='o', color='green')
plt.title("Error de validación cruzada vs Alpha (RidgeCV)")
plt.xlabel("Alpha")
plt.ylabel("Error promedio (MSE)")
plt.grid(True)
plt.tight_layout()
plt.show()
