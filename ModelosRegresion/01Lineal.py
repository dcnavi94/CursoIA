import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ——————————————————————————————————————————————
# 1. Datos simulados: [m², habitaciones, ubicación]
# ——————————————————————————————————————————————
# Variables:       m²   habitaciones   zona (1 = baja, 2 = media, 3 = alta)
X = np.array([
    [59, 2, 1],
    [65, 2, 2],
    [80, 3, 2],
    [100, 3, 3],
    [124, 4, 3],
    [140, 4, 2],
    [100, 3, 1],
    [180, 5, 3]
])

# Precio real en miles de USD
y = np.array([100, 150, 200, 250, 300, 350, 230, 450])

# ——————————————————————————————————————————————
# 2. Crear y entrenar el modelo
# ——————————————————————————————————————————————
modelo = LinearRegression()
modelo.fit(X, y)

# ——————————————————————————————————————————————
# 3. Predicción para una casa de 100 m², 3 hab, zona media (2)
# ——————————————————————————————————————————————
nueva_casa = np.array([[100, 3, 2]])
precio_predicho = modelo.predict(nueva_casa)[0]

# ——————————————————————————————————————————————
# 4. Resultados del modelo
# ——————————————————————————————————————————————
print(f"🔍 Predicción: casa de 100 m², 3 hab., zona media → ${precio_predicho:.2f} mil USD")
print(f"📈 Coeficientes:")
print(f"  ➤ m² ..........: {modelo.coef_[0]:.2f}")
print(f"  ➤ Habitaciones: {modelo.coef_[1]:.2f}")
print(f"  ➤ Zona ........: {modelo.coef_[2]:.2f}")
print(f"📊 Intercepto ....: {modelo.intercept_:.2f}")

# ——————————————————————————————————————————————
# 5. Evaluación del modelo
# ——————————————————————————————————————————————
y_pred = modelo.predict(X)
mse = mean_squared_error(y, y_pred)
print(f"⚠️ Error Cuadrático Medio (MSE): {mse:.2f}")
