import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# 1. Simular datos: [precio_BTC, volumen, precio_oro]
data = np.array([
    [28000, 1.2, 1900],
    [28250, 1.3, 1910],
    [27900, 1.1, 1895],
    [28300, 1.4, 1905],
    [28450, 1.5, 1920],
    [28700, 1.3, 1930],
    [28500, 1.2, 1925],
    [28800, 1.6, 1940],
    [29000, 1.7, 1945],
    [28950, 1.5, 1935],
    [29100, 1.8, 1950],
    [29200, 1.9, 1960],
    [29400, 2.0, 1970],
    [29500, 2.1, 1975]
])

# 2. Crear variables X (entrada) y y (precio BTC siguiente día)
X = []
y = []
for i in range(3, len(data)):
    entrada = [
        data[i-3][0],  # precio BTC -3
        data[i-2][0],  # precio BTC -2
        data[i-1][0],  # precio BTC -1
        data[i-1][1],  # volumen -1
        data[i-1][2]   # precio oro -1
    ]
    X.append(entrada)
    y.append(data[i][0])  # precio BTC actual (día a predecir)
X = np.array(X)
y = np.array(y)

# 3. Escalar datos
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# 4. GridSearch para encontrar mejores hiperparámetros
param_grid = {
    'C': [1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5]
}
grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=3)
grid_search.fit(X_scaled, y_scaled)

# 5. Mejor modelo entrenado
modelo = grid_search.best_estimator_

# 6. Predicción con datos nuevos
entrada_nueva = np.array([[29100, 29200, 29400, 2.0, 1970]])
entrada_nueva_scaled = scaler_X.transform(entrada_nueva)
y_pred_scaled = modelo.predict(entrada_nueva_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]

# 7. Visualización
dias = list(range(len(y)))
plt.figure(figsize=(10, 5))
plt.plot(dias, y, marker='o', label='Precio real BTC')
plt.scatter(len(y), y_pred, color='red', label='Predicción', zorder=5)
plt.text(len(y), y_pred + 100, f"${y_pred:.0f}", color='red', ha='center')
plt.title("SVR: Predicción del precio de Bitcoin con volumen y precio del oro")
plt.xlabel("Día")
plt.ylabel("Precio BTC (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 8. Resultados en consola
print("🔍 Mejor modelo:")
print(grid_search.best_params_)
print(f"📈 Predicción para mañana: ${y_pred:.2f} USD")
