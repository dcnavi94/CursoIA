import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# 1. Obtener precios actuales de Bitcoin y oro desde CoinGecko
def obtener_precios():
    url = 'https://api.coingecko.com/api/v3/simple/price'
    params = {
        'ids': 'bitcoin,ethereum',
        'vs_currencies': 'usd'
    }
    response = requests.get(url, params=params)
    data = response.json()
    precio_btc = data['bitcoin']['usd']
    precio_oro = data['ethereum']['usd']  # Simulamos el precio del oro con Ethereum
    return precio_btc, precio_oro

# 2. Simular datos históricos para entrenamiento
#    En un caso real, deberías obtener datos históricos reales
precios_btc = [28000, 28250, 27900, 28300, 28450, 28700, 28500, 28800, 29000, 28950]
volumenes = [1.2, 1.3, 1.1, 1.4, 1.5, 1.3, 1.2, 1.6, 1.7, 1.5]
precios_oro = [1900, 1910, 1895, 1905, 1920, 1930, 1925, 1940, 1945, 1935]

# 3. Preparar datos de entrenamiento
X = []
y = []
for i in range(3, len(precios_btc)):
    entrada = [
        precios_btc[i-3],  # Precio BTC -3
        precios_btc[i-2],  # Precio BTC -2
        precios_btc[i-1],  # Precio BTC -1
        volumenes[i-1],    # Volumen -1
        precios_oro[i-1]   # Precio oro -1
    ]
    X.append(entrada)
    y.append(precios_btc[i])  # Precio BTC actual (día a predecir)
X = np.array(X)
y = np.array(y)

# 4. Escalar datos
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# 5. GridSearch para encontrar mejores hiperparámetros
param_grid = {
    'C': [1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5]
}
grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=3)
grid_search.fit(X_scaled, y_scaled)

# 6. Mejor modelo entrenado
modelo = grid_search.best_estimator_

# 7. Obtener precios actuales para predicción
precio_btc_actual, precio_oro_actual = obtener_precios()
entrada_nueva = np.array([[precios_btc[-3], precios_btc[-2], precios_btc[-1], volumenes[-1], precio_oro_actual]])
entrada_nueva_scaled = scaler_X.transform(entrada_nueva)
y_pred_scaled = modelo.predict(entrada_nueva_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]

# 8. Visualización
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

# 9. Resultados en consola
print("🔍 Mejor modelo:")
print(grid_search.best_params_)
print(f"📈 Predicción para mañana: ${y_pred:.2f} USD")
