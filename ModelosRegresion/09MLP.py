import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# 1. Datos simulados: [educación, edad, horas/semana, tipo_empleo] → ingresos en USD
# tipo_empleo: 1 = obrero, 2 = técnico, 3 = administrativo, 4 = profesionista
X = np.array([
    [10, 25, 40, 1],
    [12, 30, 45, 2],
    [14, 35, 50, 3],
    [16, 40, 50, 4],
    [11, 28, 42, 1],
    [13, 32, 48, 2],
    [15, 38, 52, 3],
    [18, 45, 55, 4],
    [12, 26, 40, 2],
    [17, 42, 50, 4]
])
y = np.array([900, 1200, 1800, 3000, 1000, 1400, 2200, 3500, 1300, 3200])  # ingresos mensuales

# 2. Escalar los datos
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# 3. Entrenar el modelo MLP
mlp = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=3000, random_state=42)
mlp.fit(X_scaled, y_scaled)

# 4. Predicción para nueva persona: [educación=15, edad=36, 45h/sem, empleo=3]
persona_nueva = np.array([[15, 36, 45, 3]])
persona_scaled = scaler_X.transform(persona_nueva)
pred_scaled = mlp.predict(persona_scaled)
ingreso_predicho = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]

print(f"📈 Ingreso mensual estimado: ${ingreso_predicho:.2f} USD")

# 5. Visualización (educación vs ingreso para simplificar)
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], y, label='Datos reales', color='blue')
plt.scatter(persona_nueva[0][0], ingreso_predicho, color='red', s=100, label='Predicción')
plt.text(persona_nueva[0][0], ingreso_predicho + 100, f"${ingreso_predicho:.0f}", color='red', ha='center')
plt.xlabel("Años de educación")
plt.ylabel("Ingreso mensual (USD)")
plt.title("Predicción de ingreso económico con MLPRegressor")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
