# Requisitos: pip install tensorflow scikit-learn pandas numpy matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# 1. Simular datos de tráfico por hora durante 30 días (720 horas)
np.random.seed(42)
horas = 24 * 30
base = np.sin(np.linspace(0, 30 * 2 * np.pi, horas)) * 50 + 100
ruido = np.random.normal(0, 10, horas)
flujo = np.clip(base + ruido, 20, 200)

df = pd.DataFrame({"flujo": flujo})

# 2. Escalar datos
scaler = MinMaxScaler()
datos_scaled = scaler.fit_transform(df)

# 3. Crear secuencias para LSTM (ventana de 24 horas)
def crear_secuencias(data, pasos=24):
    X, y = [], []
    for i in range(len(data) - pasos):
        X.append(data[i:i+pasos])
        y.append(data[i+pasos])
    return np.array(X), np.array(y)

X, y = crear_secuencias(datos_scaled, pasos=24)

# 4. División de entrenamiento y prueba
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Crear modelo LSTM
model = Sequential([
    LSTM(50, activation='relu', input_shape=(24, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 6. Entrenar
model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

# 7. Predecir y evaluar
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

# 8. Visualizar resultados
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label='Real')
plt.plot(y_pred_inv, label='Predicho')
plt.title(f"Predicción de Tráfico Vehicular (RMSE: {rmse:.2f})")
plt.xlabel("Horas")
plt.ylabel("Flujo Vehicular")
plt.legend()
plt.tight_layout()
plt.show()
