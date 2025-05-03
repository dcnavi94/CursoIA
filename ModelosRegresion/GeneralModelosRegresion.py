import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# ——————————————————————————————————————————————
# 1. Datos: m² vs precio (en miles de USD)
# ——————————————————————————————————————————————
X = np.array([[59], [65], [80], [100], [124], [140], [100], [180]])
y = np.array([100, 150, 200, 250, 300, 350, 400, 450])

# ——————————————————————————————————————————————
# 2. División de datos
# ——————————————————————————————————————————————
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
X_plot = np.linspace(50, 200, 200).reshape(-1, 1)

# ——————————————————————————————————————————————
# 3. Modelos de regresión
# ——————————————————————————————————————————————
modelos = {
    "Lineal": LinearRegression(),
    "Polinómica": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "Árbol": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(kernel='linear'),
    "KNN": KNeighborsRegressor(n_neighbors=3),
    "MLP": MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
}

# ——————————————————————————————————————————————
# 4. Entrenamiento, predicción y visualización
# ——————————————————————————————————————————————
fig, axes = plt.subplots(3, 3, figsize=(18, 12))
axes = axes.flatten()
predicciones_100m2 = {}

for i, (nombre, modelo) in enumerate(modelos.items()):
    modelo.fit(X_train, y_train)
    y_pred_plot = modelo.predict(X_plot)
    y_pred_test = modelo.predict(X_test)
    y_100 = modelo.predict([[100]])[0]
    predicciones_100m2[nombre] = y_100
    mse = mean_squared_error(y_test, y_pred_test)

    axes[i].scatter(X, y, color='blue', label="Datos reales")
    axes[i].plot(X_plot, y_pred_plot, color='red', label="Predicción")
    axes[i].scatter([100], [y_100], color='green', s=100, label="Predicción 100 m²")
    axes[i].text(100, y_100 + 10, f"${y_100:.1f}k", color='green', fontsize=10, ha='center')
    axes[i].set_title(f"{nombre} (MSE={mse:.2f})")
    axes[i].set_xlabel("Metros cuadrados")
    axes[i].set_ylabel("Precio (miles USD)")
    axes[i].legend()
    axes[i].grid(True)

plt.suptitle("Modelos de regresión: Precio de casa vs Tamaño (predicción para 100 m²)", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()

# ——————————————————————————————————————————————
# 5. Mostrar en consola las predicciones para 100 m²
# ——————————————————————————————————————————————
print("\n📌 Predicción del precio para una casa de 100 m²:")
for nombre, pred in predicciones_100m2.items():
    print(f"{nombre:<15}: ${pred:.2f} mil USD")
