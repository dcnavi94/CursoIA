# ——————————————————————————————————————————————————
# Ejemplo sencillo de regresión lineal con scikit-learn
# ——————————————————————————————————————————————————

# Importamos la clase LinearRegression para crear nuestro modelo
from sklearn.linear_model import LinearRegression
# Importamos numpy para manejar arreglos numéricos de forma eficiente
import numpy as np
# Importamos matplotlib para la visualización de datos
import matplotlib.pyplot as plt

# ——————————————————————————————————————————————————
# 1. PREPARACIÓN DE LOS DATOS
# ——————————————————————————————————————————————————

# X: array bidimensional (n_muestras, 1) con los metros cuadrados de cada casa
X = np.array([
    [59],    # Casa de 59 m²
    [65],    # Casa de 65 m²
    [80],    # Casa de 80 m²
    [100],   # Casa de 100 m²
    [124],   # Casa de 124 m²
    [140],   # Casa de 140 m²
    [100],   # Otra casa de 100 m² para ver variabilidad
    [180]    # Casa de 180 m²
])

# y: array unidimensional con los precios (en miles de dólares) correspondientes
y = np.array([
    100,  # Precio de la casa de 59 m²
    150,  # Precio de la casa de 65 m²
    200,  # Precio de la casa de 80 m²
    250,  # Precio de la casa de 100 m²
    300,  # Precio de la casa de 124 m²
    350,  # Precio de la casa de 140 m²
    400,  # Precio distinto para otra de 100 m²
    450   # Precio de la casa de 180 m²
])

# ——————————————————————————————————————————————————
# 2. CREACIÓN Y ENTRENAMIENTO DEL MODELO
# ——————————————————————————————————————————————————

# Instanciamos el modelo de regresión lineal
modelo = LinearRegression()
# Entrenamos (ajustamos) el modelo con nuestros datos X (entrada) y y (salida)
modelo.fit(X, y)

# ——————————————————————————————————————————————————
# 3. PREDICCIÓN
# ——————————————————————————————————————————————————

# Definimos un nuevo tamaño de casa: 110 m²
nuevo_tamano = [[110]]
# Pedimos al modelo que estime el precio para 110 m²
precio_predicho = modelo.predict(nuevo_tamano)
print(f"Precio predicho para 110 m²: ${precio_predicho[0]:.2f} mil")

# ——————————————————————————————————————————————————
# 4. VISUALIZACIÓN DE RESULTADOS
# ——————————————————————————————————————————————————

plt.figure(figsize=(8, 6))                        # Creamos una figura de 8×6 pulgadas
plt.scatter(X, y, color='blue', label='Datos reales')               # Puntos reales
plt.plot(X, modelo.predict(X), color='red', label='Línea de regresión')  # Línea ajustada
plt.scatter(nuevo_tamano, precio_predicho, color='green', s=100, label='Predicción')  # Punto de predicción
plt.xlabel('Metros cuadrados')                    # Etiqueta eje X
plt.ylabel('Precio (miles $)')                    # Etiqueta eje Y
plt.title('Regresión Lineal: Precio vs Tamaño')   # Título del gráfico
plt.legend()                                      # Mostramos la leyenda
plt.grid(True)                                    # Cuadrícula para mejor lectura
plt.show()                                        # Renderizamos la gráfica

# ——————————————————————————————————————————————————
# 5. INTERPRETACIÓN DE COEFICIENTES
# ——————————————————————————————————————————————————

# coef_[0] es la pendiente: indica cuánto (en miles $) cambia el precio por cada metro cuadrado adicional
print(f"Pendiente: {modelo.coef_[0]:.2f}")
# intercept_ es el valor de precio estimado cuando el tamaño es cero (x=0)
print(f"Intercepto: {modelo.intercept_:.2f}")
