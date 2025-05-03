# ——————————————————————————————————————————————————
# Ejemplo sencillo de clasificación con DecisionTreeClassifier
# ——————————————————————————————————————————————————

# 1. PREPARACIÓN DE DATOS
# ------------------------

# X: lista de pesos (en gramos) de distintas frutas, cada muestra es una lista de un valor
X = [
    [100],  # Fruta de 100 g
    [150],  # Fruta de 150 g
    [200],  # Fruta de 200 g
    [250]   # Fruta de 250 g
]

# y: etiquetas asociadas a cada muestra en X
# 0 = manzana, 1 = naranja
y = [
    0,  # 100 g → manzana
    0,  # 150 g → manzana
    1,  # 200 g → naranja
    1   # 250 g → naranja
]

# ——————————————————————————————————————————————————
# 2. IMPORTACIÓN Y CREACIÓN DEL MODELO
# -------------------------------------

from sklearn.tree import DecisionTreeClassifier

# Instanciamos un clasificador de árbol de decisión
model = DecisionTreeClassifier()

# ——————————————————————————————————————————————————
# 3. ENTRENAMIENTO DEL MODELO
# ----------------------------

# Ajustamos el modelo a nuestros datos de entrenamiento (X, y)
model.fit(X, y)

# ——————————————————————————————————————————————————
# 4. PREDICCIÓN Y RESULTADOS
# ---------------------------

# Predecimos la etiqueta de una fruta de 120 g
pred_120 = model.predict([[120]])
print("¿Qué fruta es de 120 g?", "manzana" if pred_120[0] == 0 else "naranja")

# Predecimos la etiqueta de una fruta de 220 g
pred_220 = model.predict([[220]])
print("¿Qué fruta es de 220 g?", "manzana" if pred_220[0] == 0 else "naranja")
