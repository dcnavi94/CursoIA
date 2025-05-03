import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. Datos de entrenamiento (vectores binarios)
X = np.array([
    [1, 1, 1, 1],  # Bob Esponja
    [1, 0, 0, 1],  # Patricio
    [1, 0, 1, 0],  # Pikachu
    [0, 0, 0, 1],  # Homero
    [0, 1, 0, 0],  # Sherlock
    [1, 1, 0, 0],  # Bugs Bunny
])

y = ["Bob Esponja", "Patricio", "Pikachu", "Homero", "Sherlock", "Bugs Bunny"]

# 2. Entrenar modelo de árbol
modelo = DecisionTreeClassifier(criterion="entropy", max_depth=3)
modelo.fit(X, y)

# 3. Definir características
nombres_caracteristicas = ["Animal", "Sombrero", "Amarillo", "Tonto"]

# 4. Predicción: usuario responde (ejemplo [1, 0, 1, 0])
respuesta_usuario = np.array([[1, 0, 1, 0]])  # Ej: Pikachu
prediccion = modelo.predict(respuesta_usuario)[0]

print("🔍 Tus respuestas fueron:")
for i, val in enumerate(respuesta_usuario[0]):
    print(f"   ➤ {nombres_caracteristicas[i]}: {'Sí' if val == 1 else 'No'}")

print(f"\n🎯 ¡El árbol predice que eres: {prediccion}!")

# 5. Graficar el árbol
plt.figure(figsize=(12, 6))
plot_tree(modelo,
          feature_names=nombres_caracteristicas,
          class_names=modelo.classes_,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("🌳 Árbol de decisión: Adivina el personaje")
plt.show()
