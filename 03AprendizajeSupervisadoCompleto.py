# ——————————————————————————————————————————————————————————————————
# Diplomado de Inteligencia Artificial y Machine Learning
# Sesión de Aprendizaje Supervisado: Clasificación (Código Comentado)
# ——————————————————————————————————————————————————————————————————

# 1. Instalación (solo si fuera necesario en otro entorno)
# !pip install scikit-learn matplotlib --quiet

# 2. Importación de librerías
from sklearn.datasets import load_iris, load_wine        # Datasets de ejemplo
from sklearn.model_selection import train_test_split       # Para división en entrenamiento y prueba
from sklearn.linear_model import LogisticRegression        # Regresión logística
from sklearn.tree import DecisionTreeClassifier, plot_tree # Árbol de decisión y función para graficarlo
from sklearn.svm import SVC                                # Máquina de vectores de soporte
from sklearn.neighbors import KNeighborsClassifier         # K-Nearest Neighbors
from sklearn.naive_bayes import GaussianNB                 # Naive Bayes Gaussiano
from sklearn.metrics import classification_report, confusion_matrix  # Métricas de evaluación
import matplotlib.pyplot as plt                            # Visualización de resultados

# ——————————————————————————————————————————————————————————————————
# 3. Dataset Iris
# ——————————————————————————————————————————————————————————————————
print("=== Dataset Iris ===")
iris = load_iris()                   # Carga el dataset Iris
X, y = iris.data, iris.target        # X: variables predictoras, y: etiquetas de clase

# Dividimos en conjunto de entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ——————————————————————————————————————————————————————————————————
# 4. Ejemplo 1: Regresión Logística
# ——————————————————————————————————————————————————————————————————
print("\n=== Ejemplo 1: Regresión Logística ===")
model_logreg = LogisticRegression(max_iter=200)  # Creamos el modelo
model_logreg.fit(X_train, y_train)               # Entrenamos con los datos de Iris
y_pred_logreg = model_logreg.predict(X_test)     # Predecimos sobre el conjunto de prueba

# Mostramos métricas de clasificación y matriz de confusión
print(classification_report(y_test, y_pred_logreg))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_logreg))

# ——————————————————————————————————————————————————————————————————
# 5. Ejemplo 2: Árbol de Decisión
# ——————————————————————————————————————————————————————————————————
print("\n=== Ejemplo 2: Árbol de Decisión ===")
model_tree = DecisionTreeClassifier()  # Instanciamos árbol de decisión
model_tree.fit(X_train, y_train)       # Entrenamos el árbol
y_pred_tree = model_tree.predict(X_test)  # Predecimos

# Métricas y matriz
print(classification_report(y_test, y_pred_tree))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_tree))

# Visualización del árbol entrenado
plt.figure(figsize=(12,8))
plot_tree(
    model_tree,
    filled=True,
    feature_names=iris.feature_names,
    class_names=iris.target_names
)
plt.show()

# ——————————————————————————————————————————————————————————————————
# 6. Ejemplo 3: Máquina de Vectores de Soporte (SVM)
# ——————————————————————————————————————————————————————————————————
print("\n=== Ejemplo 3: SVM ===")
model_svm = SVC()               # Instanciamos SVM con configuración por defecto
model_svm.fit(X_train, y_train) # Entrenamos
y_pred_svm = model_svm.predict(X_test)  # Predecimos

print(classification_report(y_test, y_pred_svm))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred_svm))

# ——————————————————————————————————————————————————————————————————
# 7. Dataset Wine
# ——————————————————————————————————————————————————————————————————
print("\n=== Dataset Wine ===")
wine = load_wine()               # Carga el dataset Wine
X_wine, y_wine = wine.data, wine.target

# División entrenamiento/prueba
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(
    X_wine, y_wine, test_size=0.3, random_state=42
)

# ——————————————————————————————————————————————————————————————————
# 8. Ejemplo 4: K-Nearest Neighbors (KNN)
# ——————————————————————————————————————————————————————————————————
print("\n=== Ejemplo 4: KNN ===")
model_knn = KNeighborsClassifier(n_neighbors=5)  # 5 vecinos
model_knn.fit(X_train_wine, y_train_wine)        # Entrenamos
y_pred_knn = model_knn.predict(X_test_wine)      # Predecimos

print(classification_report(y_test_wine, y_pred_knn))
print("Matriz de confusión:\n", confusion_matrix(y_test_wine, y_pred_knn))

# ——————————————————————————————————————————————————————————————————
# 9. Ejemplo 5: Naive Bayes
# ——————————————————————————————————————————————————————————————————
print("\n=== Ejemplo 5: Naive Bayes ===")
model_nb = GaussianNB()             # Instanciamos Naive Bayes
model_nb.fit(X_train_wine, y_train_wine)  # Entrenamos
y_pred_nb = model_nb.predict(X_test_wine) # Predecimos

print(classification_report(y_test_wine, y_pred_nb))
print("Matriz de confusión:\n", confusion_matrix(y_test_wine, y_pred_nb))

# ——————————————————————————————————————————————————————————————————
# 10. Explicación General de los Modelos
# ——————————————————————————————————————————————————————————————————
print("\nExplicación General:")
print("- Regresión Logística: modela la probabilidad de cada clase usando una función logística.")
print("- Árbol de Decisión: aprende reglas de bifurcación basadas en los atributos.")
print("- SVM: busca el hiperplano que maximiza el margen entre clases.")
print("- KNN: asigna la etiqueta según la mayoría de los k vecinos más cercanos.")
print("- Naive Bayes: usa el teorema de Bayes asumiendo independencia condicional entre características.")

