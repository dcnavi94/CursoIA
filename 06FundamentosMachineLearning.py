# ——————————————————————————————————————————————————————————————
# Paso 1: Importar librerías necesarias
# ——————————————————————————————————————————————————————————————
from sklearn.datasets import load_iris                     # Dataset de ejemplo
from sklearn.model_selection import train_test_split        # Para dividir en train/test
from sklearn.tree import DecisionTreeClassifier            # Clasificador de árbol de decisión
from sklearn.neighbors import KNeighborsClassifier         # Clasificador KNN
from sklearn.metrics import accuracy_score                 # Métrica de precisión

# ——————————————————————————————————————————————————————————————
# Paso 2: Cargar el dataset Iris
# ——————————————————————————————————————————————————————————————
iris = load_iris()    # Carga características y etiquetas
X, y = iris.data, iris.target

# ——————————————————————————————————————————————————————————————
# Paso 3: Dividir datos en entrenamiento (70%) y prueba (30%)
# ——————————————————————————————————————————————————————————————
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,    # 30% de los datos para test
    random_state=42   # semilla fija para reproducibilidad
)

# ——————————————————————————————————————————————————————————————
# Paso 4: Árbol de Decisión — creación, entrenamiento y evaluación
# ——————————————————————————————————————————————————————————————
model_dt = DecisionTreeClassifier()      # instanciamos el árbol
model_dt.fit(X_train, y_train)           # ajustamos al conjunto de entrenamiento

y_pred_dt = model_dt.predict(X_test)     # predecimos etiquetas para el test
acc_dt = accuracy_score(y_test, y_pred_dt)
print("Precisión del Árbol de Decisión:", acc_dt)

# ——————————————————————————————————————————————————————————————
# Paso 5: K-Nearest Neighbors — creación, entrenamiento y evaluación
# ——————————————————————————————————————————————————————————————
knn_model = KNeighborsClassifier()       # instanciamos KNN con k=5 por defecto
knn_model.fit(X_train, y_train)          # ajustamos al conjunto de entrenamiento

knn_pred = knn_model.predict(X_test)     # predecimos etiquetas para el test
acc_knn = accuracy_score(y_test, knn_pred)
print("Precisión con KNN:", acc_knn)
