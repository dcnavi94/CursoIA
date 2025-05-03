# ——————————————————————————————————————————————————————————————————
# Ejemplo limpio de clustering con KMeans (scikit-learn)
# Suprime el warning de Loky y fija LOKY_MAX_CPU_COUNT correctamente
# ——————————————————————————————————————————————————————————————————

# 0. CONFIGURACIÓN DE ENTORNO ANTES DE IMPORTAR SKLEARN/JOBLIB
import os
import multiprocessing
import warnings

# Fija LOKY_MAX_CPU_COUNT al nº de núcleos lógicos de tu máquina
os.environ["LOKY_MAX_CPU_COUNT"] = str(multiprocessing.cpu_count())

# Suprime únicamente el warning de Loky que busca núcleos físicos
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores"
)

# 1. IMPORTAR LA CLASE KMeans
from sklearn.cluster import KMeans

# 2. PREPARACIÓN DE DATOS
#    Cada punto es una lista [x, y] en el plano
X = [
    [1, 2],
    [1, 4],
    [1, 0],
    [10, 2],
    [10, 4],
    [10, 0]
]

# 3. CONFIGURACIÓN Y ENTRENAMIENTO DEL MODELO
#    n_clusters=2 → queremos formar 2 grupos
#    random_state=42 → semilla fija para reproducibilidad
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# 4. RESULTADOS
#    .cluster_centers_ → coordenadas de los 2 centros hallados
#    .predict([...])   → asigna clúster a nuevos puntos
print("Cluster centers:", kmeans.cluster_centers_)
print("Grupo del punto [0, 0]:", kmeans.predict([[0, 0]]))
print("Grupo del punto [12, 3]:", kmeans.predict([[12, 3]]))
