import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd

# 1. Dataset de películas (simplificado)
peliculas = pd.DataFrame({
    "titulo": [
        "Inception", "Titanic", "Mad Max", "Toy Story", "The Matrix",
        "Finding Nemo", "Gladiator", "The Godfather", "Shrek", "Interstellar"
    ],
    "genero": [1, 1, 2, 3, 1, 3, 2, 1, 3, 1],  # 1: acción/drama, 2: acción/historia, 3: animación
    "duracion": [148, 195, 120, 81, 136, 100, 155, 175, 90, 169],  # en minutos
    "calificacion": [8.8, 7.8, 8.1, 8.3, 8.7, 8.1, 8.5, 9.2, 7.9, 8.6]  # IMDb rating
})

# 2. Características que usaremos para medir similitud
X = peliculas[["genero", "duracion", "calificacion"]].values

# 3. Modelo KNN (vecinos más cercanos)
modelo = NearestNeighbors(n_neighbors=4, metric='euclidean')
modelo.fit(X)

# 4. Buscar películas similares a "The Matrix"
indice_pelicula = peliculas[peliculas["titulo"] == "The Matrix"].index[0]
distancias, indices = modelo.kneighbors([X[indice_pelicula]])

# 5. Mostrar recomendaciones
print("🎥 Película base: The Matrix")
print("\n📌 Recomendaciones similares:")
for idx in indices[0][1:]:  # omitimos la primera (es la misma)
    print(f"• {peliculas.loc[idx, 'titulo']}")
