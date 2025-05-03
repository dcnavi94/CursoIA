# pip install folium scikit-learn pandas numpy matplotlib

import numpy as np
import pandas as pd
import folium
from sklearn.cluster import KMeans
from folium.plugins import MarkerCluster

# ==========================
# 1. Simular ubicaciones de demanda
# ==========================
np.random.seed(42)
n = 100  # 100 registros de paradas usadas

# Coordenadas alrededor de San José Iturbide
latitudes = np.random.normal(loc=21.002, scale=0.005, size=n)
longitudes = np.random.normal(loc=-100.387, scale=0.005, size=n)

datos = pd.DataFrame({
    "lat": latitudes,
    "lon": longitudes
})

# ==========================
# 2. Agrupar en 5 rutas (clústers)
# ==========================
kmeans = KMeans(n_clusters=5, random_state=42)
datos["ruta"] = kmeans.fit_predict(datos[["lat", "lon"]])

# ==========================
# 3. Visualizar en mapa con colores
# ==========================
colores = ["red", "blue", "green", "orange", "purple"]
mapa = folium.Map(location=[21.002, -100.387], zoom_start=14)

for i, fila in datos.iterrows():
    ruta = fila["ruta"]
    folium.CircleMarker(
        location=[fila["lat"], fila["lon"]],
        radius=6,
        color=colores[int(ruta)],
        fill=True,
        fill_opacity=0.7,
        popup=f"Ruta {ruta + 1}"
    ).add_to(mapa)

# ==========================
# 4. Guardar mapa
# ==========================
mapa.save("DemoIaSji/rutas_coloreadas_san_jose_iturbide.html")
print("✅ Mapa guardado como 'DemoIaSji/rutas_coloreadas_san_jose_iturbide.html'")
