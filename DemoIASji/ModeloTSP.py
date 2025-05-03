# Requisitos previos:
# pip install osmnx networkx folium scikit-learn pandas numpy

import osmnx as ox
import networkx as nx
import folium
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ========== 1. Definir paradas estratégicas ==========
paradas = pd.DataFrame([
    # Ruta 1 (Centro)
    {"ruta": 1, "nombre": "Terminal de Autobuses", "lat": 20.9989, "lon": -100.3872, "zona": "Centro"},
    {"ruta": 1, "nombre": "Parroquia de San José", "lat": 20.9985, "lon": -100.3870, "zona": "Centro"},
    {"ruta": 1, "nombre": "Mercado Municipal", "lat": 20.9992, "lon": -100.3885, "zona": "Centro"},

    # Ruta 2 (Sur)
    {"ruta": 2, "nombre": "La Cieneguita", "lat": 20.9950, "lon": -100.3850, "zona": "Sur"},
    {"ruta": 2, "nombre": "Parque La Alameda", "lat": 20.9938, "lon": -100.3862, "zona": "Sur"},
    {"ruta": 2, "nombre": "Escuela Secundaria Técnica", "lat": 20.9927, "lon": -100.3841, "zona": "Sur"},

    # Ruta 3 (Este)
    {"ruta": 3, "nombre": "Refugio de San José", "lat": 20.9900, "lon": -100.3800, "zona": "Este"},
    {"ruta": 3, "nombre": "Plaza del Refugio", "lat": 20.9911, "lon": -100.3788, "zona": "Este"},
    {"ruta": 3, "nombre": "Centro de Salud Oriente", "lat": 20.9915, "lon": -100.3820, "zona": "Este"},

    # Ruta 4 (Norte)
    {"ruta": 4, "nombre": "Unidad Deportiva Norte", "lat": 21.0045, "lon": -100.3905, "zona": "Norte"},
    {"ruta": 4, "nombre": "Preparatoria CBTA", "lat": 21.0030, "lon": -100.3917, "zona": "Norte"},
    {"ruta": 4, "nombre": "IMSS Norte", "lat": 21.0022, "lon": -100.3899, "zona": "Norte"},

    # Ruta 5 (Poniente)
    {"ruta": 5, "nombre": "Parque Industrial", "lat": 21.0072, "lon": -100.3958, "zona": "Poniente"},
    {"ruta": 5, "nombre": "Procter & Gamble", "lat": 21.0060, "lon": -100.3971, "zona": "Poniente"},
    {"ruta": 5, "nombre": "Libramiento Oeste", "lat": 21.0080, "lon": -100.3935, "zona": "Poniente"},
])

# Simular contexto
np.random.seed(42)
paradas["hora"] = np.random.choice([7, 12, 17], size=len(paradas))
paradas["dia"] = np.random.choice(["lunes", "miércoles", "sábado"], size=len(paradas))
paradas["clima"] = np.random.choice(["soleado", "lluvia"], size=len(paradas))

# ========== 2. Entrenar modelo de predicción de demanda ==========
dataset = []
for _, p in paradas.iterrows():
    base = 10
    base += 5 if p["hora"] in [7, 17] else 0
    base += 3 if p["dia"] in ["lunes", "miércoles"] else 0
    base += 4 if p["clima"] == "lluvia" else 0
    pasajeros = base + np.random.normal(0, 2)
    dataset.append({
        "hora": p["hora"],
        "dia": p["dia"],
        "clima": p["clima"],
        "zona": p["zona"],
        "pasajeros": int(pasajeros)
    })

df_modelo = pd.DataFrame(dataset)
X = pd.get_dummies(df_modelo.drop("pasajeros", axis=1))
y = df_modelo["pasajeros"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# ========== 3. Descargar grafo vial ==========
print("📡 Descargando grafo de San José Iturbide...")
G = ox.graph_from_place("San José Iturbide, Guanajuato, Mexico", network_type="drive")
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)

# Encontrar nodo más cercano
paradas["nodo"] = ox.distance.nearest_nodes(G, X=paradas["lon"], Y=paradas["lat"])

# ========== 4. Crear mapa ==========
colores = ["red", "blue", "green", "orange", "purple"]
mapa = folium.Map(location=[21.002, -100.387], zoom_start=14)

for ruta in range(1, 6):
    grupo = paradas[paradas["ruta"] == ruta].sort_values(by="hora")

    for _, p in grupo.iterrows():
        entrada = pd.get_dummies(pd.DataFrame([{
            "hora": p["hora"],
            "dia": p["dia"],
            "clima": p["clima"],
            "zona": p["zona"]
        }]))
        entrada = entrada.reindex(columns=X.columns, fill_value=0)
        pred = int(modelo.predict(entrada)[0])
        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=7,
            color=colores[ruta-1],
            fill=True,
            fill_opacity=0.8,
            popup=f"{p['nombre']}<br>Predicción: {pred} pasajeros<br>{p['dia']} {p['hora']}h, {p['clima']}"
        ).add_to(mapa)

    # Conectar puntos con A*
    nodos = grupo["nodo"].tolist()
    for i in range(len(nodos) - 1):
        try:
            camino = nx.astar_path(G, nodos[i], nodos[i+1], weight="length")
            puntos = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in camino]
            folium.PolyLine(puntos, color=colores[ruta-1], weight=3).add_to(mapa)
        except:
            print(f"No se pudo conectar entre nodos {nodos[i]} y {nodos[i+1]}")

# ========== 5. Guardar resultado ==========
mapa.save("DemoIaSji/sistema_transporte_san_jose.html")
print("✅ Mapa generado: sistema_transporte_san_jose.html")
