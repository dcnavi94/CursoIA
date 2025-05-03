# Requiere: pip install numpy tkinter

import numpy as np
import random
import threading
import tkinter as tk
from tkinter import ttk  # Corrección para usar ttk correctamente

# Parámetros del entorno
num_agents = 3  # Número de robots
map_size = 400  # Tamaño del mapa en discretización
q_tables = [np.zeros((map_size, map_size, 4)) for _ in range(num_agents)]  # Tablas Q por robot

# Parámetros individuales por agente
agent_params = [
    {'alpha': 0.3, 'gamma': 0.8, 'epsilon': 0.6},
    {'alpha': 0.5, 'gamma': 0.7, 'epsilon': 0.4},
    {'alpha': 0.2, 'gamma': 0.9, 'epsilon': 0.3},
]

# Contadores y estados iniciales
delivered_count = [0 for _ in range(num_agents)]
carrying = [None for _ in range(num_agents)]
object_positions = []
carried_objects = [None for _ in range(num_agents)]
start_positions = [[-8, -1], [-8, 0], [-8, 1]]

# Zona de entrega
drop_zone = [8, 0]

# Simular objetos aleatorios en el entorno
num_objects = 3
for _ in range(num_objects):
    x, y = np.random.uniform(-5, 5), np.random.uniform(-5, 5)
    object_positions.append([x, y])

def get_state(pos):
    x = int((pos[0] + 10) * 20)
    y = int((pos[1] + 10) * 20)
    return max(0, min(399, x)), max(0, min(399, y))

def move(pos, action):
    step = 1.0
    if action == 0: pos = [pos[0]+step, pos[1]]
    elif action == 1: pos = [pos[0]-step, pos[1]]
    elif action == 2: pos = [pos[0], pos[1]+step]
    elif action == 3: pos = [pos[0], pos[1]-step]
    return pos

# Interfaz
window = tk.Tk()
window.title("Simulación de IA (solo lógica)")
labels = [tk.Label(window, text=f"Robot {i} - Entregas: 0", font=("Arial", 14)) for i in range(num_agents)]
for lbl in labels: lbl.pack()
winner_label = tk.Label(window, text="", font=("Arial", 16, "bold"))
winner_label.pack(pady=10)

progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(window, variable=progress_var, maximum=100, length=300)
progress_bar.pack(pady=10)

def actualizar_monitor():
    for i in range(num_agents):
        labels[i].config(text=f"Robot {i} - Entregas: {delivered_count[i]}")
    window.after(300, actualizar_monitor)
actualizar_monitor()

def ejecutar_simulacion():
    positions = [[start_positions[i][0], start_positions[i][1]] for i in range(num_agents)]
    total_steps = 30 * 300
    step_counter = 0

    for episode in range(30):
        for i in range(num_agents):
            positions[i] = [start_positions[i][0], start_positions[i][1]]
            carrying[i] = None
            carried_objects[i] = None

        for step in range(300):
            for i in range(num_agents):
                alpha = agent_params[i]['alpha']
                gamma = agent_params[i]['gamma']
                epsilon = agent_params[i]['epsilon']

                x, y = get_state(positions[i])
                action = random.randint(0, 3) if random.uniform(0, 1) < epsilon else np.argmax(q_tables[i][x][y])
                new_pos = move(positions[i], action)
                x_new, y_new = get_state(new_pos)

                reward = -0.01
                if carrying[i] is None:
                    for j, obj_pos in enumerate(object_positions):
                        if obj_pos is not None and np.linalg.norm(np.array(new_pos) - np.array(obj_pos)) < 1.0:
                            carrying[i] = j
                            reward = 1
                            carried_objects[i] = obj_pos
                            break
                else:
                    if carried_objects[i] is not None:
                        if np.linalg.norm(np.array(new_pos) - np.array(drop_zone)) < 1.0:
                            reward = 2
                            object_positions[carrying[i]] = None
                            carrying[i] = None
                            carried_objects[i] = None
                            delivered_count[i] += 1

                            if delivered_count[i] >= 5:
                                winner_label.config(text=f"🎉 Robot {i} ganó con 5 entregas!")
                                progress_var.set(100)
                                window.after(2000, window.destroy)  # Cierra la ventana después de 2 segundos
                                return

                q_old = q_tables[i][x][y][action]
                q_tables[i][x][y][action] += alpha * (reward + gamma * np.max(q_tables[i][x_new][y_new]) - q_old)
                positions[i] = new_pos

                step_counter += 1
                progress_var.set(100 * step_counter / total_steps)

threading.Thread(target=ejecutar_simulacion).start()
window.mainloop()
