# ——————————————————————————————————————————————————————————————————
# Ejemplo de Q-learning en MountainCar-v0 (OpenAI Gym)
# Con parche para evitar el AttributeError de np.bool8 en Gym
# ——————————————————————————————————————————————————————————————————

# 0. PARCHE DE NumPy ANTES DE IMPORTAR GYM/JOBLIB
import numpy as np
# Si NumPy 2.x no define bool8, aliasamos a bool_
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# 1. Importar librerías de RL y visualización
import gym                      # Biblioteca para entornos de RL
import matplotlib.pyplot as plt # Para graficar resultados

# 2. Crear el entorno
env = gym.make("MountainCar-v0")

# 3. Discretizar el espacio de observación (posición, velocidad)
state_bins = [40, 40]  # Resolución de la discretización
obs_low  = env.observation_space.low
obs_high = env.observation_space.high
obs_bins = [
    np.linspace(obs_low[i], obs_high[i], state_bins[i])
    for i in range(len(obs_low))
]

def discretize(obs):
    """
    Convierte una observación continua en índices discretos.
    np.digitize devuelve el índice del bin al que pertenece obs[i].
    """
    return tuple(int(np.digitize(obs[i], obs_bins[i])) for i in range(len(obs)))

# 4. Inicializar la Q-table con ceros
n_actions = env.action_space.n           # Número de acciones posibles (3)
q_table   = np.zeros(state_bins + [n_actions])

# 5. Definir hiperparámetros
alpha         = 0.1    # Tasa de aprendizaje
gamma         = 0.99   # Descuento de recompensas futuras
epsilon       = 1.0    # Probabilidad inicial de explorar
epsilon_decay = 0.997  # Decaimiento por episodio
min_epsilon   = 0.01   # Límite inferior de epsilon
episodes      = 5000   # Total de episodios

# Lista para registrar recompensas
rewards = []

# 6. Bucle de entrenamiento
for episode in range(episodes):
    obs, _      = env.reset()       # Reiniciar entorno
    state       = discretize(obs)   # Estado discreto inicial
    done        = False
    total_reward = 0

    while not done:
        # a) ε-greedy: explora o explota
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # b) Ejecutar acción
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = discretize(next_obs)

        # c) Actualizar Q-table (Bellman)
        best_next = np.max(q_table[next_state])
        td_target = reward + gamma * best_next
        td_error  = td_target - q_table[state][action]
        q_table[state][action] += alpha * td_error

        state = next_state
        total_reward += reward

    # d) Decaer epsilon
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay

    rewards.append(total_reward)

    # e) Reporte cada 500 episodios
    if (episode + 1) % 500 == 0:
        print(f"📊 Episodio {episode+1}, ε={epsilon:.3f}, recompensa={total_reward}")

# 7. Cerrar entorno
env.close()

# 8. Graficar resultados
plt.figure(figsize=(10, 5))
plt.plot(rewards, label="Recompensa por episodio", alpha=0.7)
plt.xlabel("Episodio")
plt.ylabel("Recompensa acumulada")
plt.title("📈 Progreso de Q-learning en MountainCar-v0")
plt.grid(True)
plt.legend()
plt.show()
