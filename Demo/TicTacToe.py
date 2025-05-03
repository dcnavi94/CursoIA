import numpy as np
import random
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout, QMessageBox

# --------------------------
# 1. ENTORNO DE TIC TAC TOE
# --------------------------
class TicTacToeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        # Tablero de 9 posiciones inicializado en ceros
        self.board = np.zeros(9, dtype=int)
        self.done = False
        return self.board.copy()

    def available_actions(self):
        # Retorna las posiciones vacías donde se puede jugar
        return [i for i in range(9) if self.board[i] == 0]

    def check_winner(self):
        # Revisa si alguien ganó
        b = self.board.reshape(3, 3)
        for i in range(3):
            if abs(sum(b[i, :])) == 3: return b[i, 0]
            if abs(sum(b[:, i])) == 3: return b[0, i]
        if abs(b[0,0] + b[1,1] + b[2,2]) == 3: return b[1,1]
        if abs(b[0,2] + b[1,1] + b[2,0]) == 3: return b[1,1]
        return 0  # 0 = nadie gana aún

    def step(self, action, player):
        # Ejecuta la jugada
        if self.board[action] != 0 or self.done:
            return self.board.copy(), -10, True  # Penalización por jugada ilegal
        self.board[action] = player
        winner = self.check_winner()
        if winner == player:
            self.done = True
            return self.board.copy(), 1, True
        elif 0 not in self.board:
            self.done = True
            return self.board.copy(), 0.5, True  # Empate
        return self.board.copy(), 0, False  # Juego continúa

# --------------------------
# 2. AGENTE Q-LEARNING
# --------------------------
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.2):
        self.q_table = {}  # Diccionario de estados:acciones
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state_key(self, state):
        return tuple(state)

    def choose_action(self, state, available):
        # Selecciona una acción con política epsilon-greedy
        key = self.get_state_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(9)
        if random.random() < self.epsilon:
            return random.choice(available)
        q_vals = self.q_table[key].copy()
        q_vals[[i for i in range(9) if i not in available]] = -np.inf
        return np.argmax(q_vals)

    def update(self, state, action, reward, next_state, done):
        # Actualiza Q(s,a)
        key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(9)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(9)
        q_old = self.q_table[key][action]
        q_next = 0 if done else np.max(self.q_table[next_key])
        self.q_table[key][action] = q_old + self.alpha * (reward + self.gamma * q_next - q_old)

# --------------------------
# 3. ENTRENAMIENTO AUTOMÁTICO
# --------------------------
env = TicTacToeEnv()
agent = QLearningAgent()

print("🤖 Entrenando IA con 1000 partidas contra sí misma...")

for episode in range(1000):  # número de partidas
    state = env.reset()
    done = False
    while not done:
        available = env.available_actions()
        action = agent.choose_action(state, available)
        next_state, reward, done = env.step(action, 1)
        if not done:
            opp_action = random.choice(env.available_actions())
            next_state, opp_reward, done = env.step(opp_action, -1)
            reward -= opp_reward
        agent.update(state, action, reward, next_state, done)
        state = next_state

print("✅ Entrenamiento completado. Interfaz iniciando...")

# --------------------------
# 4. INTERFAZ GRÁFICA (PyQt5)
# --------------------------
class TicTacToeGUI(QWidget):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.env = TicTacToeEnv()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Tic Tac Toe - IA Aprendiendo")
        self.grid = QGridLayout()
        self.buttons = []
        for i in range(9):
            btn = QPushButton("")
            btn.setFixedSize(100, 100)
            btn.clicked.connect(lambda _, idx=i: self.jugada(idx))
            self.grid.addWidget(btn, i // 3, i % 3)
            self.buttons.append(btn)
        self.setLayout(self.grid)
        self.show()

    def jugada(self, idx):
        if self.env.board[idx] != 0 or self.env.done:
            return
        self.env.step(idx, -1)  # Jugador humano
        self.actualizar_ui()
        if self.env.done:
            self.mostrar_resultado()
            return
        state = self.env.board.copy()
        ai_move = self.agent.choose_action(state, self.env.available_actions())
        self.env.step(ai_move, 1)
        self.actualizar_ui()
        if self.env.done:
            self.mostrar_resultado()

    def actualizar_ui(self):
        for i, val in enumerate(self.env.board):
            if val == 1:
                self.buttons[i].setText("O")  # IA
            elif val == -1:
                self.buttons[i].setText("X")  # Humano
            else:
                self.buttons[i].setText("")

    def mostrar_resultado(self):
        ganador = self.env.check_winner()
        if ganador == 1:
            QMessageBox.information(self, "Resultado", "¡La IA gana!")
        elif ganador == -1:
            QMessageBox.information(self, "Resultado", "¡Tú ganas!")
        else:
            QMessageBox.information(self, "Resultado", "Empate.")
        self.env.reset()
        self.actualizar_ui()

# --------------------------
# 5. EJECUCIÓN DE LA APP
# --------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = TicTacToeGUI(agent)
    sys.exit(app.exec_())
