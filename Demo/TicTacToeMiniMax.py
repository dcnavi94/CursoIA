import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout, QMessageBox, QLabel, QVBoxLayout, QComboBox
import numpy as np

class TicTacToeGame:
    def __init__(self):
        self.board = np.zeros(9, dtype=int)
        self.done = False

    def reset(self):
        self.board = np.zeros(9, dtype=int)
        self.done = False

    def available_moves(self):
        return [i for i in range(9) if self.board[i] == 0]

    def check_winner(self, board=None):
        b = board.reshape(3, 3) if board is not None else self.board.reshape(3, 3)
        for i in range(3):
            if abs(sum(b[i])) == 3:
                return b[i, 0]
            if abs(sum(b[:, i])) == 3:
                return b[0, i]
        if abs(b[0, 0] + b[1, 1] + b[2, 2]) == 3:
            return b[1, 1]
        if abs(b[0, 2] + b[1, 1] + b[2, 0]) == 3:
            return b[1, 1]
        return 0

    def is_draw(self):
        return 0 not in self.board and self.check_winner() == 0

    def make_move(self, index, player):
        if self.board[index] == 0:
            self.board[index] = player
            return True
        return False

    def minimax(self, board, is_maximizing, depth, max_depth):
        winner = self.check_winner(board)
        if winner != 0:
            return winner
        if 0 not in board or depth == max_depth:
            return 0

        if is_maximizing:
            best_score = -np.inf
            for i in range(9):
                if board[i] == 0:
                    board[i] = 1
                    score = self.minimax(board.copy(), False, depth + 1, max_depth)
                    board[i] = 0
                    best_score = max(score, best_score)
            return best_score
        else:
            best_score = np.inf
            for i in range(9):
                if board[i] == 0:
                    board[i] = -1
                    score = self.minimax(board.copy(), True, depth + 1, max_depth)
                    board[i] = 0
                    best_score = min(score, best_score)
            return best_score

    def best_move(self, difficulty):
        max_depth = {"Fácil": 1, "Media": 3, "Difícil": 9}[difficulty]
        best_score = -np.inf
        move = None
        for i in range(9):
            if self.board[i] == 0:
                self.board[i] = 1
                score = self.minimax(self.board.copy(), False, 0, max_depth)
                self.board[i] = 0
                if score > best_score:
                    best_score = score
                    move = i
        return move

class TicTacToeUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tic Tac Toe - IA Invencible")
        self.setFixedSize(320, 400)
        self.layout = QVBoxLayout()

        self.difficulty_box = QComboBox()
        self.difficulty_box.addItems(["Fácil", "Media", "Difícil"])
        self.layout.addWidget(QLabel("Dificultad de la IA:"))
        self.layout.addWidget(self.difficulty_box)

        self.stats_label = QLabel("Jugador: 0 | IA: 0 | Empates: 0")
        self.layout.addWidget(self.stats_label)

        self.grid = QGridLayout()
        self.layout.addLayout(self.grid)
        self.setLayout(self.layout)

        self.game = TicTacToeGame()
        self.buttons = []
        self.init_ui()
        self.stats = {'player': 0, 'ai': 0, 'draws': 0}

    def init_ui(self):
        for i in range(9):
            button = QPushButton("")
            button.setFixedSize(100, 100)
            button.setStyleSheet("font-size: 24px;")
            button.clicked.connect(lambda _, idx=i: self.player_move(idx))
            self.grid.addWidget(button, i // 3, i % 3)
            self.buttons.append(button)
        self.show()

    def player_move(self, idx):
        if self.game.board[idx] != 0 or self.game.done:
            return
        self.game.make_move(idx, -1)
        self.update_board()
        if self.check_end():
            return
        difficulty = self.difficulty_box.currentText()
        ai = self.game.best_move(difficulty)
        if ai is not None:
            self.game.make_move(ai, 1)
            self.update_board()
            self.check_end()

    def update_board(self):
        for i in range(9):
            if self.game.board[i] == -1:
                self.buttons[i].setText("X")
            elif self.game.board[i] == 1:
                self.buttons[i].setText("O")
            else:
                self.buttons[i].setText("")

    def check_end(self):
        winner = self.game.check_winner()
        if winner != 0 or self.game.is_draw():
            self.game.done = True
            if winner == 1:
                self.stats['ai'] += 1
                QMessageBox.information(self, "Resultado", "¡La IA gana!")
            elif winner == -1:
                self.stats['player'] += 1
                QMessageBox.information(self, "Resultado", "¡Tú ganas!")
            else:
                self.stats['draws'] += 1
                QMessageBox.information(self, "Resultado", "Empate.")
            self.update_stats()
            self.game.reset()
            self.update_board()
            return True
        return False

    def update_stats(self):
        self.stats_label.setText(f"Jugador: {self.stats['player']} | IA: {self.stats['ai']} | Empates: {self.stats['draws']}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = TicTacToeUI()
    sys.exit(app.exec_())