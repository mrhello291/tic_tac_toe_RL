# tic_tac_toe_gui.py
import tkinter as tk
from tkinter import messagebox
import numpy as np
import os
from tic_tac_toe_rl import TicTacToeEnv, QLearningAgent

# Function to check the winner.
def check_winner(board):
    win_conditions = [
        (0,1,2), (3,4,5), (6,7,8),
        (0,3,6), (1,4,7), (2,5,8),
        (0,4,8), (2,4,6)
    ]
    for (i,j,k) in win_conditions:
        if board[i] != 0 and board[i] == board[j] == board[k]:
            return board[i]
    if 0 not in board:
        return 0  # draw
    return None

class TicTacToeGUI:
    def __init__(self, master):
        self.master = master
        master.title("Tic Tac Toe")
        self.mode = None  # "single" or "multi"
        self.current_player = 1  # 1 for X, -1 for O
        self.board = [0] * 9  # 0: empty, 1: X, -1: O
        
        # Load the RL agent for single-player mode.
        self.agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0)
        self.q_table_file = "q_table.pkl"
        if os.path.exists(self.q_table_file):
            self.agent.load_q_table(self.q_table_file)
        else:
            print("Q-table not found. AI may not work as expected.")

        self.create_menu()

    def create_menu(self):
        self.menu_frame = tk.Frame(self.master)
        self.menu_frame.pack(pady=20)

        tk.Label(self.menu_frame, text="Select Game Mode:", font=("Helvetica", 14)).pack(pady=10)
        tk.Button(self.menu_frame, text="Single Player (vs. AI)", font=("Helvetica", 12),
                  command=lambda: self.start_game("single")).pack(pady=5)
        tk.Button(self.menu_frame, text="Multiplayer", font=("Helvetica", 12),
                  command=lambda: self.start_game("multi")).pack(pady=5)

    def start_game(self, mode):
        self.mode = mode
        self.current_player = 1  # Reset to player 1 (X)
        self.board = [0] * 9
        # Destroy menu and create game board.
        self.menu_frame.destroy()
        self.create_board()
        self.status_label = tk.Label(self.master, text="Player X's turn", font=("Helvetica", 14))
        self.status_label.pack(pady=10)

    def create_board(self):
        self.buttons = []
        board_frame = tk.Frame(self.master)
        board_frame.pack()
        for i in range(9):
            btn = tk.Button(board_frame, text=" ", font=("Helvetica", 24), width=5, height=2,
                            command=lambda i=i: self.on_button_click(i))
            btn.grid(row=i//3, column=i%3)
            self.buttons.append(btn)
        self.reset_button = tk.Button(self.master, text="Reset Game", font=("Helvetica", 12),
                                      command=self.reset_game)
        self.reset_button.pack(pady=10)

    def on_button_click(self, index):
        if self.board[index] != 0:
            return  # already taken
        # In both modes, human move:
        if self.mode == "single" or self.mode == "multi":
            self.make_move(index, self.current_player)
            winner = check_winner(self.board)
            if winner is not None:
                self.end_game(winner)
                return
            # For multiplayer, simply alternate turns.
            if self.mode == "multi":
                self.current_player *= -1
                self.status_label.config(text=f"Player {'X' if self.current_player==1 else 'O'}'s turn")
            elif self.mode == "single":
                # In single player, after human (always player 1 / X) move, let AI move if game not over.
                self.master.after(300, self.ai_move)

    def make_move(self, index, player):
        self.board[index] = player
        self.buttons[index].config(text="X" if player == 1 else "O", state="disabled")

    def ai_move(self):
        valid_actions = [i for i, cell in enumerate(self.board) if cell == 0]
        if not valid_actions:
            return
        # Our RL agent is trained as player 1.
        # Since in single-player mode the human is player 1 (X) and AI is player -1 (O),
        # we transform the board for the agent by multiplying by -1.
        transformed_state = -np.array(self.board)
        action = self.agent.choose_action(transformed_state, valid_actions)
        self.make_move(action, -1)
        winner = check_winner(self.board)
        if winner is not None:
            self.end_game(winner)
        else:
            self.status_label.config(text="Player X's turn")

    def end_game(self, winner):
        if winner == 1:
            result = "Player X wins!"
        elif winner == -1:
            result = "Player O wins!" if self.mode == "multi" else "AI wins!"
        else:
            result = "It's a draw!"
        messagebox.showinfo("Game Over", result)
        self.status_label.config(text=result)
        # Disable all buttons
        for btn in self.buttons:
            btn.config(state="disabled")

    def reset_game(self):
        self.board = [0] * 9
        self.current_player = 1
        for btn in self.buttons:
            btn.config(text=" ", state="normal")
        if self.mode == "single":
            self.status_label.config(text="Player X's turn")
        else:
            self.status_label.config(text="Player X's turn")

if __name__ == "__main__":
    root = tk.Tk()
    game = TicTacToeGUI(root)
    root.mainloop()
