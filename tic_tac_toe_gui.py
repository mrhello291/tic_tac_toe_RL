# import tkinter as tk
# from tkinter import messagebox
# import os
# from tic_tac_toe_rl import TicTacToeEnv, TDLearningAgent

# class TicTacToeGUI:
#     def __init__(self, master):
#         self.master = master
#         master.title("Tic Tac Toe TD-Learning")
        
#         # Toggle option: Who starts first? (1 for Agent, -1 for Human)
#         self.starting_player_frame = tk.Frame(master)
#         self.starting_player_frame.pack(pady=10)
#         self.first_player_var = tk.IntVar()
#         self.first_player_var.set(1)  # Default is 1 (Agent starts first)
#         tk.Label(self.starting_player_frame, text="Who starts first?").pack(side="left")
#         tk.Radiobutton(self.starting_player_frame, text="Agent (X)", variable=self.first_player_var, value=1).pack(side="left")
#         tk.Radiobutton(self.starting_player_frame, text="Human (O)", variable=self.first_player_var, value=-1).pack(side="left")
        
#         # Initialize environment and agent.
#         self.env = TicTacToeEnv()
#         self.agent = TDLearningAgent(alpha=0.1, epsilon=0.1, debug=True, tableFile="selfTable2.pkl")
#         self.value_table_file = self.agent.tableFile
#         if not os.path.exists(self.value_table_file):
#             print("No saved value table found. Starting fresh.")
        
#         self.reset_game()
    
#     def reset_game(self):
#         # Reset the board state using the environment.
#         self.board = self.env.reset().tolist()  # a list of 9 ints
#         self.last_state = self.board.copy()       # keep track of previous state for TD update
        
#         # Use the toggle option to determine who starts first.
#         self.current_player = self.first_player_var.get()
        
#         # Create the GUI board.
#         if hasattr(self, 'frame'):
#             self.frame.destroy()
#         self.frame = tk.Frame(self.master)
#         self.frame.pack()
        
#         self.buttons = []
#         for i in range(9):
#             btn = tk.Button(self.frame, text=" ", font=("Helvetica", 24), width=5, height=2,
#                             command=lambda i=i: self.human_move(i))
#             btn.grid(row=i//3, column=i%3)
#             self.buttons.append(btn)
        
#         # Status label and Reset button.
#         if hasattr(self, 'status_label'):
#             self.status_label.destroy()
#         self.status_label = tk.Label(self.master, text="", font=("Helvetica", 14))
#         self.status_label.pack(pady=10)
        
#         if hasattr(self, 'reset_button'):
#             self.reset_button.destroy()
#         self.reset_button = tk.Button(self.master, text="Reset Game", font=("Helvetica", 12), command=self.reset_game)
#         self.reset_button.pack(pady=10)
        
#         # Depending on the toggle, either start with the agent or wait for the human.
#         if self.current_player == 1:
#             self.status_label.config(text="Agent's turn (X)")
#             self.master.after(500, self.agent_move)
#         else:
#             self.status_label.config(text="Your turn (O)")
    
#     def update_button_texts(self):
#         for i, cell in enumerate(self.board):
#             if cell == 1:
#                 self.buttons[i].config(text="X", state="disabled")
#             elif cell == -1:
#                 self.buttons[i].config(text="O", state="disabled")
#             else:
#                 self.buttons[i].config(text=" ", state="normal")
    
#     def check_winner(self):
#         win_conditions = [
#             (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
#             (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
#             (0, 4, 8), (2, 4, 6)              # diagonals
#         ]
#         for (i, j, k) in win_conditions:
#             if self.board[i] != 0 and self.board[i] == self.board[j] == self.board[k]:
#                 return self.board[i]
#         if 0 not in self.board:
#             return 0  # draw
#         return None
    
#     def agent_move(self):
#         if self.current_player != 1:
#             return  # Not the agent's turn.
        
#         valid_actions = [i for i, cell in enumerate(self.board) if cell == 0]
#         if not valid_actions:
#             return
        
#         # Agent selects a move based on its current value estimates.
#         action = self.agent.choose_action(self.board, valid_actions)
#         self.make_move(action, 1)  # Agent (X) plays.
        
#         # Check if the agent's move ended the game.
#         winner = self.check_winner()
#         if winner is not None:
#             terminal_reward = 1 if winner == 1 else 0
#             self.agent.update(self.last_state, self.board, reward=terminal_reward)
#             self.end_game(winner)
#         else:
#             self.agent.update(self.last_state, self.board)
#             self.last_state = self.board.copy()
#             self.current_player = -1
#             self.status_label.config(text="Your turn (O)")
    
#     def human_move(self, index):
#         if self.current_player != -1:
#             return  # Not human's turn.
#         if self.board[index] != 0:
#             return
        
#         self.make_move(index, -1)  # Human (O) plays.
        
#         # Check if the human's move ended the game.
#         winner = self.check_winner()
#         if winner is not None:
#             terminal_reward = 1 if winner == 1 else 0
#             self.agent.update(self.last_state, self.board, reward=terminal_reward)
#             self.end_game(winner)
#         else:
#             self.agent.update(self.last_state, self.board)
#             self.last_state = self.board.copy()
#             self.current_player = 1
#             self.status_label.config(text="Agent's turn (X)")
#             self.master.after(500, self.agent_move)
    
#     def make_move(self, index, player):
#         self.board[index] = player
#         self.update_button_texts()
    
#     def end_game(self, winner):
#         if winner == 1:
#             result = "Agent wins!"
#         elif winner == -1:
#             result = "You win!"
#         else:
#             result = "It's a draw!"
#         self.status_label.config(text=result)
#         for btn in self.buttons:
#             btn.config(state="disabled")
#         messagebox.showinfo("Game Over", result)
        
#         # Save the updated value table to disk.
#         self.agent.save_value_table()

# if __name__ == "__main__":
#     root = tk.Tk()
#     root.title("Tic Tac Toe TD-Learning GUI")
#     gui = TicTacToeGUI(root)
#     root.mainloop()


import tkinter as tk
from tkinter import messagebox
import os
from tic_tac_toe_rl import TicTacToeEnv, TDLearningAgent

class TicTacToeGUI:
    def __init__(self, master):
        self.master = master
        master.title("Tic Tac Toe TD-Learning")
        
        # Toggle option: Who starts first? (1 for Agent, -1 for Human)
        self.starting_player_frame = tk.Frame(master)
        self.starting_player_frame.pack(pady=10)
        self.first_player_var = tk.IntVar()
        self.first_player_var.set(1)  # Default is 1 (Agent starts first)
        tk.Label(self.starting_player_frame, text="Who starts first?").pack(side="left")
        tk.Radiobutton(self.starting_player_frame, text="Agent (X)", variable=self.first_player_var, value=1).pack(side="left")
        tk.Radiobutton(self.starting_player_frame, text="Human (O)", variable=self.first_player_var, value=-1).pack(side="left")
        
        # Initialize environment.
        self.env = TicTacToeEnv()
        
        # Call reset_game to create agent (and load the correct value table) and setup board.
        self.reset_game()
    
    def reset_game(self):
        # Decide which value table to load based on who is set to start first.
        # If Agent starts first, use "selfTable1.pkl", otherwise use "selfTable2.pkl".
        table_file = "selfTable1.pkl" if self.first_player_var.get() == 1 else "selfTable2.pkl"
        self.agent = TDLearningAgent(alpha=0.1, epsilon=0.1, debug=True, tableFile=table_file)
        self.value_table_file = self.agent.tableFile
        if not os.path.exists(self.value_table_file):
            print("No saved value table found. Starting fresh.")
        
        # Reset the board state using the environment.
        self.board = self.env.reset().tolist()  # a list of 9 ints
        self.last_state = self.board.copy()       # keep track of previous state for TD update
        
        # Set current player based on toggle.
        self.current_player = self.first_player_var.get()
        
        # Create the GUI board.
        if hasattr(self, 'frame'):
            self.frame.destroy()
        self.frame = tk.Frame(self.master)
        self.frame.pack()
        
        self.buttons = []
        for i in range(9):
            btn = tk.Button(self.frame, text=" ", font=("Helvetica", 24), width=5, height=2,
                            command=lambda i=i: self.human_move(i))
            btn.grid(row=i//3, column=i%3)
            self.buttons.append(btn)
        
        # Status label and Reset button.
        if hasattr(self, 'status_label'):
            self.status_label.destroy()
        self.status_label = tk.Label(self.master, text="", font=("Helvetica", 14))
        self.status_label.pack(pady=10)
        
        if hasattr(self, 'reset_button'):
            self.reset_button.destroy()
        self.reset_button = tk.Button(self.master, text="Reset Game", font=("Helvetica", 12), command=self.reset_game)
        self.reset_button.pack(pady=10)
        
        # Depending on who starts, either let the agent move or wait for the human.
        if self.current_player == 1:
            self.status_label.config(text="Agent's turn (X)")
            self.master.after(500, self.agent_move)
        else:
            self.status_label.config(text="Your turn (O)")
    
    def update_button_texts(self):
        for i, cell in enumerate(self.board):
            if cell == 1:
                self.buttons[i].config(text="X", state="disabled")
            elif cell == -1:
                self.buttons[i].config(text="O", state="disabled")
            else:
                self.buttons[i].config(text=" ", state="normal")
    
    def check_winner(self):
        win_conditions = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
            (0, 4, 8), (2, 4, 6)              # diagonals
        ]
        for (i, j, k) in win_conditions:
            if self.board[i] != 0 and self.board[i] == self.board[j] == self.board[k]:
                return self.board[i]
        if 0 not in self.board:
            return 0  # draw
        return None
    
    def agent_move(self):
        if self.current_player != 1:
            return  # Not the agent's turn.
        
        valid_actions = [i for i, cell in enumerate(self.board) if cell == 0]
        if not valid_actions:
            return
        
        # Agent selects a move based on its current value estimates.
        action = self.agent.choose_action(self.board, valid_actions)
        self.make_move(action, 1)  # Agent (X) plays.
        
        # Check if the agent's move ended the game.
        winner = self.check_winner()
        if winner is not None:
            terminal_reward = 1 if winner == 1 else 0
            self.agent.update(self.last_state, self.board, reward=terminal_reward)
            self.end_game(winner)
        else:
            self.agent.update(self.last_state, self.board)
            self.last_state = self.board.copy()
            self.current_player = -1
            self.status_label.config(text="Your turn (O)")
    
    def human_move(self, index):
        if self.current_player != -1:
            return  # Not human's turn.
        if self.board[index] != 0:
            return
        
        self.make_move(index, -1)  # Human (O) plays.
        
        # Check if the human's move ended the game.
        winner = self.check_winner()
        if winner is not None:
            terminal_reward = 1 if winner == 1 else 0
            self.agent.update(self.last_state, self.board, reward=terminal_reward)
            self.end_game(winner)
        else:
            self.agent.update(self.last_state, self.board)
            self.last_state = self.board.copy()
            self.current_player = 1
            self.status_label.config(text="Agent's turn (X)")
            self.master.after(500, self.agent_move)
    
    def make_move(self, index, player):
        self.board[index] = player
        self.update_button_texts()
    
    def end_game(self, winner):
        if winner == 1:
            result = "Agent wins!"
        elif winner == -1:
            result = "You win!"
        else:
            result = "It's a draw!"
        self.status_label.config(text=result)
        for btn in self.buttons:
            btn.config(state="disabled")
        messagebox.showinfo("Game Over", result)
        
        # Save the updated value table to disk.
        self.agent.save_value_table()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Tic Tac Toe TD-Learning GUI")
    gui = TicTacToeGUI(root)
    root.mainloop()

