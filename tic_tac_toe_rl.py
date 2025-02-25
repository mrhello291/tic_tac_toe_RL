# tic_tac_toe_rl.py
import numpy as np
import random
import pickle
import os

def canonical_state(state):
    """
    Given a tic-tac-toe board (list or 1D np.array of length 9), return its canonical form.
    The canonical form is defined as the lexicographically smallest tuple among all
    rotations and reflections.
    """
    board = np.array(state).reshape((3, 3))
    states = []
    for i in range(4):
        rotated = np.rot90(board, i)
        states.append(tuple(rotated.flatten()))
        # Also include the horizontal reflection of the rotated state.
        reflected = np.fliplr(rotated)
        states.append(tuple(reflected.flatten()))
    return min(states)

class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros(9, dtype=int)  # 0: empty, 1: agent, -1: opponent

    def reset(self):
        self.board = np.zeros(9, dtype=int)
        return self.board.copy()

    def get_valid_actions(self):
        return [i for i, cell in enumerate(self.board) if cell == 0]

    def step(self, action, player):
        if self.board[action] != 0:
            raise ValueError("Invalid move!")
        self.board[action] = player
        reward, done = self.check_game_over(player)
        return self.board.copy(), reward, done

    def check_game_over(self, player):
        # Define win conditions (rows, columns, diagonals)
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]              # diagonals
        ]
        for condition in win_conditions:
            if all(self.board[i] == player for i in condition):
                return (1 if player == 1 else -1), True  # win for agent (1) or loss (-1)
        if 0 not in self.board:
            return 0, True  # draw
        return 0, False  # game continues

class TDLearningAgent:
    def __init__(self, alpha=0.1, epsilon=0.1, debug=False, tableFile=None):
        self.tableFile = tableFile
        self.value_table = {}  # key: canonical state (tuple), value: probability of winning
        if tableFile and os.path.exists(tableFile):
            self.load_value_table(tableFile)
        self.alpha = alpha
        self.epsilon = epsilon
        self.debug = debug  # If True, print updates

    def get_state_key(self, state):
        # Use the canonical form of the state to reduce redundancy due to symmetries.
        return canonical_state(state)

    def get_value(self, state):
        key = self.get_state_key(state)
        if key not in self.value_table:
            self.value_table[key] = 0.5  # Initialize unknown states with 0.5
        return self.value_table[key]

    def choose_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)  # Exploration

        # Exploitation: choose the move that leads to the state with highest estimated value.
        best_action = max(valid_actions, key=lambda a: self.get_value(self.get_next_state(state, a)))
        return best_action

    def get_next_state(self, state, action):
        next_state = state.copy()
        next_state[action] = 1  # Assume our player always plays as 1
        return next_state

    def update(self, state, next_state, reward=None):
        v_s = self.get_value(state)
        # If a terminal reward is provided, use it; otherwise, use the estimated value.
        v_s_prime = reward if reward is not None else self.get_value(next_state)
        delta = self.alpha * (v_s_prime - v_s)
        new_value = v_s + delta
        self.value_table[self.get_state_key(state)] = new_value
        if self.debug:
            print(f"Updated state {self.get_state_key(state)}: {v_s:.3f} -> {new_value:.3f}")

    def save_value_table(self):
        if self.tableFile:
            with open(self.tableFile, 'wb') as f:
                pickle.dump(self.value_table, f)
            print(f"Value table saved to '{self.tableFile}'")
        else:
            print("No table file provided; not saving.")

    def load_value_table(self, filename):
        with open(filename, 'rb') as f:
            self.value_table = pickle.load(f)
        print(f"Value table loaded from '{filename}'")


if __name__ == "__main__":
    num_episodes = 50000
    alpha = 0.1
    epsilon = 0.1
    value_table_file = "value_table.pkl"
    
    agent = TDLearningAgent(alpha=alpha, epsilon=epsilon, debug=True, tableFile=value_table_file)
    env = TicTacToeEnv()
        
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.choose_action(state, valid_actions)
            next_state, reward, done = env.step(action, player=1)
            
            if not done:
                opp_valid = env.get_valid_actions()
                if opp_valid:
                    opp_action = random.choice(opp_valid)
                    next_state, _, done = env.step(opp_action, player=-1)
            
            agent.update(state, next_state)
            state = next_state
        
        if (episode + 1) % 5000 == 0:
            print(f"Episode {episode+1}/{num_episodes}")
    
    agent.save_value_table()

# ---------------------------------------------------------------------------
# Many tic-tac-toe positions appear different but are really the same due to
# symmetries (rotations and reflections). By mapping each board state to its
# canonical form (e.g., the lexicographically smallest representation among
# all symmetric variants), we reduce the number of distinct states in the
# value table. This means that learning in one state will benefit all states
# that are symmetric to it, potentially speeding up learning and reducing the
# storage requirements.
#
# However, if the opponent does not take advantage of symmetries (i.e., plays
# differently in symmetric positions), then merging symmetric states might
# cause the agent to learn suboptimally in those cases. In that scenario, 
# symmetrically equivalent positions might not necessarily have the same value,
# as the opponent's non-symmetric play could result in different outcomes.
# ---------------------------------------------------------------------------
