# tic_tac_toe_rl.py
import numpy as np
import random
import pickle
import os

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
                return (1 if player == 1 else -1), True  # win for agent or loss
        if 0 not in self.board:
            return 0, True  # draw
        return 0, False  # game continues

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0):
        self.q_table = {}  # key: state (tuple), value: np.array of Q-values for 9 actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state_key(self, state):
        return tuple(state)

    def get_q_values(self, state):
        key = self.get_state_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(9)  # initialize Q-values for all 9 actions
        return self.q_table[key]

    def choose_action(self, state, valid_actions):
        q_values = self.get_q_values(state)
        # Epsilon-greedy policy: explore or choose best valid action.
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        q_valid = {a: q_values[a] for a in valid_actions}
        return max(q_valid, key=q_valid.get)

    def update(self, state, action, reward, next_state, next_valid_actions):
        q_values = self.get_q_values(state)
        q_next = self.get_q_values(next_state)
        max_q_next = max([q_next[a] for a in next_valid_actions]) if next_valid_actions else 0
        q_values[action] += self.alpha * (reward + self.gamma * max_q_next - q_values[action])
    
    def save_q_table(self, filename):
        """Save the Q-table to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to '{filename}'")
    
    def load_q_table(self, filename):
        """Load the Q-table from a file."""
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Q-table loaded from '{filename}'")

# If you run this file directly, it will train the agent and update the Q-table.
if __name__ == "__main__":
    # Hyperparameters
    num_episodes = 50000
    alpha = 0.1
    gamma = 0.9
    epsilon = 1.0
    min_epsilon = 0.1
    epsilon_decay = 0.99995  # Adjust decay rate as needed
    q_table_file = "q_table.pkl"

    # Initialize agent and environment
    agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)
    env = TicTacToeEnv()

    # Load existing Q-table if available
    if os.path.exists(q_table_file):
        agent.load_q_table(q_table_file)
    else:
        print("No Q-table file found. Starting from scratch.")

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.choose_action(state, valid_actions)
            next_state, reward, done = env.step(action, player=1)

            # Opponent makes a move (random policy)
            if not done:
                opp_valid = env.get_valid_actions()
                if opp_valid:
                    opp_action = random.choice(opp_valid)
                    next_state, opp_reward, done = env.step(opp_action, player=-1)
                    # If game ended after opponent's move, adjust reward.
                    if done:
                        reward = -opp_reward

            next_valid_actions = env.get_valid_actions()
            agent.update(state, action, reward, next_state, next_valid_actions)
            state = next_state

        # Decay epsilon after each episode
        agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)

        if (episode + 1) % 5000 == 0:
            print(f"Episode {episode+1}/{num_episodes} - Epsilon: {agent.epsilon:.4f}")

    # Save the updated Q-table after training
    agent.save_q_table(q_table_file)
