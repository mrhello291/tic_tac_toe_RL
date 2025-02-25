import numpy as np
import random

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
        self.q_table = {}  # state (tuple) -> list of Q-values for each action
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state_key(self, state):
        return tuple(state)

    def get_q_values(self, state):
        key = self.get_state_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(9)  # 9 possible actions
        return self.q_table[key]

    def choose_action(self, state, valid_actions):
        q_values = self.get_q_values(state)
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        # Choose the action with the highest Q-value among valid moves
        q_valid = {a: q_values[a] for a in valid_actions}
        return max(q_valid, key=q_valid.get)

    def update(self, state, action, reward, next_state, next_valid_actions):
        q_values = self.get_q_values(state)
        q_next = self.get_q_values(next_state)
        max_q_next = max([q_next[a] for a in next_valid_actions]) if next_valid_actions else 0
        q_values[action] = q_values[action] + self.alpha * (reward + self.gamma * max_q_next - q_values[action])

# Hyperparameters
num_episodes = 50000
alpha = 0.1
gamma = 0.9
epsilon = 1.0
min_epsilon = 0.1
epsilon_decay = 0.99995  # Adjust decay rate as needed

agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon=epsilon)
env = TicTacToeEnv()

for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        valid_actions = env.get_valid_actions()
        action = agent.choose_action(state, valid_actions)
        # Agent plays (player 1)
        next_state, reward, done = env.step(action, player=1)

        # If game isn't over, let the opponent play (random move)
        if not done:
            opp_valid = env.get_valid_actions()
            if opp_valid:  # Opponent makes a move
                opp_action = random.choice(opp_valid)
                next_state, opp_reward, done = env.step(opp_action, player=-1)
                # Invert the opponent's reward for the agent's perspective
                if done:
                    reward = -opp_reward

        next_valid_actions = env.get_valid_actions()
        agent.update(state, action, reward, next_state, next_valid_actions)
        state = next_state

    # Decay epsilon
    agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)

    if (episode + 1) % 5000 == 0:
        print(f"Episode {episode+1}/{num_episodes} - Epsilon: {agent.epsilon:.4f}")

# After training, you can save agent.q_table to a file for later use.
