# tic_tac_toe_selfplay.py
import numpy as np
import random
from tic_tac_toe_rl import TicTacToeEnv, TDLearningAgent
import time

def check_winner(board):
    """Return 1 if X wins, -1 if O wins, 0 for draw, or None if the game is not over."""
    win_conditions = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
        (0, 4, 8), (2, 4, 6)              # diagonals
    ]
    for (i, j, k) in win_conditions:
        if board[i] != 0 and board[i] == board[j] == board[k]:
            return board[i]  # returns 1 if X wins, -1 if O wins
    if 0 not in board:
        return 0  # draw
    return None

def agent2_choose_action(agent, board, valid_actions):
    """
    For agent2 (playing as O), transform the board (multiply by -1)
    so that from its perspective, its moves are like playing as 1.
    """
    transformed_board = [-x for x in board]
    return agent.choose_action(transformed_board, valid_actions)

def agent2_update(agent, state, next_state, reward=None):
    """
    Update agent2's value table using transformed states.
    The agent always learns as if playing as 1.
    """
    transformed_state = [-x for x in state]
    transformed_next_state = [-x for x in next_state]
    agent.update(transformed_state, transformed_next_state, reward=reward)

def self_play_game(agent1, agent2):
    """
    Plays one game between agent1 (as X) and agent2 (as O).  
    Both agents update their value estimates after every move.
    Returns the winner: 1 (agent1 wins), -1 (agent2 wins), or 0 (draw).
    """
    env = TicTacToeEnv()
    board = env.reset().tolist()  # working with list for convenience

    # Record the "previous" state for each agent (used for TD updates)
    last_state_agent1 = board.copy()
    last_state_agent2 = board.copy()

    # X (agent1) starts
    current_player = 1

    while True:
        valid_actions = [i for i, cell in enumerate(board) if cell == 0]
        if not valid_actions:
            break  # No moves left

        if current_player == 1:
            # Agent1 (X) plays normally
            action = agent1.choose_action(board, valid_actions)
            board[action] = 1
            winner = check_winner(board)
            if winner is not None:
                # Terminal update for agent1: win=1, loss/draw=0
                terminal_reward = 1 if winner == 1 else 0
                agent1.update(last_state_agent1, board, reward=terminal_reward)
                # Also update agent2 using terminal outcome from its perspective
                terminal_reward_agent2 = 1 if winner == -1 else 0
                agent2_update(agent2, last_state_agent2, board, reward=terminal_reward_agent2)
                return winner
            else:
                agent1.update(last_state_agent1, board)
                last_state_agent1 = board.copy()
                current_player = -1  # Switch turn
        else:
            # Agent2 (O) plays using transformed states
            action = agent2_choose_action(agent2, board, valid_actions)
            board[action] = -1
            winner = check_winner(board)
            if winner is not None:
                terminal_reward = 1 if winner == -1 else 0
                agent2_update(agent2, last_state_agent2, board, reward=terminal_reward)
                # Also update agent1 with terminal outcome from its perspective
                terminal_reward_agent1 = 1 if winner == 1 else 0
                agent1.update(last_state_agent1, board, reward=terminal_reward_agent1)
                return winner
            else:
                agent2_update(agent2, last_state_agent2, board)
                last_state_agent2 = board.copy()
                current_player = 1  # Switch turn

if __name__ == "__main__":
    # Create two independent agents.
    # (Insert load_value_table calls here if desired.)
    agent1 = TDLearningAgent(alpha=0.1, epsilon=0.1, debug=True, tableFile="selfTable1.pkl")  # For X
    agent2 = TDLearningAgent(alpha=0.1, epsilon=0.1, debug=True, tableFile="selfTable2.pkl")  # For O


    game_count = 0
    wins_agent1 = 0
    wins_agent2 = 0
    draws = 0

    try:
        while True:
            winner = self_play_game(agent1, agent2)
            game_count += 1

            if winner == 1:
                wins_agent1 += 1
                print(f"Game {game_count}: Agent1 (X) wins")
            elif winner == -1:
                wins_agent2 += 1
                print(f"Game {game_count}: Agent2 (O) wins")
            else:
                draws += 1
                print(f"Game {game_count}: Draw")

            print(f"Score => Agent1: {wins_agent1}, Agent2: {wins_agent2}, Draws: {draws}\n")
            agent1.save_value_table()
            agent2.save_value_table()
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Self-play interrupted by user. Exiting.")

        
        
