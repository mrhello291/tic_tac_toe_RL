import os
import time
import random
from tic_tac_toe_rl import TicTacToeEnv, TDLearningAgent

def print_board(board):
    """
    Prints the current Tic Tac Toe board in a human-readable format.
    Board is a list of 9 integers:
      1  -> X
     -1  -> O
      0  -> empty
    """
    symbols = []
    for cell in board:
        if cell == 1:
            symbols.append("X")
        elif cell == -1:
            symbols.append("O")
        else:
            symbols.append(" ")
    print(f" {symbols[0]} | {symbols[1]} | {symbols[2]} ")
    print("---+---+---")
    print(f" {symbols[3]} | {symbols[4]} | {symbols[5]} ")
    print("---+---+---")
    print(f" {symbols[6]} | {symbols[7]} | {symbols[8]} ")

def check_winner(board):
    """Returns 1 if X wins, -1 if O wins, 0 for draw, or None if game continues."""
    win_conditions = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
        (0, 4, 8), (2, 4, 6)              # diagonals
    ]
    for (i, j, k) in win_conditions:
        if board[i] != 0 and board[i] == board[j] == board[k]:
            return board[i]
    if 0 not in board:
        return 0  # draw
    return None

def agent2_choose_action(agent, board, valid_actions):
    """
    For agent2 (playing as O), we transform the board by multiplying by -1 so that,
    from its perspective, it's playing as if it were always X.
    """
    transformed_board = [-x for x in board]
    return agent.choose_action(transformed_board, valid_actions)

def agent2_update(agent, state, next_state, reward=None):
    """
    Update agent2's value table using transformed states.
    This makes the agent learn as if it were playing as X.
    """
    transformed_state = [-x for x in state]
    transformed_next_state = [-x for x in next_state]
    agent.update(transformed_state, transformed_next_state, reward=reward)

def self_play_game(agent1, agent2, sleep_time=1.0):
    """
    Plays one game between agent1 (X) and agent2 (O), printing the board after each move.
    Returns the winner: 1 (agent1 wins), -1 (agent2 wins), or 0 (draw).
    """
    env = TicTacToeEnv()
    board = env.reset().tolist()  # Using a list for easy printing
    last_state_agent1 = board.copy()
    last_state_agent2 = board.copy()
    current_player = 1

    os.system("clear")
    print("Starting a new game:")
    print_board(board)
    time.sleep(sleep_time)
    
    while True:
        valid_actions = [i for i, cell in enumerate(board) if cell == 0]
        if not valid_actions:
            break
        
        if current_player == 1:
            # Agent1 (X) move
            action = agent1.choose_action(board, valid_actions)
            board[action] = 1
            os.system("clear")
            print("Agent1 (X) moved:")
            print_board(board)
            time.sleep(sleep_time)
            
            winner = check_winner(board)
            if winner is not None:
                terminal_reward = 1 if winner == 1 else 0
                agent1.update(last_state_agent1, board, reward=terminal_reward)
                # Update agent2 with the opposite reward from its perspective.
                terminal_reward_agent2 = 1 if winner == -1 else 0
                agent2_update(agent2, last_state_agent2, board, reward=terminal_reward_agent2)
                return winner
            else:
                agent1.update(last_state_agent1, board)
                last_state_agent1 = board.copy()
                current_player = -1
        else:
            # Agent2 (O) move, using board transformation
            action = agent2_choose_action(agent2, board, valid_actions)
            board[action] = -1
            os.system("clear")
            print("Agent2 (O) moved:")
            print_board(board)
            time.sleep(sleep_time)
            
            winner = check_winner(board)
            if winner is not None:
                terminal_reward = 1 if winner == -1 else 0
                agent2_update(agent2, last_state_agent2, board, reward=terminal_reward)
                terminal_reward_agent1 = 1 if winner == 1 else 0
                agent1.update(last_state_agent1, board, reward=terminal_reward_agent1)
                return winner
            else:
                agent2_update(agent2, last_state_agent2, board)
                last_state_agent2 = board.copy()
                current_player = 1
    return 0

if __name__ == "__main__":
    # Create two independent agents.
    # (Insert load_value_table calls here if desired.)
    agent1 = TDLearningAgent(alpha=0.1, epsilon=0.1, debug=True, tableFile="selfTable1.pkl")  # For X
    agent2 = TDLearningAgent(alpha=0.1, epsilon=0.1, debug=True, tableFile="selfTable2.pkl")  # For O

    wins_agent1 = 0
    wins_agent2 = 0
    draws = 0
    game_count = 0

    try:
        while True:
            winner = self_play_game(agent1, agent2, sleep_time=1.0)
            game_count += 1

            if winner == 1:
                wins_agent1 += 1
                print(f"Game {game_count}: Agent1 (X) wins!")
            elif winner == -1:
                wins_agent2 += 1
                print(f"Game {game_count}: Agent2 (O) wins!")
            else:
                draws += 1
                print(f"Game {game_count}: Draw!")
            
            print(f"Score => Agent1: {wins_agent1}, Agent2: {wins_agent2}, Draws: {draws}")
            agent1.save_value_table()
            agent2.save_value_table()
            print("Waiting 2 seconds before starting next game...\n")
            time.sleep(2)
    except KeyboardInterrupt:
        print("Self-play interrupted by user. Exiting.")
