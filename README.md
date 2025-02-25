# Tic-tac-toe RL agent

## How to play?

Clone the repo.
Run the tic_tac_toe_gui.py

## What did I do to make this?

I simply made a RL agent in tic_tac_toe_rl.py file. It simply takes a storage file and loads records of previous behaviour in it.
Now I made a selfplay file. This is just two RL agents fighting against one another. The first agent stores his state probabilites in selfTable1.pkl. The second starts second always and stores its probabilities in selfTable2.pkl. Then I ran the selfplay.py file to build those pkl files that I can use in the gui file against real players. Initially the result was very bad because I was not exploiting symmetries, but as I found the problem, I fixed it. The selfplay_live.py is to see two RL agents fighting against one another live.
Now in the gui, when the the user goes first, the selfPlay2.pkl is loaded, otherwise selfPlay1.pkl is loaded, because they will have better training of winning in both cases. Also it was quite clear while simulation that the one who starts first has way way way higher chance of winning. (Like out of 500 epochs, 400-430 were of agent1 winning, 20-30 draw and rest were of agent2 winning)