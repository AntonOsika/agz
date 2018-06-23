from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import copy
import random
import itertools

import numpy as np
from six.moves import input

import agz
import policyvalue

# from gostate_pachi import GoState
from gostate_pachi import GoState

N_SIMULATIONS = 50

def training_loop(policy_value_class=policyvalue.SimpleCNN,
                  board_size=5,
                  n_simulations=N_SIMULATIONS,
                  games_per_iteration=2,
                  train_per_iteration=1000,
                  eval_games=10,
                  batch_size=32,
                  visualise_freq=10):

    # obs_shape = GoState(board_size=board_size).observed_state().shape
    # memory = np.array([memory_size] + obs_shape)
    # memory_idx = 0
    # memory_used = 0

    max_game_length = 2*board_size**2

    input_shape = [board_size, board_size, 2]

    memory = []

    improvements = 0.0
    duels = 0.0

    model = policy_value_class(input_shape)
    best_model = model

    print("Training... Abort with Ctrl-C.")
    for i in itertools.count():
        print("Iteration", i)
        try:
            for j in range(games_per_iteration):
                history, winner = agz.play_game(start_state=GoState(board_size),
                                                policy_value=best_model,
                                                n_simulations=n_simulations,
                                                max_game_length=max_game_length)


                for state, obs, pi in history:
                    memory.append([obs, pi, winner])

                if j % visualise_freq == 0:
                    print("Visualising one game:")
                    for state, board, choice in history:
                        print(state)
                        # input("")
                    if winner == 1:
                        print("Black won")
                    elif winner == 0:
                        print("Tie")
                    else:
                        print("White won")

            for j in range(int(train_per_iteration/batch_size)):
                samples = [random.choice(memory) for _ in range(batch_size)]
                obs, pi, z = [np.stack(x) for x in zip(*samples)]

                model.train_on_batch(obs, [pi, z])

            score = 0
            for i in range(eval_games):
                start_state = GoState(board_size)
                old_agent = agz.MCTSAgent(best_model, GoState(board_size), n_simulations=n_simulations, self_play=True) # FIXME: adding noise through setting self_play
                new_agent = agz.MCTSAgent(model, GoState(board_size), n_simulations=n_simulations, self_play=True) # FIXME: adding noise through setting self_play

                if i % 2 == 0: # Play equal amounts of games as black/white
                    history, winner = agz.duel(start_state, new_agent, old_agent)
                    score += winner
                else:
                    history, winner = agz.duel(start_state, old_agent, new_agent)
                    score -= winner

                # Store history:
                for state, obs, pi in history:
                    memory.append([obs, pi, winner])

            print("New model won {} more games than old.".format(score))
            if score > eval_games*0.05:
                best_model = model
                improvements += 1
            duels += 1
            print("{:2f} % of games were improvements".format(100.0*improvements/duels))


        except KeyboardInterrupt:
            print("Stopped training with Ctrl-C.")
            break

    best_model.model.save('model_{}x{}_{}.h5'.format(board_size,
                                                     board_size,
                                                     i))
    return best_model

def main(n_simulations=N_SIMULATIONS):
    board_size = 9
    input_shape = [board_size, board_size, 2]
    dumb_model = policyvalue.SimpleCNN(input_shape=input_shape)
    smart_model = training_loop(board_size=board_size)

    print("First playing against initial version:")
    agz.main(policy_value=dumb_model, n_simulations=n_simulations)
    print("Now playing against trained version:")
    agz.main(policy_value=smart_model, n_simulations=n_simulations)


if __name__ == "__main__":
    main()
