from __future__ import print_function

import numpy as np
import random
import itertools

from six.moves import input

import agz
import policyvalue
from gostate import GoState

N_SIMULATIONS = 100

def training_loop(policy_value_class=policyvalue.SimpleCNN,
                  board_size=5,
                  n_simulations=N_SIMULATIONS,
                  games_per_iteration=10,
                  train_per_iteration=10,
                  eval_games=10,
                  batch_size=32,
                  visualise_freq=10):

    # obs_shape = GoState(board_size=board_size).observed_state().shape
    # memory = np.array([memory_size] + obs_shape)
    # memory_idx = 0
    # memory_used = 0

    input_shape = [board_size, board_size, 2]

    memory = []

    model = policy_value_class(input_shape)
    best_model = model

    print("Training... Abort with Ctrl-C.")
    for i in itertools.count():
        print("Iteration", i)
        try:
            for j in range(games_per_iteration):
                history, winner = agz.play_game(start_state=GoState(board_size),
                                                policy_value=best_model,
                                                n_simulations=n_simulations)

                for state, obs, pi in history:
                    memory.append([obs, pi, winner])

                if j % visualise_freq == 0:
                    print("Visualising one game:")
                    for state, board, choice in history:
                        print(state)
                        # input("")
                    if winner == 1:
                        print("Black won")
                    else:
                        print("White won")

            for j in range(train_per_iteration):
                samples = [random.choice(memory) for _ in range(batch_size)]
                obs, pi, z = [np.stack(x) for x in zip(*samples)]

                model.train_on_batch(obs, [pi, z])

            # TODO: Implement agent class and duels
            # for i in range(eval_games):
            #     history, winner = agz.play_game(policyvalue=model, opponent=)

            best_model = model

        except KeyboardInterrupt:
            print("Stopped training with Ctrl-C.")
            break

    best_model.model.save('model_{}x{}_{}.h5'.format(board_size,
                                                     board_size,
                                                     i))
    return best_model

def main(n_simulations=N_SIMULATIONS):
    board_size = 5
    input_shape = [board_size, board_size, 2]
    dumb_model = policyvalue.SimpleCNN(input_shape=input_shape)
    smart_model = training_loop(board_size=board_size)

    print("First playing against initial version:")
    agz.main(policy_value=dumb_model, n_simulations=n_simulations)
    print("Now playing against trained version:")
    agz.main(policy_value=smart_model, n_simulations=n_simulations)


if __name__ == "__main__":
    main()
