import numpy as np
import random
import itertools

from __future__ import print_function

import agz
import policyvalue
from gostate import GoState



def training_loop(policy_value_class=policyvalue.SimpleCNN,
                  board_size=9,
                  games_per_iteration=10,
                  train_per_iteration=10,
                  eval_games=10,
                  batch_size=32,
                  memory_size=1e7):

    # obs_shape = GoState(board_size=board_size).observed_state().shape
    #
    # memory = np.array([memory_size] + obs_shape)
    # memory_idx = 0
    # memory_used = 0

    memory = []

    model = policy_value_class(board_size)
    best_model = model

    print("Training... Abort with Ctrl-C.")
    for i in itertools.counter():
        print("Iteration", i)
        try:
            for i in range(games_per_iteration):
                history, winner = agz.play_game(policyvalue=best_model)

                for state, obs, pi in history:
                    memory.append([obs, pi, winner])

            for i in range(train_per_iteration):
                batch = random.choices(memory, k=batch_size)
                model.train_on_batch(batch)

            # TODO: Implement agents and duels
            # for i in range(eval_games):
            #     history, winner = agz.play_game(policyvalue=model, opponent=)

            best_model = model

        except KeyboardInterrupt:
            print("Stopped training with Ctrl-C.")
            break

    return best_model

def main():
    board_size = 9
    dumb_model = policyvalue.SimpleCNN(board_size)
    smart_model = training_loop()

    print("First playing against initial version:")
    agz.main(policy_value=dumb_model)
    print("Now playing against trained version:")
    agz.main(policy_value=smart_model)


if __name__ == "__main__":
    main()