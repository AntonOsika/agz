import numpy as np
import random

import agz
import policyvalue
from gostate import GoState



def training_loop(board_size=9,
                  games_per_iteration=10,
                  eval_games=10,
                  batch_size=32,
                  memory_size=1e7):

    # obs_shape = GoState(board_size=board_size).observed_state().shape
    #
    # memory = np.array([memory_size] + obs_shape)
    # memory_idx = 0
    # memory_used = 0
    memory = []

    model = policyvalue.SimpleCNN(board_size)
    best_model = last_model

    for i in range(games_per_iteration):
        history, winner = agz.play_game(policyvalue=best_model)

        for state, obs, pi in history:
            memory.append([obs, pi, winner])

    random.choices(memory, k=batch_size)

    policyvalue.train_on_batch()



def main():
    training_loop()


if __name__ == "__main__":
    main()