import random
import copy
import logging
import time

import numpy as np

logger = logging.getLogger("__main__")

class NaivePolicyValue(object):
    def __init__(self):
        pass

    def value_network_counter(self, state):
        """Some logistic regression thing on sum of stones"""
        black_stones = 0
        white_stones = 0
        for x in state.board.values():
            if x == 'b':
                black_stones += 1
            if x == 'w':
                white_stones += 1
        value = np.tanh((black_stones - white_stones)/3.0)
        return value

    def value_network_rollout(self, state):
        """Returns value of position for player 1."""
        # simple rollout placeholder:
        t0 = time.time()
        state = copy.deepcopy(state)
        t1 = time.time()
        counter = 0
        while not state.game_over:
            # choice = sample(policy_network(state)[state.allowed_actions])
            choice = random.randint(0, len(state.valid_actions) - 1)
            state.step(choice)
            counter += 1
        logger.debug("took {} + {} to copy + roll out for {}:".format(
            t1 - t0, time.time() - t1, counter))
        return state.winner

    def policy(self, state):
        """Returns distribution over all allowed actions"""
        # uniform placeholder:
        return np.zeros([state.action_space]) + 1.0/state.action_space

    def value(self, state):
        return self.value_network_rollout(state)

    def predict(self, state):
        return self.policy(state), self.value(state)
