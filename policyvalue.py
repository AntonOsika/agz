import random
import copy
import logging
import time

import numpy as np

from resnet import ResNet


logger = logging.getLogger("__main__")

"""
Model evaluating prior-policy and value for MCTS.
Classes here are Go specific so far.
"""

class SimpleCNN(ResNet):
    """
    Uses the keras resnet implementation.
    It reverses order of input and their shape!
    """
    def __init__(self, input_shape):
        super(SimpleCNN, self).__init__(input_shape=input_shape,
                n_filter=256,
                n_blocks=20)

        self.compile()

    def predict(self, state):
        x = state.observed_state()
        x = x[None, ...]  # batch_size = 1 here (using queue in paper)
        p, v = self.model.predict(x)
        return p.flatten(), v.flatten()[0]

    def train_on_batch(self, x, y):
        self.model.train_on_batch(x, y)

    def load(self, number):
        fn = "model_{}x{}_{}.h5".format(self.input_shape[0], self.input_shape[1], number)
        try:
            self.model.load_weights(fn)
        except:
            print("Couldnt load model weights {}".format(fn))


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
