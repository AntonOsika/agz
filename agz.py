import logging
import sys
import copy
import random
import time
import os
import itertools

import numpy as np

from six.moves import input

"""
from gostate import GoState
"""
from gostate_pachi import GoState
from gostate_pachi import step

from policyvalue import NaivePolicyValue
from policyvalue import SimpleCNN

# import tqdm

#The following is used when GPU memory is full FIXME
os.environ['CUDA_VISIBLE_DEVICES'] = ''

BOARD_SIZE = 5
C_PUCT = 1.0
N_SIMULATIONS = 160

"""
MCTS logic and go playing / visualisation.

TODO/fix:
- Decide on CLI arguments and use argparse

"""

# '-d level' argument for printing specific level:
if "-d" in sys.argv:
    level_idx = sys.argv.index("-d") + 1
    level = int(sys.argv[level_idx]) if level_idx < len(sys.argv) else 10
    logging.basicConfig(level=level)
else:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
np.set_printoptions(3)


if 'step' not in globals():
    def step(state, choice):
        """Functional stateless version of env.step() """
        t0 = time.time()
        new_state = copy.deepcopy(state)
        logger.log(6, "took {} to deepcopy \n{}".format(time.time()-t0, state) )
        new_state.step(choice)
        return new_state


class TreeStructure(object):
    def __init__(self, state, parent=None, choice_that_led_here=None):

        self.children = {}  # map from choice to node

        self.parent = parent
        self.state = state

        self.w = np.zeros(len(state.valid_actions))
        self.n = np.zeros(len(state.valid_actions))
        self.n += (1.0 + np.random.rand(len(self.n)))*1e-10
        self.prior_policy = 1.0/len(self.state.valid_actions)

        self.sum_n = 1
        self.choice_that_led_here = choice_that_led_here

        self.move_number = 0

        if parent:
            self.move_number = parent.move_number + 1


    def history_sample(self):
        """Returns a representation of state to be stored for training"""
        pi = np.zeros(self.state.action_space.n)
        pi[self.state.valid_actions] = self.n/self.n.sum()
        return [self.state, self.state.observed_state(), pi]

    def add_noise_to_prior(self, noise_frac=0.25, dirichlet_alpha=0.03):
        noise = np.random.dirichlet(dirichlet_alpha*np.ones(len(self.state.valid_actions)))
        self.prior_policy = (1-noise_frac)*self.prior_policy + noise_frac*noise

def sample(probs):
    """Sample from unnormalized probabilities"""

    probs = probs / probs.sum()
    return np.random.choice(np.arange(len(probs)), p=probs.flatten())

def puct_distribution(node):

    """Puct equation"""
    # this should never be a distribution but always maximised over?
    # Took some time:
    # logger.debug("Selecting node at move {}".format(node.move_number))
    # logger.debug(node.w)
    # logger.debug(node.n)
    # logger.debug(node.prior_policy)

    return node.w/node.n + C_PUCT*node.prior_policy*np.sqrt(node.sum_n)/(1 + node.n)

def puct_choice(node):
    """Selects the next move."""
    return np.argmax(puct_distribution(node))


def choice_to_play(node, opponent=None):
    """Samples a move if beginning of self play game."""
    logger.debug("Selecting move # {}".format(node.move_number))
    logger.debug(node.w)
    logger.debug(node.n)
    logger.debug(node.prior_policy)

    if node.move_number < 30 and opponent is None:
        return sample(node.n)
    else:
        return np.argmax(node.n)

def backpropagate(node, value):
    """MCTS backpropagation"""

    def _increment(node, choice, value):
        # Mirror value for odd states:
        value *= 1 - 2*(node.move_number % 2)  # TODO: use node.state.current_player after changing it to (+1, -1)
        node.w[choice] += value
        node.n[choice] += 1
        node.sum_n += 1

    while node.parent:
        _increment(node.parent, node.choice_that_led_here, value)
        node = node.parent


def mcts(tree_root, policy_value, n_simulations):
    # for i in tqdm.tqdm(range(n_simulations)):
    for i in range(n_simulations):
        node = tree_root
        # Select from "PUCT/UCB1 equation" in paper.
        choice = puct_choice(node)
        while choice in node.children.keys():
            node = node.children[choice]
            choice = puct_choice(node)

        if node.state.game_over:
            # This only happens the second time we go to a winning state.
            # Logic for visiting "winning nodes" multiple times is probably correct?
            value = node.state.winner
            backpropagate(node, value)
            continue

        # Expand tree:
        new_state = step(node.state, choice)
        node.children[choice] = TreeStructure(new_state, node, choice)
        node = node.children[choice]

        if new_state.game_over:
            value = new_state.winner  # Probably look at the depth to see who won here?
        else:
            policy, value = policy_value.predict(node.state)
            node.prior_policy = policy[node.state.valid_actions]

        backpropagate(node, value)


def print_tree(tree_root, level):
    print(" "*level, tree_root.choice_that_led_here, tree_root.state.board, tree_root.n, tree_root.w)
        # [print_tree(tree_root.children[i], level + 1) for i in tree_root.children]

class MCTSAgent(object):
    """Object that keeps track of MCTS tree and can perform actions"""

    def __init__(self, policy_value, state, n_simulations=N_SIMULATIONS):
        self.policy_value = policy_value
        self.game_history = list()
        self.tree_root = TreeStructure(state)
        self.n_simulations = n_simulations

        policy, value = self.policy_value.predict(self.tree_root.state)
        self.tree_root.prior_policy = policy[self.tree_root.state.valid_actions]
        assert type(self.tree_root.prior_policy) != float, "Prior_policy was not np array"
        self.tree_root.add_noise_to_prior()

    def update_state(self, choice):
        self.game_history.append(self.tree_root.history_sample())

        if choice in self.tree_root.children:
            self.tree_root = self.tree_root.children[choice]
        else:
            new_state = step(self.tree_root.state, choice)
            self.tree_root = TreeStructure(new_state)
            policy, value = self.policy_value.predict(self.tree_root.state)
            self.tree_root.prior_policy = policy[self.tree_root.state.valid_actions]
        self.tree_root.add_noise_to_prior()
        self.tree_root.parent = None

    def perform_simulations(self, n_simulations=None):
        n_simulations = n_simulations or self.n_simulations
        mcts(self.tree_root, self.policy_value, n_simulations)

    def decision(self, self_play=False):
        return choice_to_play(self.tree_root, not self_play)


def duel(state, agent_1, agent_2, max_game_length=1e99):
    """Plays two agants against each other"""
    history = []

    agents = itertools.cycle([agent_1, agent_2])

    move_number = 0
    while not state.game_over and move_number < max_game_length:
        actor = next(agents)
        actor.perform_simulations()
        choice = actor.decision()

        history.append(actor.tree_root.history_sample())

        state.step(choice)
        agent_1.update_state(choice)
        agent_2.update_state(choice)

        move_number += 1

    if move_number >= max_game_length:
        state.winner = state._compute_winner()

    return history, state.winner

class MCTSAgent(object):
    """Object that keeps track of MCTS tree and can perform actions"""

    def __init__(self, policy_value, state, n_simulations=N_SIMULATIONS):
        self.policy_value = policy_value
        self.game_history = list()
        self.tree_root = TreeStructure(state)
        self.n_simulations = n_simulations

        policy, value = self.policy_value.predict(self.tree_root.state)
        tree_root.prior_policy = policy[tree_root.state.valid_actions]

    def update_state(self, choice):
        self.game_history.append(tree_root.history_sample())

        if choice in self.tree_root.children:
            self.tree_root = self.tree_root.children[choice]
        else:
            new_state = step(self.tree_root.state, choice)
            self.tree_root = TreeStructure(new_state)
            policy, value = self.policy_value.predict(self.tree_root.state)
            tree_root.prior_policy = policy[tree_root.state.valid_actions]
        tree_root.parent = None

    def perform_simulations(self, n_simulations=None):
        n_simulations = n_simulations or self.n_simulations
        mcts(self.tree_root, self.policy_value, n_simulations)

    def decision(self, self_play=False):
        return choice_to_play(self.tree_root, not self_play)

# TODO: Create agent class from this that can be queried
def play_game(start_state=GoState(),
              policy_value=NaivePolicyValue(),
              max_game_length=1e99,
              opponent=None,
              n_simulations=N_SIMULATIONS):
    """
    Plays a game against itself or specified opponent.

    The state should be prepared so that it is the agents turn,
    and so that `self.winner == 1` when the agent won.
    """

    # TODO: This will set .move_number = 0, should maybe track whose turn it is instead:
    tree_root = TreeStructure(start_state)
    policy, value = policy_value.predict(tree_root.state)
    tree_root.prior_policy = policy[tree_root.state.valid_actions]
    tree_root.add_noise_to_prior()
    game_history = []

    while not tree_root.state.game_over and tree_root.move_number < max_game_length:

        mcts(tree_root, policy_value, n_simulations)

        # print_tree(tree_root,0)
        # Store the state and distribution before we prune the tree:
        # TODO: Refactor this

        game_history.append(tree_root.history_sample())

        choice = choice_to_play(tree_root, bool(opponent))
        tree_root = tree_root.children[choice]
        tree_root.parent = None
        tree_root.add_noise_to_prior()

        if opponent:
            game_history.append(tree_root.history_sample())
            choice = opponent(tree_root.state)
            if choice in tree_root.children:
                tree_root = tree_root.children[choice]
            else:
                new_state = step(tree_root.state, choice)
                tree_root = TreeStructure(new_state)
                policy, value = policy_value.predict(tree_root.state)
                tree_root.prior_policy = policy[tree_root.state.valid_actions]
            tree_root.parent = None

    if tree_root.move_number >= max_game_length:
        tree_root.state.winner = tree_root.state._compute_winner()

    return game_history, tree_root.state.winner


# UI code below:
def human_opponent(state):
    """Queries human for move when called."""
    print(state)
    while True:
        inp = input("What is your move? \n")
        if inp == 'pass':
            return len(state.valid_actions) - 1
        if inp == 'random':
            return random.randint(0, len(state.valid_actions) - 1)

        try:
            pos = [int(x) for x in inp.split()]
            action = pos[0]*state.board_size + pos[1]
            choice = state.valid_actions.index(action)
            return choice
        except:
            print("Invalid move {} try again.".format(inp))


def self_play_visualisation(board_size=BOARD_SIZE):
    """Visualises one game of self_play"""
    policy_value = SimpleCNN([board_size, board_size, 2])
    history, winner = play_game(policy_value=policy_value)
    print("Watching game replay\nPress Return to advance board")
    for state, board, hoice in history:
        print(state)
        input("")

    if winner == 1:
        print("Black won")
    else:
        print("White won")

def duel_players(player_1, player_2):
    return winner(player_1, player_2)

def main(policy_value=NaivePolicyValue(), board_size=BOARD_SIZE, n_simulations=N_SIMULATIONS):

    if "-selfplay" in sys.argv:
        self_play_visualisation()
        return

    if "-40" in sys.argv:
        n_simulations = 40
        print("Letting MCTS search for {} moves!".format(n_simulations))

    # Loads weights that trained for 60 iterations
    policy_value = SimpleCNN([board_size, board_size, 2])
    policy_value.load(6)

    print("")
    print("Welcome!")
    print("Format moves like: y x")
    print("(or pass/random)")
    print("")
    try:
        history, winner = play_game(start_state=GoState(board_size),
                                    policy_value=policy_value,
                                    opponent=human_opponent,
                                    n_simulations=n_simulations)
    except KeyboardInterrupt:
        print("Game aborted.")
        return

    if winner == 1:
        print("AI won")
    else:
        print("Human won")

if __name__ == "__main__":
