import logging
import sys
import random

import numpy as np

import time

if "-d" in sys.argv:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TreeStructure():
    """Node that tracks how good a specific choice was"""
    def __init__(self, state, parent=None, choice_that_led_here=None):

        self.children = {}  # map from choice to node

        self.parent = parent
        self.state = state

        self.w = np.zeros_like(state.valid_actions)
        self.n = np.zeros_like(state.valid_actions) + np.finfo(np.float).resolution
        self.policy_result = policy_network(state)[state.valid_actions]  # TODO: use a setter function

        self.sum_n = 0
        self.choice_that_led_here = choice_that_led_here

        self.move_number = 0

        if parent:
            self.move_number = parent.move_number + 1

def sample(probs):
    """Sample from unnormalized probabilities"""

    probs = probs / probs.sum()
    return np.random.choice(np.arange(len(probs)), p=probs.flatten())

def puct_distribution(node):
    """Puct equation"""
    # this should never be a distribution but always maximised over?
    logger.debug("Selecting node at move {}".format(node.move_number))
    logger.debug(node.w.astype('int'))
    logger.debug(node.n.astype('int'))
    return node.w/node.n + C_PUCT*node.policy_result*np.sqrt(node.sum_n)/(1 + node.n)

def puct_choice(node):
    """Selects the next move."""
    return np.argmax(puct_distribution(node))

def choice_to_play(node, opponent=None):
    """Samples a move if beginning of self play game."""
    logger.debug("Selecting move # {}".format(node.move_number))
    logger.debug(node.w.astype('int'))
    logger.debug(node.n.astype('int'))
    if node.move_number < 30 and opponent is None:
        return sample(node.n)
    else:
        return np.argmax(node.n)

def backpropagate(node, value):

    def _increment(node, choice, value):
        # Mirror value for odd states:
        value *= 1 - 2*(node.move_number % 2)
        node.w[choice] += value
        node.n[choice] += 1
        node.sum_n += 1

    while node.parent:
        _increment(node.parent, node.choice_that_led_here, value)
        node = node.parent

def play_game(state=GoState(), opponent=None):
    """
    Plays a game against itself or specified opponent.

    The state should be prepared so that it is the agents turn, 
    and so that `self.winner == 1` when the agent won.
    """

    # TODO: This will set .move_number = 0, should maybe track whose turn it is instead:
    tree_root = TreeStructure(state)  
    game_history = []

    while not tree_root.state.game_over:

        # for i in tqdm.tqdm(range(N_SIMULATIONS)):
        for i in range(N_SIMULATIONS):
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
            
            # Now happens in constructor, probably should do it outside (and parallellize):
            # node.policy_result = policy_network(state)
            
            if new_state.game_over:
                value = new_state.winner  # Probably look at the depth to see who won here?
            else:
                value = value_network(state)
            
            backpropagate(node, value)

        # Store the state and distribution before we prune the tree:
        game_history.append([tree_root.state, tree_root.n/tree_root.n.sum()])

        choice = choice_to_play(tree_root, bool(opponent))
        tree_root = tree_root.children[choice]
        tree_root.parent = None

        if opponent:
            game_history.append([tree_root.state, tree_root.n])
            choice = opponent(tree_root.state)
            if choice in tree_root.children:
                tree_root = tree_root.children[choice]
            else:
                new_state = step(tree_root.state, choice)
                tree_root = TreeStructure(new_state, tree_root)
            tree_root.parent = None


    return game_history, tree_root.state.winner
