import logging
import sys
import copy
import random
import time

import numpy as np


from six.moves import input

from goboard import GoBoard
from scoring import evaluate_territory

from policyvalue import NaivePolicyValue

import tqdm


BOARD_SIZE = 5
C_PUCT = 1.0
N_SIMULATIONS = 4  # FIXME

"""

TODO/fix:
- Implement about minimum viable NN design that should be able to learn!
- Render self play from history
- Implement NN, and training NN
- make network (functions) into a class

"""

if "-d" in sys.argv:
    level_idx = sys.argv.index("-d") + 1
    level = int(sys.argv[level_idx]) if level_idx < len(sys.argv) else 10
    logging.basicConfig(level=level)
else:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
np.set_printoptions(3)



class GoState(GoBoard):
    """
    Wrapper of betago class.
    TODO: Possibly replace the state with numpy arrays for less memory consumption
    """

    def __init__(self, board_size=BOARD_SIZE):
        super(GoState, self).__init__(board_size)

        self.game_over = False
        self.winner = None
        self.current_player = 'b'  # TODO represent this with (1, -1) is faster
        self.action_space = board_size**2 + 1
        self.valid_actions = self._valid_actions()

        self.last_action = -1
        self.last_action_2 = -1

        self.player_transition = {'b': 'w', 'w': 'b'}

    def step(self, choice):

        action = self.valid_actions[choice]
        pos = self._action_pos(action)

        # random_ordering = iter(np.random.permutation(self.action_space))
        # TODO: "test if valid" uses deepcopy and took too long (especially when doing rollouts)
        # Find first legal move:
        # while pos and not self.is_move_legal(self.current_player, pos):
        #     try:
        #         action = next(random_ordering)
        #     except:
        #         raise Exception("No legal action.")
        #     pos = self._action_pos(action)

        # If illegal move: Will pass
        logger.log(5, "Did action {} in:\n{}".format(pos, self))

        if pos and not self.is_move_legal(self.current_player, pos):
            pos = None
            logger.log(5, "Which was not allowed")

        if pos:
            super(GoState, self).apply_move(self.current_player, pos)

        self.current_player = self.player_transition[self.current_player]

        self.last_action_2 = self.last_action
        self.last_action = pos

        self._new_state_checks()  # Updates self.game_over and self.winner

    def _action_pos(self, action):
        if action == self.action_space - 1:  # pass turn
            return None
        else:
            return (action // self.board_size, action % self.board_size)

    def _new_state_checks(self):
        """Checks if game is over and who won"""
        board_is_full = len(self.board) == self.board_size**2
        double_pass = (self.last_action is None) and \
                      (self.last_action_2 is None)
        self.game_over = board_is_full or double_pass

        if self.game_over:
            self.winner = self._compute_winner()

        self.valid_actions = self._valid_actions()

    def _compute_winner(self):
        counts = evaluate_territory(self)
        black_won = counts.num_black_stones + counts.num_black_territory > counts.num_white_stones + counts.num_white_territory
        white_won = counts.num_black_stones + counts.num_black_territory < counts.num_white_stones + counts.num_white_territory
        # Make sure tie -> 0
        return black_won - white_won

    def _valid_actions(self):
        actions = []
        for action in range(self.action_space):
            if self._action_pos(action) not in self.board:
                actions.append(action)

        return actions

    def observed_state(self):
        board = np.zeros([2, self.board_size, self.board_size])
        for key, val in self.board.items():
            if val == 'b':
                board[0, key] = 1.0
            if val == 'w':
                board[1, key] = 1.0

        return board



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

        self.w = np.zeros_like(state.valid_actions)
        self.n = np.zeros_like(state.valid_actions) + np.finfo(np.float).resolution
        self.prior_policy = 1.0

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

    return node.w/node.n + C_PUCT*node.prior_policy*np.sqrt(node.sum_n)/(1 + node.n)

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

def play_game(start_state=GoState(), policy_value=NaivePolicyValue(), opponent=None):
    """
    Plays a game against itself or specified opponent.

    The state should be prepared so that it is the agents turn, 
    and so that `self.winner == 1` when the agent won.
    """

    # TODO: This will set .move_number = 0, should maybe track whose turn it is instead:
    tree_root = TreeStructure(start_state)
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
            
            if new_state.game_over:
                value = new_state.winner  # Probably look at the depth to see who won here?
            else:
                policy, value = policy_value.predict(node.state)
                node.prior_policy = policy[node.state.valid_actions]

            backpropagate(node, value)

        # Store the state and distribution before we prune the tree:
        # TODO: Refactor this
        game_history.append([tree_root.state, tree_root.state.observed_state(), tree_root.n])

        choice = choice_to_play(tree_root, bool(opponent))
        tree_root = tree_root.children[choice]
        tree_root.parent = None

        if opponent:
            game_history.append([tree_root.state, tree_root.state.observed_state(), tree_root.n])
            choice = opponent(tree_root.state)
            if choice in tree_root.children:
                tree_root = tree_root.children[choice]
            else:
                new_state = step(tree_root.state, choice)
                tree_root = TreeStructure(new_state, tree_root)
            tree_root.parent = None


    return game_history, tree_root.state.winner


# UI code below:

def human_opponent(state):
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


def self_play_visualisation():
    history, winner = play_game()
    print("Watching game replay\nPress Return to advance board")
    for state, board, hoice in history:
        print(state)
        input("")

    if winner == 1:
        print("Black won")
    else:
        print("White won")

if __name__ == "__main__":
    if "-selfplay" in sys.argv:
        self_play_visualisation()
        sys.exit()

    print("")
    print("Welcome!")
    print("Format moves like: y x")
    print("")
    history, winner = play_game(opponent=human_opponent)
    if winner == 1:
        print("Black won")
    else:
        print("White won")
