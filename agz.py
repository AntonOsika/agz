import numpy as np

import betago
from betago.dataloader.goboard import GoBoard

BOARD_SIZE = 9
ACTION_SPACE = [9*9 + 1]
C_PUCT = 1.0

"""
Needs a state class that has lots of logic

TODO/fix:
- Implement State object that has action_space, etc
- Implement step(state, action)
- Evaluate pure MCTS on a simple problem
- Render play with opponent
- Render self play from history
- Implement NN, and training NN
- make networks into a class
- Compute winner from board

"""


class GoState(GoBoard):

    def __init__(self, board_size=BOARD_SIZE):
        super(GoBoard, self).__init__(board_size)

        self.game_over = False
        self.winner = None
        self.action_space = list(range(board_size**2 + 1))
        self.current_player = 1  # in (1, -1)

        self.last_action = None
        self.last_action_2 = None

    def step(self, action):
        random_ordering = iter(np.random.permutation(len(self.action_space)))

        # Find first legal move:
        while not self.is_move_legal(self.current_player, action):
            try:
                action_index = next(random_ordering)
            except:
                raise Exception("No legal action.")
            action = self.action_space[action_index]

        super(GoBoard, self).apply_move(action, self.current_player)

        self.current_player = -self.current_player
        self.check_state()
        self.game_over = self._game_over()
        self.winner =

        self.last_action_2 = self.last_action
        self.last_action = action

    def _new_state_checks(self):
        full_board = len(self.board) == self.board_size**2
        double_pass = (self.last_action == len(self.action_space) - 1) and \
                      (self.last_last_2 == self.last_action)
        self.game_over = full_board or double_pass
        if self.game_over:
            self.winner = self._compute_winner()

    def _winner(self):
        return 1


def step(state, action):
    """

    Args:
        state: GoState
    """

    new_state = deepcopy.copy(state)
    new_state.step()
    return state

def policy_network(state):
    """Returns distribution over all allowed actions"""
    # uniform placeholder:
    return np.zeros_like(state.action_space) + 1.0/len(state.action_space)

    all_action_probs = model.predict(state)
    return all_action_probs[state.action_space]  # -> gets the probabilities that are allowed by indices

def value_network(state):
    """Returns value of position for player 1."""
    # simple rollout placeholder:
    while not state.game_over:
        action = sample(policy_network(state))
        state = play_action(state, action)
    return state.winner

def play_action(state, action_space_index):
    """Maps "action index" to action and plays it."""
    action = state.action_space[action_space_index]
    next_state = step(state, action)
    return next_state

class TreeStructure():
    def __init__(self, state, parent=None, action_that_led_here=None):
        self.children = {}  # map from action to node

        self.parent = parent
        self.state = state

        self.q = np.zeros_like(state.action_space)
        self.n = np.zeros_like(state.action_space) + np.finfo(np.float).resolution
        self.policy_result = policy_network(state)  # TODO: use a setter function

        self.n_passed = 0
        state.action_that_led_here = action_that_led_here

        self.move_number = 0
        if parent:
            self.move_number = parent.move_number + 1

def sample(action_probs):
    """Sample from unnormalized probabilities"""

    action_probs = action_probs / action_probs.sum()
    return np.random.choice(np.arange(len(action_probs)), p=action_probs.flatten())

def puct_distribution(node):
    """Puct equation"""
    # this shouldnt be a distribution but always maximised over?
    return node.q/node.n + C_PUCT*node.policy_result*np.sqrt(node.n_passed)/(1 + node.n)

def puct_action(node):
    """Selects the next move."""
    return np.argmax(puct_distribution(node))

def action_to_play(node, opponent=None):
    """Samples a move if beginning of self play game."""
    if node.move_number < 30 and opponent is None:
        return sample(node.n)
    else:
        return np.argmax(node.n)

def backpropagate(node, value):

    def _increment(node, action, value):
        # Mirror value for odd states (?):
        # value *= 2*(node.move_number % 2 ) - 1
        node.q[action] += value
        node.n[action] += 1
        node.n_passed += 1

    while node.parent:
        _increment(node.parent, node.action_that_led_here, value)
        node = node.parent


def play_game(state=START_STATE, opponent=None):
    """
    Plays a game against itself or specified opponent.

    The state should be prepared so that it is the agents turn, 
    and so that `self.winner == 1` when the agent won.
    """

    tree_root = TreeStructure(state)
    game_history = []

    while not tree_root.state.game_over:

        for i in xrange(1600):
            node = tree_root
            # Select action from "PUCT/UCB1 equation" in paper.
            action = puct_action(node)
            while action in node.children.keys():
                node = node.children[action]
                action = puct_action(node)

            if node.state.game_over:
                # This only happens the second time we go to a winning state.
                # Logic for visiting "winning nodes" multiple times is probably correct? FIXME
                value = node.state.winner
                backpropagate(node, value)
                continue
            
            # Expand tree:
            new_state = play_action(node.state, action)
            node.children[action] = TreeStructure(new_state, node, action)
            node = node.children[action]
            
            # Now happens in constructor (might want to parallellize):
            # node.policy_result = policy_network(state)
            
            if new_state.game_over:
                value = new_state.winner  # Probably look at the depth to see who won here?
            else:
                value = value_network(state)
            
            backpropagate(node, value)

        # Store the state and distribution before we prune the tree:
        game_history.append([tree_root.state, tree_root.n/tree_root.n.sum()])

        action = action_to_play(tree_root, opponent)
        tree_root = tree_root.children[action]

        if opponent:
            game_history.append([tree_root.state, tree_root.n])
            action = opponent(state)
            tree_root = tree_root.children.get(action) or TreeStructure()

    return game_history, tree_root.state.winner

    


