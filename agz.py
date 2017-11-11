import numpy as np

ACTION_SPACE = [9*9 + 1]
C_PUCT = 1.0

"""
TODO/fix:
- implement naive policy function (uniform?)
- implement play_action
- implement value function with e.g. rollout -> who wins.

"""

def step(state, action):
    """Game env placeholder"""
    return state

def policy_network(state):
    """Returns distribution over all allowed actions"""
    # uniform placeholder:
    return np.zeros_like(state.action_space) + 1.0/len(state.action_space)

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
        self.policy_result = policy_network(state)

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
        game_history.append([tree_root.state, tree_root.n])

        action = action_to_play(tree_root, opponent)
        tree_root = tree_root.children[action]

        if opponent:
            game_history.append([tree_root.state, tree_root.n])
            action = opponent(state)
            tree_root = tree_root.children.get(action) or TreeStructure()

    return game_history, tree_root.state.winner

    


