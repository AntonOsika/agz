from gym import error
try:
    import pachi_py
except ImportError as e:
    # The dependency group [pachi] should match the name is setup.py.
    raise error.DependencyNotInstalled('{}. (HINT: you may need to install the Go dependencies via "pip install gym[pachi]".)'.format(e))

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from six import StringIO
import sys
import six

BLACK = pachi_py.BLACK
WHITE = pachi_py.WHITE
BOARD_SIZE = 5

# The coordinate representation of Pachi (and pachi_py) is defined on a board
# with extra rows and columns on the margin of the board, so positions on the board
# are not numbers in [0, board_size**2) as one would expect. For this Go env, we instead
# use an action representation that does fall in this more natural range.

def _pass_action(board_size):
    return board_size**2

def _resign_action(board_size):
    return board_size**2 + 1

def _coord_to_action(board, c):
    '''Converts Pachi coordinates to actions'''
    if c == pachi_py.PASS_COORD: return _pass_action(board.size)
    if c == pachi_py.RESIGN_COORD: return _resign_action(board.size)
    i, j = board.coord_to_ij(c)
    return i*board.size + j

def _action_to_coord(board, a):
    '''Converts actions to Pachi coordinates'''
    if a == _pass_action(board.size): return pachi_py.PASS_COORD
    if a == _resign_action(board.size): return pachi_py.RESIGN_COORD
    return board.ij_to_coord(a // board.size, a % board.size)

def str_to_action(board, s):
    return _coord_to_action(board, board.str_to_coord(s.encode()))


class GoState(object):
    '''
    Go game state. Consists of a current player and a board.
    Actions are exposed as integers in [0, num_actions), which is different
    from Pachi's internal "coord_t" encoding.
    '''
    def __init__(self, board_size=BOARD_SIZE, color=BLACK, board=None):
        '''
        Args:
            board_size: size of board
            color: color of current player
            board: current board
        '''
        assert color in [pachi_py.BLACK, pachi_py.WHITE], 'Invalid player color'

        if board:
            self.board = board
        else:
            self.board = pachi_py.CreateBoard(board_size)

        self.last_action = -1
        self.last_action_2 = -1

        self.color = color
        self.board_size = board_size

        self.game_over = False
        self.winner = None
        self.action_space = spaces.Discrete(board_size**2 + 1)

        self._new_state_checks()

        if color == BLACK:
            self.current_player = 1
        else:
            self.current_player = -1

    def act(self, action):
        '''
        Executes an action for the current player
        '''

        try:
            self.board = self.board.play(_action_to_coord(self.board, action), self.color)
        except pachi_py.IllegalMove:
            # Will do pass turn on disallowed move
            action = _pass_action(self.board_size)
            self.board = self.board.play(_action_to_coord(self.board, action), self.color)

        self.color = pachi_py.stone_other(self.color)
        self.current_player = -self.current_player

        self.last_action_2 = self.last_action
        self.last_action = action

        self._new_state_checks()  # Updates self.game_over and self.winner

    def stateless_act(self, action):
        '''
        Executes an action for the current player
        Returns:
            a new GoState with the new board and the player switched
        '''
        try:
            new_board = self.board.play(_action_to_coord(self.board, action), self.color)
        except pachi_py.IllegalMove:
            # Will do pass turn on invalid move
            action = _pass_action(self.board_size)
            new_board = self.board.play(_action_to_coord(self.board, action), self.color)

        new_state = GoState(
                board_size=self.board_size,
                color=pachi_py.stone_other(self.color),
                board=new_board
                )

        new_state.last_action_2 = new_state.last_action
        new_state.last_action = action

        new_state._new_state_checks()

        return new_state

    def step(self, choice):
        """Makes an action from choice of valid_actions"""
        action = self.valid_actions[choice]
        self.act(action)

    def observed_state(self):
        return self._observed_state

    def _new_state_checks(self):
        """Checks if game is over and who won"""
        double_pass = (self.last_action is _pass_action(self.board_size)) and \
                      (self.last_action_2 is _pass_action(self.board_size))

        self.game_over = self.board.is_terminal or double_pass

        if self.game_over:
            self.winner = self._compute_winner()

        encoded_board = self.board.encode()

        #TODO: change input to model to include empty pos
        self._observed_state = encoded_board[:2].transpose() 
        self.valid_actions = self._valid_actions(encoded_board[2])

    def _valid_actions(self, empty_positions):
        actions = []
        for action in range(self.board_size**2):
            # coord = board.ij_to_coord(action // board.size, action % board.size)
            if empty_positions[action // self.board.size, action % self.board.size] == 1:
                actions.append(action)

        return actions + [_pass_action(self.board_size)]

    def _compute_winner(self):
        """Returns winner as -1/0/1 for white/tie/black"""
        white_won = self.board.official_score > 0
        black_won = self.board.official_score < 0
        return black_won - white_won

    def __repr__(self):
        return 'To play: {}\n{}'.format(six.u(pachi_py.color_to_str(self.color)), self.board.__repr__().decode())



def act(state, action):
    """Functional version of act"""
    return state.stateless_act(action)

def step(state, choice):
    """Functional version of step"""
    return act(state, state.valid_actions[choice])


### Adversary policies ###
def make_random_policy(np_random):
    def random_policy(curr_state, prev_state, prev_action):
        b = curr_state.board
        legal_coords = b.get_legal_coords(curr_state.color)
        return _coord_to_action(b, np_random.choice(legal_coords))
    return random_policy

def make_pachi_policy(board, engine_type='uct', threads=1, pachi_timestr=''):
    engine = pachi_py.PyPachiEngine(board, engine_type, six.b('threads=%d' % threads))

    def pachi_policy(curr_state, prev_state, prev_action):
        if prev_state is not None:
            assert engine.curr_board == prev_state.board, 'Engine internal board is inconsistent with provided board. The Pachi engine must be called consistently as the game progresses.'
            prev_coord = _action_to_coord(prev_state.board, prev_action)
            engine.notify(prev_coord, prev_state.color)
            engine.curr_board.play_inplace(prev_coord, prev_state.color)
        out_coord = engine.genmove(curr_state.color, pachi_timestr)
        out_action = _coord_to_action(curr_state.board, out_coord)
        engine.curr_board.play_inplace(out_coord, curr_state.color)
        return out_action

    return pachi_policy
