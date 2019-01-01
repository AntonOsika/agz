import logging
import numpy as np

from goboard import GoBoard
from scoring import evaluate_territory

BOARD_SIZE = 5

logger = logging.getLogger("__main__")


class GoState(GoBoard):
    """
    OpenAI-gym env for go board.
    Has .valid_actions to sample from. If step receives an invalid actions -> pass turn is played.
    Can generate the numeric observation with .observed_state.

    properties:

    .winner
    .game_over
    .current_player
    .action_space
    .valid_actions


    TODO: Replace go engine code so that checking valid states does not require a deepcopy.

    """

    def __init__(self, board_size=BOARD_SIZE):
        super(GoState, self).__init__(board_size)

        self.game_over = False
        self.winner = None
        self.current_player = 'b'
        self.action_space = board_size**2 + 1
        self.valid_actions = self._valid_actions()

        self.last_action = -1
        self.last_action_2 = -1

        self.player_transition = {'b': 'w', 'w': 'b'}

    def step(self, choice):

        action = self.valid_actions[choice]
        pos = self._action_pos(action)

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
        board = np.zeros([self.board_size, self.board_size, 2])
        for key, val in self.board.items():
            if val == 'b':
                board[key, 0] = 1.0
            if val == 'w':
                board[key, 1] = 1.0

        return board

def step(state, choice):
    """Functional stateless version of env.step() """
    t0 = time.time()
    new_state = copy.deepcopy(state)
    logger.log(6, "took {} to deepcopy \n{}".format(time.time()-t0, state) )
    new_state.step(choice)
    return new_state


