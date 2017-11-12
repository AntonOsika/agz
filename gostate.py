
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



