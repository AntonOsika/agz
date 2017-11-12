from agz import *

class AddDivState():
    """
    Wrapper of betago class.
    TODO: Possibly replace the state with numpy arrays for less memory consumption
    """

    def __init__(self, target = 7.4):
        print("init")
        self.game_over = False
        self.winner = None
        self.current_player = 'b'  # TODO represent this with (1, -1) is faster
        self.action_space = 2
        self.valid_actions = self._valid_actions()

        self.state = 3.14
        self.target = 5.31
        self.player_transition = {'b': 'b', 'b': 'b'}

    def step(self, choice):


        # If illegal move: Will pass
        logger.log(5, "Did action {} in:\n{}".format(choice, self))

        if choice > 0.5:
            self.state = self.state*0.75
        else:
            self.state = self.state+1.0

        print("choice", choice, "state", self.state)
        """
        self.current_player = self.player_transition[self.current_player]

        self.last_action_2 = self.last_action
        self.last_action = pos

        """
        self._new_state_checks()  # Updates self.game_over and self.winner

    def _action_pos(self, action):
        if action == self.action_space - 1:  # pass turn
            return None
        else:
            return (action // self.board_size, action % self.board_size)

    def _new_state_checks(self):

        self.game_over = self.state > self.target
        if self.game_over:
            self.winner = self._compute_winner()


    def _compute_winner(self):

        return 1/(10*abs(self.state - self.target) + 1)


    def _valid_actions(self):
        actions = []
        for action in range(self.action_space):
            actions.append(action)

        return actions

    def observed_state(self):

        return self.state


class test_value_policy:

    def policy(self, state):
        """Returns distribution over all allowed actions"""
        # uniform placeholder:
        return np.zeros([state.action_space]) + 1.0/state.action_space

    def value(self, state):
        return 1/(10*abs((state.state - state.target%1.0)) + 1)



    def predict(self, state):
        return self.policy(state), self.value(state)

if __name__ == "__main__":

    start_state = AddDivState()
    tree_root = TreeStructure(start_state)

    hist, rew = play_game(start_state,policy_value=test_value_policy())
    print(hist, rew)