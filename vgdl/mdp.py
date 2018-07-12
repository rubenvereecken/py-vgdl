import numpy as np
from pybrain.utilities import flood

import vgdl.interfaces
from vgdl.interfaces.pybrain import VGDLPybrainEnvironment

from typing import Any, List
State = Any


class MDPConverter:
    """
    This class relies entirely on the Pybrain Environment interface
    """
    def __init__(self, task):
        self.task = task
        self.env = task.env
        self.game = task.env.game

        # [(s, a, s')]
        self.transitions: List[State, int, State] = []
        # S' -> R
        self.rewards = {}

        assert not self.game.is_stochastic, 'TODO: stochastic env'


    def convert_task_to_mdp(self):
        # Finds all states, all the while logging transitions and rewards
        self.states = sorted(flood(self.get_neighbors, None, [self.env.init_state]))
        state_dict = { state: state_i for state_i, state in enumerate(self.states) }

        # Reward function R(s')
        R = np.fromiter((self.rewards[state] for state in self.states), dtype=np.double)

        # Transition matrix A x S x S
        T = np.zeros((self.env.numActions, len(self.states), len(self.states)))

        for state, action_i, next_state in self.transitions:
            # Careful, states are actual states but action_i is an index
            T[action_i, state_dict[state], state_dict[next_state]] = 1

        return T, R



    def get_neighbors(self, state, save_transitions=True):
        """
        For use by pybrain.utilities.flood
        Also logs (s,a,s') transitions and R(s')
        """

        # TODO maybe want to move this to core.py, if it turns out useful
        # Should keep everything intact, including random state
        def _get_neighbor(action_i, action):
            assert not self.game.is_stochastic
            action = self.env.action_set[action_i]
            # TODO will have to not include random state for stochastic
            game_state = self.game.getGameState(include_random_state=True)
            self.task.performAction(action)
            next_state = self.task.getObservation()

            if save_transitions:
                # (s, a, s')
                self.transitions.append((state, action_i, next_state))
                # Assume R(s'), not R(s, a, s')
                self.rewards[next_state] = self.task.getReward()

            self.game.setGameState(game_state) # include random state
            return next_state

        next_states = [_get_neighbor(action_i, action) for action_i, action \
                       in enumerate(self.env.action_set)]
        return next_states

