import numpy as np
from pybrain.utilities import flood

from typing import Any, List, Optional, Tuple

from vgdl.core import GameState
from vgdl.state import Observation


class MDPConverter:
    """
    This class relies entirely on the Pybrain Environment interface
    """

    def __init__(self, task):
        self.task = task
        self.env = task.env
        self.game = task.env.game

        self.states: List[GameState] = []
        # [(s, a, s')]
        self.transitions: List[Tuple[GameState, int, GameState]] = []
        # S' -> R
        self.rewards: Dict[GameState, int] = {self.env.init_game_state: 0}

        assert not self.game.is_stochastic, 'TODO: stochastic env'

    def convert_task_to_mdp(self):
        # Finds all states, all the while logging transitions and rewards
        self.states = sorted(flood(self.get_neighbors, None, [self.env.init_game_state]))
        state_dict = {state: state_i for state_i, state in enumerate(self.states)}

        # Reward function R(s')
        R = np.fromiter((self.rewards[state] for state in self.states), dtype=np.double)

        # Transition matrix A x S x S
        T = np.zeros((self.env.numActions, len(self.states), len(self.states)))

        for state, action_i, next_state in self.transitions:
            # Careful, states are actual states but action_i is an index
            T[action_i, state_dict[state], state_dict[next_state]] = 1

        return T, R

    def get_neighbors(self, state: GameState, save_transitions=True) -> List[GameState]:
        """
        For use by pybrain.utilities.flood
        Also logs (s,a,s') transitions and R(s')
        Will leave the environment in an arbitrary neighbor state
        """
        # Final states are absorbing state
        if state.ended():
            return []

        # TODO maybe want to move this to core.py, if it turns out useful
        # Should keep everything intact, including random state
        def _get_neighbor(action_i, action):
            assert not self.game.is_stochastic, 'Deterministic only'
            action = self.env.action_set[action_i]
            # TODO will have to not include random state for stochastic
            # game_state = self.game.getGameState(include_random_state=True)
            self.game.set_game_state(state)
            self.task.performAction(action)
            next_state = self.game.get_game_state()

            if save_transitions:
                # (s, a, s')
                self.transitions.append((state, action_i, next_state))
                # Assume R(s'), not R(s, a, s')
                self.rewards[next_state] = self.task.getReward()

            return next_state

        next_states = [_get_neighbor(action_i, action) for action_i, action \
                       in enumerate(self.env.action_set)]
        return next_states

    def get_observations(self, states: Optional[List[GameState]] = None) -> List[Observation]:
        """
        Get observations corresponding to states.
        Only works after the initial conversion where self.states is populated.
        Leaves the environment in an arbitrary state.
        """
        states = states or self.states
        assert len(states) > 0, 'Must run a conversion method first, or pass in states'

        def _get_observation(state: GameState):
            self.game.set_game_state(state)
            observation = self.task.getObservation()
            return observation

        observations = [_get_observation(state) for state in states]
        return observations
