import numpy as np

from pybrain.rl.environments.environment import Environment
from pybrain.rl.environments.episodic import EpisodicTask

from typing import List, Tuple, Dict, Union
from vgdl.core import Action
from vgdl.state import StateObserver, Observation


class VGDLPybrainEnvironment(Environment):
    """
    Assumptions:
        - Single avatar
        - See relevant state handler for their assumptions

    TODO:
        - Test unspecified avatar start state
    """
    # Pybrain Environment attributes
    discreteActions = True

    # BasicGame
    headless = True

    def __init__(self, game, state_handler):
        self.game = game
        self.state_handler = state_handler

        self.action_set: List[Action] = list(game.get_possible_actions().values())
        # self.init_state = state_handler.get_state()
        self.init_game_state = game.get_game_state()

        # Pybrain Environment attributes
        self.numActions = len(self.action_set)

        self.reset(init=True)

        # Some observers need pygame initialised first
        self.init_observation = state_handler.get_observation()
        self.outdim = len(self.init_observation)


    def reset(self, init=False):
        # This wil reset to initial game state
        self.game.reset()


    def getSensors(self):
        return self.state_handler.get_observation()


    def prune_action_set(self, actions):
        if not hasattr(actions, '__iter__'):
            actions = [actions]
        for action in actions:
            assert action in self.action_set, \
                'Unknown action %s' % action
            self.action_set.remove(action)
        self.numActions = len(self.action_set)


    def performAction(self, action: Union[Action, int]):
        if not isinstance(action, Action):
            action = self.action_set[int(action)]
        assert isinstance(action, Action)
        self.game.tick(action)


class VGDLPybrainTask(EpisodicTask):
    """
    Our Task is really just a wrapper for the environment, which represents
    a specific game with reward functi
    on and everything and is itself episodic.
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.get_observation = self.getObservation
        self.perform_action = self.performAction

    def getReward(self):
        return self.env.game.last_reward

    def isFinished(self):
        return self.env.game.ended

    # A Pybrain task is supposed to constrain the observations
    # def getObservation(self):
    #     pass


import wrapt

class SparseMDPObserver(wrapt.ObjectProxy):
    """
    Assigns each state a unique id so states become discrete.
    Used when feature domains are sparse (and hence probably discrete).
    Observations must be hashable.
    """
    def __init__(self, wrapped):
        super().__init__(wrapped)
        self._obs_cache: Dict[Observation, int] = {}

    def get_observation(self, obs=None):
        # Allow passing in an observation, to be cached.
        # Otherwise get it from the underlying observer
        if obs is None:
            obs = self.__wrapped__.get_observation()
        if obs not in self._obs_cache:
            self._obs_cache[obs] = np.array([len(self._obs_cache)])
        return self._obs_cache[obs]
