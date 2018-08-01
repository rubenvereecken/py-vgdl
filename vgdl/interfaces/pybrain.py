import numpy as np

from pybrain.rl.environments.environment import Environment
from pybrain.rl.environments.episodic import EpisodicTask

from typing import List, Tuple, Dict
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

        self.action_set: List[Action] = list(game.getPossibleActions().values())
        # self.init_state = state_handler.get_state()
        self.init_game_state = game.getGameState()

        # Pybrain Environment attributes
        self.numActions = len(self.action_set)

        self.reset(init=True)

        # Some observers need pygame initialised first
        self.init_observation = state_handler.get_observation()
        self.outdim = len(self.init_observation)


    def reset(self, init=False):
        # This wil reset to initial game state
        self.game.reset()
        self.game.initScreen(self.headless)

        # TODO at some point we might want random locations
        # self.game.randomizeAvatar()


    def getSensors(self):
        return self.state_handler.get_observation()


    def performAction(self, action: Action):
        if not isinstance(action, Action):
            action = self.action_set[int(action)]
        assert isinstance(action, Action)
        self.game.tick(action)


class VGDLPybrainTask(EpisodicTask):
    """
    Our Task is really just a wrapper for the environment, which represents
    a specific game with reward function and everything and is itself episodic.
    """

    def getReward(self):
        # TODO this is actually accumulated, but let's roll for now
        # TODO should make score change available in game
        score = self.env.game.score
        return score

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

    def get_observation(self):
        obs = self.__wrapped__.get_observation()
        if obs not in self._obs_cache:
            self._obs_cache[obs] = [len(self._obs_cache)]
        return self._obs_cache[obs]

