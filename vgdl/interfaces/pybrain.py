import numpy as np

from pybrain.rl.environments.environment import Environment
from pybrain.rl.environments.episodic import EpisodicTask

from typing import List
from vgdl.core import Action


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
        self.init_observation = state_handler.get_observation()

        # Pybrain Environment attributes
        self.numActions = len(self.action_set)
        self.outdim = len(self.init_observation)

        self.reset(init=True)
        print(self.action_set)


    def reset(self, init=False):
        self.game.reset()
        self.game.initScreen(self.headless)

        if not init:
            # self.game.setState(self.init_state)
            self.game.setGameState(self.init_game_state)

        # TODO test this
        self.game.randomizeAvatar()


    def getSensors(self):
        return self.state_handler.get_observation()


    def performAction(self, action: Action):
        self.game.tick(action)


class VGDLPybrainTask(EpisodicTask):
    # def __init__(self):
    #     super()


    def getReward(self):
        # TODO this is actually accumulated, but let's roll for now
        # TODO should make score change available in game
        score = self.env.game.score
        return score

    # A Pybrain task is supposed to constrain the observations
    # def getObservation(self):
    #     pass
