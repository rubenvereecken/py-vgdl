from collections import UserDict, OrderedDict
from abc import ABCMeta, abstractmethod

from vgdl.core import BasicGame
from vgdl.ontology import GridPhysics, Avatar
from vgdl.tools import PrettyDict


class Observation:
    def as_array(self):
        raise NotImplemented()


class KeyValueObservation(PrettyDict, OrderedDict, Observation):
    """
    Currently a glorified dictionary that keeps its contents in the order it's
    received them. For that reason, it is crucial that values are always passed
    in in the same order, as there is currently no other way to enforce order.
    """
    def as_array(self):
        import numpy as np
        return np.hstack(list(self.values()))

    def as_dict(self):
        return self

    def __iter__(self):
        for el in self.as_array():
            yield el


class StateObserver:
    def __init__(self, game: BasicGame) -> None:
        self._game = game

    def get_observation(self):
        raise NotImplemented()

    def _rect_to_pos(self, r):
        return (r.left // self._game.block_size, r.top // self._game.block_size)

    @property
    def observation_shape(self):
        obs = self.get_observation()
        shape = obs.as_array().shape
        return shape

    @property
    def observation_length(self):
        obs = self.get_observation()
        length = len(obs.as_array())
        return length


class AbsoluteObserver(StateObserver):
    """
    - Assumes a single-avatar grid physics game
    - Assumes only the avatar can possess resources
    """
    def __init__(self, game: BasicGame) -> None:
        super().__init__(game)

        avatars = game.getSprites('avatar')
        assert len(avatars) == 1, 'Single avatar'
        avatar = avatars[0]
        assert issubclass(avatar.physicstype, GridPhysics)


    def get_observation(self) -> Observation:
        avatars = self._game.getAvatars()
        assert avatars
        position = self._rect_to_pos(avatars[0].rect)
        observation = KeyValueObservation(x=position[0], y=position[1])
        return observation
