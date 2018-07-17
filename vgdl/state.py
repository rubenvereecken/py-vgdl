from collections import UserDict
from abc import ABCMeta, abstractmethod

from vgdl.core import BasicGame
from vgdl.ontology import GridPhysics, Avatar
from vgdl.tools import PrettyDict


class StateObserver:
    def get_observation(self):
        raise NotImplemented()

    def _rect_to_pos(self, r):
        return (r.left // self._game.block_size, r.top // self._game.block_size)


class Observation(PrettyDict, UserDict):
    def as_array(self):
        raise NotImplemented()

    def as_dict(self):
        return self.data


class AbsoluteObserver(StateObserver):
    """
    - Assumes a single-avatar grid physics game
    - Assumes only the avatar can possess resources
    """
    def __init__(self, game: BasicGame) -> None:
        avatars = game.getSprites('avatar')
        assert len(avatars) == 1, 'Single avatar'
        avatar = avatars[0]
        assert issubclass(avatar.physicstype, GridPhysics)

        self._game = game


    def get_observation(self) -> Observation:
        avatars = self._game.getAvatars()
        assert avatars
        position = self._rect_to_pos(avatars[0].rect)
        observation = Observation(x=position[0], y=position[1])
        return observation
