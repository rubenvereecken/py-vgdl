from abc import ABCMeta, abstractmethod
from vgdl.core import BasicGame
from vgdl.ontology import GridPhysics, Avatar


class StateObserver:
    def get_state(self):
        raise NotImplemented()

    def _rect_to_pos(self, r):
        return (r.left / self._game.block_size, r.top / self._game.block_size)


class State:
    def as_array(self):
        pass

    def as_dict(self):
        pass


class AbsoluteObserver(StateObserver):
    """
    - Assumes a single-avatar grid physics game
    - Assumes only the avatar can possess resources
    """
    def __init__(self, game: BasicGame):
        avatars = game.sprite_groups.get('avatar', [])
        assert len(avatars) == 1
        avatar = avatars[0]
        assert issubclass(avatar.physicstype, GridPhysics)

        self._avatar = avatar
        self._game = game


    def get_state(self) -> State:
        return self._rect_to_pos(self._avatar.rect)
