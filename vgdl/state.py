from abc import ABCMeta, abstractmethod
from vgdl.core import BasicGame
from vgdl.ontology import GridPhysics, Avatar


class StateObserver:
    def get_observation(self):
        raise NotImplemented()

    def _rect_to_pos(self, r):
        return (r.left / self._game.block_size, r.top / self._game.block_size)


class Observation:
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
        avatars = game.getSprites('avatar')
        assert len(avatars) == 1, 'Single avatar'
        avatar = avatars[0]
        assert issubclass(avatar.physicstype, GridPhysics)

        # TODO it is currently unsafe to keep a reference to a sprite
        # self._avatar = avatar
        self._game = game


    def get_observation(self) -> Observation:
        avatars = self._game.getAvatars()
        if not avatars:
            import ipdb; ipdb.set_trace()
        return self._rect_to_pos(avatars[0].rect)
