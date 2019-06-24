from collections import OrderedDict

from vgdl.core import BasicGame, BasicGameLevel
from vgdl.ontology import GridPhysics
from vgdl.tools import PrettyDict

import copy


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

    def __hash__(self):
        return hash(tuple(self.items()))

    def merge(self, other):
        out = copy.deepcopy(self)
        out.update(other)
        return out


class StateObserver:
    def __init__(self, game: BasicGame) -> None:
        self.game = game

#     @property
#     def game(self):
#         import ipdb; ipdb.set_trace()
#         return self._game

#     @game.setter
#     def game(self, game):
#         print('>>>>>>>')
#         self._game = game

    def get_observation(self) -> Observation:
        return KeyValueObservation()

    def _rect_to_pos(self, r):
        return r.left // self.game.block_size, r.top // self.game.block_size

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

    def set_game(self, game):
        self.game = game

    def __repr__(self):
        return self.__class__.__name__

    def __getstate__(self):
        state = vars(self).copy()
        state.pop('game', None)
        return state


class AbsoluteObserver(StateObserver):
    """
    - Assumes a single-avatar grid physics game
    - Observation is (x, y) of avatar's rectangle, in pixels
    """

    def __init__(self, game: BasicGame) -> None:
        super().__init__(game)

        avatar = game.sprite_registry.get_avatar()
        assert issubclass(avatar.physicstype, GridPhysics)

    def get_observation(self) -> Observation:
        obs = super().get_observation()
        avatar = self.game.sprite_registry.get_avatar()
        obs = obs.merge(KeyValueObservation(x=avatar.rect.left, y=avatar.rect.top))
        return obs


class AbsoluteGridObserver(StateObserver):
    """
    TODO: This is actually deprecated, get rid of it.
    - Assumes a single-avatar grid physics game
    - Observation is (x, y) of avatar converted to grid (not raw pixels)
    """

    def __init__(self, game: BasicGame) -> None:
        super().__init__(game)

        avatars = game.get_sprites('avatar')
        assert len(avatars) == 1, 'Single avatar'
        avatar = avatars[0]
        assert issubclass(avatar.physicstype, GridPhysics)

    def get_observation(self) -> Observation:
        avatars = self.game.get_avatars()
        assert avatars
        position = self._rect_to_pos(avatars[0].rect)
        observation = KeyValueObservation(x=position[0], y=position[1])
        return observation


class OrientationObserver(StateObserver):
    def __init__(self, game: BasicGame) -> None:
        super().__init__(game)
        from vgdl.ontology import OrientedAvatar
        avatar = game.sprite_registry.get_avatar()
        assert isinstance(avatar, OrientedAvatar)

    def get_observation(self):
        obs = super().get_observation()
        avatar = self.game.sprite_registry.get_avatar()
        obs = obs.merge(KeyValueObservation({
            'orientation.x': avatar.orientation[0],
            'orientation.y': avatar.orientation[1],
        }))
        return obs


class ResourcesObserver(StateObserver):
    def __init__(self, game: BasicGameLevel) -> None:
        super().__init__(game)
        # TODO verify it's a resource avatar

    def get_observation(self):
        obs = super().get_observation()
        avatar = self.game.sprite_registry.get_avatar()
        resources = { key: avatar.resources.get(key, 0) for key in self.game.domain.notable_resources }
        obs = obs.merge(KeyValueObservation(resources))
        return obs


class PositionAndResourceObserver(AbsoluteObserver, ResourcesObserver):
    pass
