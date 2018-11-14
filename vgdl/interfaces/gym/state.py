from vgdl.state import StateObserver, KeyValueObservation
import math

from typing import Union, List, Dict


class AvatarOrientedObserver(StateObserver):

    def _get_distance(self, s1, s2):
        return math.hypot(s1.rect.x - s2.rect.x, s1.rect.y - s2.rect.y)


    def get_observation(self):
        avatars = self._game.getAvatars()
        assert avatars
        avatar = avatars[0]

        avatar_pos = self._rect_to_pos(avatar.rect)
        resources = [avatar.resources[r] for r in self._game.notable_resources]

        sprite_distances = []
        for key in self._game.sprite_registry.sprite_keys:
            dist = 100
            for s in self._game.getSprites(key):
                dist = min(self._get_distance(avatar, s)/self._game.block_size, dist)
            sprite_distances.append(dist)

        obs = KeyValueObservation(
            position=avatar_pos, speed=avatar.speed, resources=resources,
            distances=sprite_distances
        )
        return obs


class NotableSpritesObserver(StateObserver):
    """
    TODO: There is still a problem with games where the avatar
    transforms into a different type
    """
    def __init__(self, game, notable_sprites: Union[List, Dict] = None):
        super().__init__(game)
        self.notable_sprites = notable_sprites or game.sprite_registry.groups()


    def get_observation(self):
        state = []

        sprite_keys = list(self.notable_sprites)
        num_classes = len(sprite_keys)
        resource_types = self._game.notable_resources

        for i, key in enumerate(sprite_keys):
            class_one_hot = [float(j==i) for j in range(num_classes)]

            # TODO this code is currently unsafe as getSprites does not
            # guarantee the same order for each call (Python < 3.6),
            # meaning observations will have inconsistent ordering of values
            for s in self._game.getSprites(key):
                position = self._rect_to_pos(s.rect)
                if hasattr(s, 'orientation'):
                    orientation = [float(a) for a in s.orientation]
                else:
                    orientation = [0.0, 0.0]

                resources = [ float(s.resources[r]) for r in resource_types ]

                state += [
                    (s.id + '.position', position),
                    (s.id + '.orientation', orientation),
                    (s.id + '.class', class_one_hot),
                    (s.id + '.resources', resources),
                ]

        return KeyValueObservation(state)
