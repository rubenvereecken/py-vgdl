import logging
from typing import NewType, Optional, Union, Dict, List, Tuple

import pygame
from pygame.math import Vector2

from vgdl.core import Termination


__all__ = [
    'SpriteCounter',
    'MultiSpriteCounter',
    'ResourceCounter',
    'Timeout',
]


class Timeout(Termination):
    def __init__(self, limit=0):
        super().__init__(**kwargs)
        self.limit = limit
        self.win = win

    def is_done(self, game):
        if game.time >= self.limit:
            return True, self.win
        else:
            return False, None

class SpriteCounter(Termination):
    """ Game ends when the number of sprites of type 'stype' hits 'limit' (or below). """
    def __init__(self, limit=0, stype=None, **kwargs):
        super().__init__(**kwargs)
        self.limit = limit
        self.stype = stype

    def __repr__(self):
        return 'SpriteCounter(stype={})'.format(self.stype)

    def is_done(self, game):
        if game.num_sprites(self.stype) <= self.limit:
            return True, self.win
        else:
            return False, None

class MultiSpriteCounter(Termination):
    """ Game ends when the sum of all sprites of types 'stypes' hits 'limit'. """
    def __init__(self, limit=0, win=True, **kwargs):
        super().__init__(win)
        self.limit = limit
        self.stypes = kwargs.values()

    def is_done(self, game):
        if sum([game.num_sprites(st) for st in self.stypes]) == self.limit:
            return True, self.win
        else:
            return False, None


class ResourceCounter(Termination):
    def __init__(self, stype, limit, **kwargs):
        super().__init__(**kwargs)
        self.stype = stype
        self.limit = limit

    def is_done(self, game):
        avatar = game.get_avatars()[0]
        satisfied = avatar.resources.get(self.stype, 0) >= self.limit

        return satisfied, self.win
