import logging
from typing import NewType, Optional, Union, Dict, List, Tuple

import pygame
from pygame.math import Vector2

from vgdl.core import Termination


__all__ = [
    'SpriteCounter',
    'MultiSpriteCounter',
    'Timeout',
]


class Timeout(Termination):
    def __init__(self, limit=0, win=False):
        self.limit = limit
        self.win = win

    def isDone(self, game):
        if game.time >= self.limit:
            return True, self.win
        else:
            return False, None

class SpriteCounter(Termination):
    """ Game ends when the number of sprites of type 'stype' hits 'limit' (or below). """
    def __init__(self, limit=0, stype=None, win=True):
        self.limit = limit
        self.stype = stype
        self.win = win

    def __repr__(self):
        return 'SpriteCounter(stype={})'.format(self.stype)

    def isDone(self, game):
        if game.numSprites(self.stype) <= self.limit:
            return True, self.win
        else:
            return False, None

class MultiSpriteCounter(Termination):
    """ Game ends when the sum of all sprites of types 'stypes' hits 'limit'. """
    def __init__(self, limit=0, win=True, **kwargs):
        self.limit = limit
        self.win = win
        self.stypes = kwargs.values()

    def isDone(self, game):
        if sum([game.numSprites(st) for st in self.stypes]) == self.limit:
            return True, self.win
        else:
            return False, None

