from functools import *
from collections import UserDict, defaultdict
from math import sqrt
import pygame
from pygame.math import Vector2


def _fmt_value(v):
    return v


class PrettyDict:
    """
    Mixin for pretty user-defined dictionary printing
    """
    pretty_name = None

    def __repr__(self):
        attributes = ', '.join('{}={!r}'.format(k, _fmt_value(v)) for k, v in self.items())
        return '{}({})'.format(self.__class__.__name__, attributes)

    def __str__(self):
        attributes = ', '.join('{}={!s}'.format(k, _fmt_value(v)) for k, v in self.items())
        return '{}({})'.format(self.pretty_name or self.__class__.__name__, attributes)


class PrettyClass:
    def __repr__(self):
        attributes = ','.join('{}={!r}'.format(k, v) for k, v in vars(self).items())
        return '{}({})'.format(self.__class__.__name__, attributes)



_is_dict = lambda d: isinstance(d, dict) or isinstance(d, UserDict)


def freeze_dict(original, freezers={}):
    """
    - Assumes d is immutable
    - Assumes item ordering doesn't matter (ignores OrderedDict)
    - Assumes lists are unordered. This is a big one actually
        .. so how about you use tuples for sorted things? Hmm.
    - Assumes False, 0 and None are interchangeable
    """
    if not _is_dict(original):
        return original

    d = {}

    for k, v in original.items():
        vtype = type(v)
        if vtype in freezers:
            d[k] = freezers[vtype](v)
        elif _is_dict(v):
            d[k] = freeze_dict(v, freezers)
        elif isinstance(v, list):
            v = frozenset(freeze_dict(el, freezers) for el in v)
            d[k] = v
        elif isinstance(v, pygame.Rect) or isinstance(v, pygame.math.Vector2):
            d[k] = tuple(v)
        elif v is None:
            # Careful: None and 0 will be considered the same!
            d[k] = 0
        else:
            d[k] = v

    return frozenset(d.items())


def unit_vector(v):
    v = Vector2(v)
    if v.length() > 0:
        return v.normalize()
    else:
        return Vector2(1, 0)


def once_per_step(sprite, game, name):
    """ Utility for guaranteeing that an event gets triggered only once per time-step on each sprite. """
    if name in sprite._effect_data:
        if sprite._effect_data[name] == game.time:
            return False
    sprite._effect_data[name] = game.time
    return True
