from typing import Union

import pygame
from pygame.math import Vector2


def neighbor_position(target: Union[pygame.Rect, Vector2], direction: Vector2) -> Vector2:
    # NOTE recommended use with pygame.Rect!
    if isinstance(target, pygame.Rect):
        topleft = target.topleft
        size = target.size
    else:
        topleft = target
        size = (1, 1)

    direction = Vector2(direction)
    topleft = topleft + direction.elementwise() * size
    return topleft

