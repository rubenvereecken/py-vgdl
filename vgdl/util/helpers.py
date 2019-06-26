import pygame
from pygame.math import Vector2


def neighbor_position(rect: pygame.Rect, direction: Vector2) -> Vector2:
    direction = Vector2(direction)
    topleft = rect.topleft + direction.elementwise() * rect.size
    return topleft

