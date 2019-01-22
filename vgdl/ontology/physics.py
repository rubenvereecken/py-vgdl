import itertools
import logging
from typing import NewType, Optional, Union, Dict, List, Tuple

import numpy as np
import pygame
from pygame.math import Vector2

from vgdl.core import Action, Physics
from vgdl.ontology.constants import NOOP

__all__ = [
    'GridPhysics',
    'ContinuousPhysics',
    'GravityPhysics'
]

class GridPhysics(Physics):
    """ Define actions and key-mappings for grid-world dynamics. """

    def passive_movement(self, sprite):
        """
        TODO This can be slightly dangerous and should probably be refactored.
        All sprites with an orientation and speed and GridPhysics will automatically
        move in that direction, passively.
        """
        if sprite.speed is None:
            speed = 1
        else:
            speed = sprite.speed
        if speed != 0 and hasattr(sprite, 'orientation'):
            sprite._update_position(sprite.orientation, speed * self.gridsize[0])

    def active_movement(self, sprite, action, speed=None):
        if speed is None:
            if sprite.speed is None:
                speed = 1
            else:
                speed = sprite.speed
        if speed != 0 and action is not None and action != NOOP:
            # TODO have all actions be Action
            if isinstance(action, Action):
                action = action.as_vector()
            sprite._update_position(action, speed * self.gridsize[0])


    def distance(self, r1, r2):
        """ Grid physics use Hamming distances. """
        return (abs(r1.top - r2.top)
                + abs(r1.left - r2.left))


class ContinuousPhysics(GridPhysics):
    def passive_movement(self, sprite):
        if not hasattr(sprite, 'orientation'):
            return
        # sprite._updatePos(sprite.orientation, sprite.speed)
        # if self.gravity > 0 and sprite.mass > 0:
        #     sprite.passive_force = (0, self.gravity * sprite.mass)
        #     self.activeMovement(sprite, sprite.passive_force)

        if self.gravity > 0 and sprite.mass > 0:
            sprite.passive_force = Vector2(0, self.gravity * sprite.mass)
            self.active_movement(sprite, sprite.passive_force)


    def active_movement(self, sprite, force, speed=None):
        """
        Updates sprite.orientation and sprite.speed, which together make up
        the sprite's velocity.
        """

        # TODO understand how this is now completely different from
        # gridphysics activeMovement, and why I changed it to be so?
        # Might have to grab old schematics.

        if speed is None:
            old_velocity = sprite.velocity
        else:
            old_velocity = sprite.orientation * speed

        force = Vector2(force)
        velocity = old_velocity + force / sprite.mass

        sprite.velocity = velocity


    def distance(self, r1, r2):
        """ Continuous physics use Euclidean distances. """
        return (Vector2(r1.topleft) - Vector2(r2.topleft)).normalize()

class GravityPhysics(ContinuousPhysics):
    gravity = 1



