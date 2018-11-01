import itertools
import logging
from typing import NewType, Optional, Union, Dict, List, Tuple
from math import sqrt

import numpy as np
import pygame
from pygame.math import Vector2

from vgdl.core import VGDLSprite, Avatar, Action, Resource
from vgdl.tools import unitVector
from .constants import *
from .sprites import SpriteProducer, OrientedSprite
from .physics import GridPhysics, ContinuousPhysics, GravityPhysics


__all__ = [
    'AimedAvatar',
    'AimedFlakAvatar',
    'Avatar',
    'FlakAvatar',
    'HorizontalAvatar',
    'InertialAvatar',
    'MarioAvatar',
    'MovingAvatar',
    'NoisyRotatingFlippingAvatar',
    'OrientedAvatar',
    'RotatingAvatar',
    'RotatingFlippingAvatar',
    'ShootAvatar',
    'VerticalAvatar',
]


class MovingAvatar(VGDLSprite, Avatar):
    """ Default avatar, moves in the 4 cardinal directions. """
    color = WHITE
    speed = 1 # type: Optional[int]
    is_avatar = True
    alternate_keys=False


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        possible_actions = self.__class__.declare_possible_actions()
        self.keys_to_action = {tuple(sorted(a.keys)): a for a in possible_actions.values()}


    @classmethod
    def declare_possible_actions(cls) -> Dict[str, Action]:
        """
        Assume this does not change throughout the game. That is, we commit
        to the semantics that all actions are always possible, no matter
        whether they will actually have an effect or not.

        Composite actions (multiple keys) must be defined separately.
        It is important that a composite action is defined explicitly,
        as most RL agents work with enumerated actions instead of
        actions represented by multi-dimensional vectors (i.e. keypresses).
        """
        from pygame.locals import K_LEFT, K_RIGHT, K_UP, K_DOWN
        actions = {}
        actions["UP"] = Action(K_UP)
        actions["DOWN"] = Action(K_DOWN)
        actions["LEFT"] = Action(K_LEFT)
        actions["RIGHT"] = Action(K_RIGHT)
        actions["NO_OP"] = Action()
        return actions

    def _readAction(self, game) -> Action:
        """
        An action can consist of multiple key presses. The action corresponding
        to the most key presses will be returned. Ties are broken arbitrarily.
        """
        active_keys = tuple(sorted(game.active_keys))

        # Up to 3 buttons active at a time, at least 0,
        for num_keys in range(max(3, len(active_keys)), -1, -1):

            # Find the longest key combo that matches a known action
            for key_combo in itertools.combinations(active_keys, num_keys):
                if key_combo in self.keys_to_action:
                    return self.keys_to_action[key_combo]

        assert False, 'No valid actions encountered, consider allowing NO_OP'


    def update(self, game):
        VGDLSprite.update(self, game)
        action = self._readAction(game)
        if not action == NOOP:
            self.physics.activeMovement(self, action)

class HorizontalAvatar(MovingAvatar):
    """ Only horizontal moves.  """

    @classmethod
    def declare_possible_actions(cls):
        from pygame.locals import K_LEFT, K_RIGHT
        actions = {}
        actions["LEFT"] = Action(K_LEFT)
        actions["RIGHT"] = Action(K_RIGHT)
        actions["NO_OP"] = Action()
        return actions


    def update(self, game):
        VGDLSprite.update(self, game)
        action = self._readAction(game)
        if action.as_vector() in [RIGHT, LEFT]:
            self.physics.activeMovement(self, action)

class VerticalAvatar(MovingAvatar):
    """ Only vertical moves.  """

    @classmethod
    def declare_possible_actions(cls):
        from pygame.locals import K_UP, K_DOWN
        actions = {}
        actions["UP"] = Action(K_UP)
        actions["DOWN"] = Action(K_DOWN)
        actions["NO_OP"] = Action()
        return actions

    def update(self, game):
        VGDLSprite.update(self, game)
        action = self._readAction(game)
        if action.as_vector() in [UP, DOWN]:
            self.physics.activeMovement(self, action)

class FlakAvatar(HorizontalAvatar, SpriteProducer):
    """ Hitting the space button creates a sprite of the
    specified type at its location. """

    @classmethod
    def declare_possible_actions(cls):
        from pygame.locals import K_SPACE
        actions = super().declare_possible_actions()
        actions["SPACE"] = Action(K_SPACE)
        return actions

    color = GREEN
    def update(self, game):
        HorizontalAvatar.update(self, game)
        self._shoot(game)

    def _shoot(self, game):
        from pygame.locals import K_SPACE
        if self.stype and K_SPACE in game.active_keys:
            game.create_sprite(self.stype, (self.rect.left, self.rect.top))

class OrientedAvatar(OrientedSprite, MovingAvatar):
    """ Avatar retains its orientation, but moves in cardinal directions. """
    draw_arrow = True
    def update(self, game):
        tmp = self.orientation
        self.orientation = (0, 0)
        VGDLSprite.update(self, game)
        action = self._readAction(game)
        if action:
            self.physics.activeMovement(self, action)
        d = self.lastdirection
        if sum(map(abs, d)) > 0:
            # only update if the sprite moved.
            self.orientation = d
        else:
            self.orientation = tmp

class RotatingAvatar(OrientedSprite, MovingAvatar):
    """ Avatar retains its orientation, and moves forward/backward or rotates
    relative to that. """
    draw_arrow = True
    speed = 0
    def update(self, game):
        actions = self._readMultiActions(game)
        if UP in actions:
            self.speed = 1
        elif DOWN in actions:
            self.speed = -1
        if LEFT in actions:
            i = BASEDIRS.index(self.orientation)
            self.orientation = BASEDIRS[(i + 1) % len(BASEDIRS)]
        elif RIGHT in actions:
            i = BASEDIRS.index(self.orientation)
            self.orientation = BASEDIRS[(i - 1) % len(BASEDIRS)]
        VGDLSprite.update(self, game)
        self.speed = 0

class RotatingFlippingAvatar(RotatingAvatar):
    """ Uses a different action set: DOWN makes it spin around 180 degrees.
    Optionally, a noise level can be specified
    """

    noiseLevel = 0.

    def update(self, game):
        actions = self._readMultiActions(game)
        if len(actions) > 0 and self.noiseLevel > 0:
            # pick a random one instead
            if game.random_generator.random() < self.noiseLevel*4:
                actions = [game.random_generator.choice([UP, LEFT, DOWN, RIGHT])]
        if UP in actions:
            self.speed = 1
        elif DOWN in actions:
            i = BASEDIRS.index(self.orientation)
            self.orientation = BASEDIRS[(i + 2) % len(BASEDIRS)]
        elif LEFT in actions:
            i = BASEDIRS.index(self.orientation)
            self.orientation = BASEDIRS[(i + 1) % len(BASEDIRS)]
        elif RIGHT in actions:
            i = BASEDIRS.index(self.orientation)
            self.orientation = BASEDIRS[(i - 1) % len(BASEDIRS)]
        VGDLSprite.update(self, game)
        self.speed = 0

    @property
    def is_stochastic(self):
        return self.noiseLevel > 0

class NoisyRotatingFlippingAvatar(RotatingFlippingAvatar):
    noiseLevel = 0.1

class ShootAvatar(OrientedAvatar, SpriteProducer):
    """ Produces a sprite in front of it (e.g., Link using his sword). """
    ammo=None

    @classmethod
    def declare_possible_actions(cls):
        from pygame.locals import K_SPACE
        actions = super().declare_possible_actions()
        actions["SPACE"] = Action(K_SPACE)
        return actions

    def __init__(self, stype=None, **kwargs):
        self.stype = stype
        OrientedAvatar.__init__(self, **kwargs)

    def update(self, game):
        OrientedAvatar.update(self, game)
        if self._hasAmmo():
            self._shoot(game)

    def _hasAmmo(self):
        if self.ammo is None:
            return True
        elif self.ammo in self.resources:
            return self.resources[self.ammo] > 0
        return False

    def _reduceAmmo(self):
        if self.ammo is not None and self.ammo in self.resources:
            self.resources[self.ammo] -= 1

    def _shoot(self, game):
        from pygame.locals import K_SPACE
        if self.stype and K_SPACE in game.active_keys:
            u = unitVector(self.orientation)
            sprite = game.create_sprite(self.stype, (self.lastrect.left + u[0] * self.lastrect.size[0],
                                                       self.lastrect.top + u[1] * self.lastrect.size[1]))
            if sprite and isinstance(sprite, OrientedSprite):
                sprite.orientation = unitVector(self.orientation)

            self._reduceAmmo()

class AimedAvatar(ShootAvatar):
    """ Can change the direction of firing, but not move. """
    speed = 0 # type: Optional[int]
    angle_diff=0.05

    def update(self, game):
        VGDLSprite.update(self, game)
        self._aim(game)
        self._shoot(game)

    def _aim(self, game):
        action = self._readAction(game)
        if action in [UP, DOWN]:
            if action == DOWN:
                angle = self.angle_diff
            else:
                angle = -self.angle_diff
            from math import cos, sin
            self.orientation = unitVector((self.orientation[0]*cos(angle)-self.orientation[1]*sin(angle),
                                           self.orientation[0]*sin(angle)+self.orientation[1]*cos(angle)))

class AimedFlakAvatar(AimedAvatar):
    """ Can move left and right """
    only_active=True
    speed=None

    def update(self, game):
        AimedAvatar.update(self, game)
        action = self._readAction(game)
        if action in [RIGHT, LEFT]:
            self.physics.activeMovement(self, action)

class InertialAvatar(OrientedAvatar):
    speed = 1
    physicstype = ContinuousPhysics
    def update(self, game):
        MovingAvatar.update(self, game)

class MarioAvatar(OrientedAvatar):
    """
    Inertia only works when airborne. There is no inertia when walking.
    """
    physicstype = GravityPhysics
    draw_arrow = False
    strength = 3
    jump_strength = 10
    airsteering = False

    # active_force is only used during event handling, so don't save it
    state_attributes = OrientedAvatar.state_attributes + ['passive_force']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passive_force = (0, 0)
        self.active_force = (0, 0)

    def resolve_active_force(self, action, game):
        from pygame.locals import K_SPACE

        # Force will just have a horizontal component as we do not allow up/down
        force = action.as_vector()

        if K_SPACE in action.keys and self.passive_force[1] == 0:
            # Not airborne and attempting to jump
            force = (force[0] * self.strength, -self.jump_strength)
        elif self.passive_force[1] != 0 and self.airsteering:
            # Airborne and actively steering
            force = (force[0] * self.strength, 0)
        elif self.passive_force[1] != 0 and not self.airsteering:
            # Airborne and not allowed to steer, so just let fly
            force = (0, 0)
        elif self.passive_force[1] == 0 and force[0]:
            # Actively walking along, you want the net velocity to be fixed
            force = (force[0] * self.strength, 0)
        else:
            # Not walking, should actively halt
            force = (0, 0)

        return force


    def resolve_passive_force(self, action, game):
        """
        I have not decided whether this should go inside Physics.
        This is really part of the physics, but it's up to the avatar implementation
        to decide when to ignore physics such as when airsteering.

        Things like friction, sticky floors, should be resolved here.
        """
        if self.passive_force[1] != 0 and not self.airsteering:
            force = (0, 0)
        else:
            # This undoes horizontal velocity, so we stop when we don't move
            force = (-self.velocity[0] / self.mass, 0)

        return force


    def update(self, game):
        action = self._readAction(game)
        active_force = self.resolve_active_force(action, game)
        passive_force = self.resolve_passive_force(action, game)
        self.active_force = Vector2(active_force)

        self.physics.activeMovement(self, Vector2(active_force) + passive_force)

        if not self.is_static and not self.only_active:
            # Resolve gravity, also sticky floors
            self.physics.passiveMovement(self)

        # Body of VGDLSprite.update
        self.lastrect = self.rect
        self.lastmove += 1

        self._updatePos(self.orientation, self.speed)


    @classmethod
    def declare_possible_actions(cls):
        from pygame.locals import K_SPACE, K_RIGHT, K_LEFT
        actions = super().declare_possible_actions()
        actions['SPACE'] = Action(K_SPACE)
        actions['SPACE_RIGHT'] = Action(K_SPACE, K_RIGHT)
        actions['SPACE_LEFT'] = Action(K_SPACE, K_LEFT)
        return actions


