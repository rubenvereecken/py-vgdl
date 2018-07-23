'''
Video game lling to description language -- ontology of concepts.

@author: Tom Schaul
'''

import numpy as np
from math import sqrt
import itertools
import pygame
from pygame.math import Vector2
import logging
from typing import NewType, Optional, Union, Dict, List, Tuple

from .tools import triPoints, unitVector, vectNorm, oncePerStep
from .core import Action, Color, Direction


# ---------------------------------------------------------------------
#     Constants
# ---------------------------------------------------------------------
GREEN = Color((0, 200, 0))
BLUE = Color((0, 0, 200))
RED = Color((200, 0, 0))
GRAY = Color((90, 90, 90))
WHITE = Color((250, 250, 250))
BROWN = Color((140, 120, 100))
BLACK = Color((0, 0, 0))
ORANGE = Color((250, 160, 0))
YELLOW = Color((250, 250, 0))
PINK = Color((250, 200, 200))
GOLD = Color((250, 212, 0))
LIGHTRED = Color((250, 50, 50))
LIGHTORANGE = Color((250, 200, 100))
LIGHTBLUE = Color((50, 100, 250))
LIGHTGREEN = Color((50, 250, 50))
LIGHTGRAY = Color((150, 150, 150))
DARKGRAY = Color((30, 30, 30))
DARKBLUE = Color((20, 20, 100))

UP = Direction((0, -1))
DOWN = Direction((0, 1))
LEFT = Direction((-1, 0))
RIGHT = Direction((1, 0))

BASEDIRS = [UP, LEFT, DOWN, RIGHT]

NOOP = Action()

# ---------------------------------------------------------------------
#     Types of physics
# ---------------------------------------------------------------------
class Physics:
    pass

class GridPhysics(Physics):
    """ Define actions and key-mappings for grid-world dynamics. """

    def passiveMovement(self, sprite):
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
            sprite._updatePos(sprite.orientation, speed * self.gridsize[0])

    def activeMovement(self, sprite, action, speed=None):
        if speed is None:
            if sprite.speed is None:
                speed = 1
            else:
                speed = sprite.speed
        if speed != 0 and action is not None:
            sprite._updatePos(action, speed * self.gridsize[0])


    def distance(self, r1, r2):
        """ Grid physics use Hamming distances. """
        return (abs(r1.top - r2.top)
                + abs(r1.left - r2.left))


class ContinuousPhysics(GridPhysics):
    def passiveMovement(self, sprite):
        if not hasattr(sprite, 'orientation'):
            return
        sprite._updatePos(sprite.orientation, sprite.speed)
        if self.gravity > 0 and sprite.mass > 0:
            sprite.passive_force = (0, self.gravity * sprite.mass)
            self.activeMovement(sprite, sprite.passive_force)

    def activeMovement(self, sprite, force, speed=None):
        """
        Updates sprite.orientation and sprite.speed, which together make up
        the sprite's velocity.
        """
        if speed is None:
            speed = sprite.speed
        force = np.array(force)
        orientation = np.array(sprite.orientation)
        speed = np.array(speed)

        old_velocity = orientation * speed
        velocity = old_velocity + force / sprite.mass

        sprite.update_velocity(velocity)


    def distance(self, r1, r2):
        """ Continuous physics use Euclidean distances. """
        return np.linalg.norm(np.array(r1.topleft) - np.array(r2.topleft))

class GravityPhysics(ContinuousPhysics):
    gravity = 1


# ---------------------------------------------------------------------
#     Sprite types
# ---------------------------------------------------------------------
from .core import VGDLSprite, Resource

class Immovable(VGDLSprite):
    """ A gray square that does not budge. """
    color = GRAY
    is_static = True

class Passive(VGDLSprite):
    """ A square that may budge. """
    color = RED

class ResourcePack(Resource):
    """ Can be collected, and in that case adds/increases a progress bar on the collecting sprite.
    Multiple resource packs can refer to the same type of base resource. """
    is_static = True

class Flicker(VGDLSprite):
    """ A square that persists just a few timesteps. """
    color = RED
    limit = 1
    def __init__(self, **kwargs):
        self._age = 0
        VGDLSprite.__init__(self, **kwargs)

    def update(self, game):
        VGDLSprite.update(self, game)
        if self._age > self.limit:
            killSprite(self, None, game)
        self._age += 1

class Spreader(Flicker):
    """ Spreads to its four canonical neighbor positions, and replicates itself there,
    if these are unoccupied. """
    spreadprob = 1.
    def update(self, game):
        Flicker.update(self, game)
        if self._age == 2:
            for u in BASEDIRS:
                if game.random_generator.random() < self.spreadprob:
                    game.create_sprite(self.name, (self.lastrect.left + u[0] * self.lastrect.size[0],
                                                     self.lastrect.top + u[1] * self.lastrect.size[1]))

class SpriteProducer(VGDLSprite):
    """ Superclass for all sprites that may produce other sprites, of type 'stype'. """
    stype = None

class Portal(SpriteProducer):
    is_static = True
    color = BLUE

class SpawnPoint(SpriteProducer):
    prob = None
    total = None
    color = BLACK
    is_static = True
    def __init__(self, cooldown=1, prob=1, total=None, **kwargs):
        SpriteProducer.__init__(self, **kwargs)
        if prob:
            self.prob = prob
            self.is_stochastic = (prob > 0 and prob < 1)
        if cooldown:
            self.cooldown = cooldown
        if total:
            self.total = total
        self.counter = 0

    def update(self, game):
        if (game.time % self.cooldown == 0 and game.random_generator.random() < self.prob):
            game.create_sprite(self.stype, (self.rect.left, self.rect.top))
            self.counter += 1

        if self.total and self.counter >= self.total:
            killSprite(self, None, game)

class RandomNPC(VGDLSprite):
    """ Chooses randomly from all available actions each step. """
    speed = 1
    is_stochastic = True

    def update(self, game):
        VGDLSprite.update(self, game)
        self.physics.activeMovement(self, game.random_generator.choice(BASEDIRS))

class OrientedSprite(VGDLSprite):
    """ A sprite that maintains the current orientation. """
    draw_arrow = False
    orientation = RIGHT

    state_attributes = VGDLSprite.state_attributes + ['orientation']

    def _draw(self, game):
        """ With a triangle that shows the orientation. """
        VGDLSprite._draw(self, game)
        if self.draw_arrow:
            col = (self.color[0], 255 - self.color[1], self.color[2])
            pygame.draw.polygon(game.screen, col, triPoints(self.rect, unitVector(self.orientation)))

class Conveyor(OrientedSprite):
    """ A static object that used jointly with the 'conveySprite' interaction to move
    other sprites around."""
    is_static = True
    color = BLUE
    strength = 1
    draw_arrow = True

class Missile(OrientedSprite):
    """ A sprite that constantly moves in the same direction. """
    speed = 1

class OrientedFlicker(OrientedSprite, Flicker):
    """ Preserves directionality """
    draw_arrow = True
    speed = 0

class Walker(Missile):
    """ Keep moving in the current horizontal direction. If stopped, pick one randomly. """
    airsteering = False
    is_stochastic = True
    def update(self, game):
        if self.airsteering or self.lastdirection[0] == 0:
            if self.orientation[0] > 0:
                d = 1
            elif self.orientation[0] < 0:
                d = -1
            else:
                d = game.random_generator.choice([-1, 1])
            self.physics.activeMovement(self, (d, 0))
        Missile.update(self, game)

class WalkJumper(Walker):
    prob = 0.1
    strength = 10
    def update(self, game):
        if self.lastdirection[0] == 0:
            if self.prob < game.random_generator.random():
                self.physics.activeMovement(self, (0, -self.strength))
        Walker.update(self, game)


class RandomInertial(OrientedSprite, RandomNPC):
    physicstype = ContinuousPhysics

class RandomMissile(Missile):
    def __init__(self, **kwargs):
        Missile.__init__(self, orientation=game.random_generator.choice(BASEDIRS),
                         speed=game.random_generator.choice([0.1, 0.2, 0.4]), **kwargs)

class ErraticMissile(Missile):
    """ A missile that randomly changes direction from time to time.
    (with probability 'prob' per timestep). """
    def __init__(self, prob=0.1, **kwargs):
        Missile.__init__(self, orientation=game.random_generator.choice(BASEDIRS), **kwargs)
        self.prob = prob
        self.is_stochastic = (prob > 0 and prob < 1)

    def update(self, game):
        Missile.update(self, game)
        if game.random_generator.random() < self.prob:
            self.orientation = game.random_generator.choice(BASEDIRS)

class Bomber(SpawnPoint, Missile):
    color = ORANGE
    is_static = False
    def update(self, game):
        Missile.update(self, game)
        SpawnPoint.update(self, game)

class Chaser(RandomNPC):
    """ Pick an action that will move toward the closest sprite of the provided target type. """
    stype = None
    fleeing = False

    def _closestTargets(self, game):
        bestd = 1e100
        res = []
        for target in game.getSprites(self.stype):
            d = self.physics.distance(self.rect, target.rect)
            if d < bestd:
                bestd = d
                res = [target]
            elif d == bestd:
                res.append(target)
        return res

    def _movesToward(self, game, target):
        """ Find the canonical direction(s) which move toward
        the target. """
        res = []
        basedist = self.physics.distance(self.rect, target.rect)
        for a in BASEDIRS:
            r = self.rect.copy()
            r = r.move(a)
            newdist = self.physics.distance(r, target.rect)
            if self.fleeing and basedist < newdist:
                res.append(a)
            if not self.fleeing and basedist > newdist:
                res.append(a)
        return res

    def update(self, game):
        VGDLSprite.update(self, game)
        options = []
        for target in self._closestTargets(game):
            options.extend(self._movesToward(game, target))
        if len(options) == 0:
            options = BASEDIRS
        self.physics.activeMovement(self, game.random_generator.choice(options))

class Fleeing(Chaser):
    """ Just reversing directions"""
    fleeing = True

class AStarChaser(RandomNPC):
    from .ai import AStarWorld
    """ Move towards the character using A* search. """
    stype = None
    fleeing = False
    drawpath = None
    walkableTiles = None
    neighborNodes = None

    def _movesToward(self, game, target):
        """ Find the canonical direction(s) which move toward
            the target. """
        res = []
        basedist = self.physics.distance(self.rect, target.rect)
        for a in BASEDIRS:
            r = self.rect.copy()
            r = r.move(a)
            newdist = self.physics.distance(r, target.rect)
            if self.fleeing and basedist < newdist:
                res.append(a)
            if not self.fleeing and basedist > newdist:
                res.append(a)
        return res

    def _draw(self, game):
        """ With a triangle that shows the orientation. """
        RandomNPC._draw(self, game)

        if self.walkableTiles:
            col = pygame.Color(0, 0, 255, 100)
            for sprite in self.walkableTiles:
                pygame.draw.rect(game.screen, col, sprite.rect)

        if self.neighborNodes:
            #logToFile("len(neighborNodes)=%s" %len(self.neighborNodes))
            col = pygame.Color(0, 255, 255, 80)
            for node in self.neighborNodes:
                pygame.draw.rect(game.screen, col, node.sprite.rect)

        if self.drawpath:
            col = pygame.Color(0, 255, 0, 120)
            for sprite in self.drawpath[1:-1]:
                pygame.draw.rect(game.screen, col, sprite.rect)

    def _setDebugVariables(self, world, path):
        '''
            Sets the variables required for debug drawing of the paths
            resulting from the A-Star search.
            '''

        path_sprites = [node.sprite for node in path]

        self.walkableTiles = world.get_walkable_tiles()
        self.neighborNodes = world.neighbor_nodes_of_sprite(self)
        self.drawpath = path_sprites

    def update(self, game):
        VGDLSprite.update(self, game)

        world = AStarWorld(game)
        path = world.getMoveFor(self)

        # Uncomment below to draw debug paths.
        # self._setDebugVariables(world,path)

        if len(path)>1:
            move = path[1]

            nextX, nextY = world.get_sprite_tile_position(move.sprite)
            nowX, nowY = world.get_sprite_tile_position(self)

            movement = None

            if nowX == nextX:
                if nextY > nowY:
                    #logToFile('DOWN')
                    movement = DOWN
                else:
                    #logToFile('UP')
                    movement = UP
            else:
                if nextX > nowX:
                    #logToFile('RIGHT')
                    movement = RIGHT
                else:
                    #logToFile('LEFT')
                    movement = LEFT

        self.physics.activeMovement(self, movement)


# ---------------------------------------------------------------------
#     Avatars: player-controlled sprite types
# ---------------------------------------------------------------------
from .core import Avatar

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
        active_keys = np.where(game.keystate)[0]
        active_keys = tuple(sorted(active_keys))

        # Up to 3 buttons active at a time, at least 0,
        for num_keys in range(max(3, len(active_keys)), -1, -1):

            # Find the longest key combo that matches a known action
            for key_combo in itertools.combinations(active_keys, num_keys):
                if key_combo in self.keys_to_action:
                    return self.keys_to_action[key_combo]


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
        actions["LEFT"] = K_LEFT
        actions["RIGHT"] = K_RIGHT
        actions["NO_OP"] = 0
        return actions


    def update(self, game):
        VGDLSprite.update(self, game)
        action = self._readAction(game)
        if action in [RIGHT, LEFT]:
            self.physics.activeMovement(self, action)

class VerticalAvatar(MovingAvatar):
    """ Only vertical moves.  """

    @classmethod
    def declare_possible_actions(cls):
        from pygame.locals import K_UP, K_DOWN
        actions = {}
        actions["UP"] = K_UP
        actions["DOWN"] = K_DOWN
        actions["NO_OP"] = 0
        return actions

    def update(self, game):
        VGDLSprite.update(self, game)
        action = self._readAction(game)
        if action in [UP, DOWN]:
            self.physics.activeMovement(self, action)

class FlakAvatar(HorizontalAvatar, SpriteProducer):
    """ Hitting the space button creates a sprite of the
    specified type at its location. """

    @classmethod
    def declare_possible_actions(cls):
        from pygame.locals import K_SPACE
        actions = super()
        actions["SPACE"] = K_SPACE
        return actions

    color = GREEN
    def update(self, game):
        HorizontalAvatar.update(self, game)
        self._shoot(game)

    def _shoot(self, game):
        from pygame.locals import K_SPACE
        if self.stype and game.keystate[K_SPACE]:
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
        actions["SPACE"] = K_SPACE
        return actions

    def __init__(self, stype=None, **kwargs):
        self.stype = stype
        OrientedSprite.__init__(self, **kwargs)

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
        if self.stype and game.keystate[K_SPACE]:
            u = unitVector(self.orientation)
            newones = game.create_sprite(self.stype, (self.lastrect.left + u[0] * self.lastrect.size[0],
                                                       self.lastrect.top + u[1] * self.lastrect.size[1]))
            if len(newones) > 0  and isinstance(newones[0], OrientedSprite):
                newones[0].orientation = unitVector(self.orientation)
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
    strength = 10
    airsteering = False

    # TODO seriously take this out of here
    passive_force = (0, 0)

    def update(self, game):
        from pygame.locals import K_SPACE

        action = self._readAction(game)
        # Force will just have a horizontal component as we do not allow up/down
        force = action.as_force()

        # You have to keep exerting force to keep moving
        horizontal_stopping_force = -self.velocity[0] / self.mass

        if K_SPACE in action.keys and self.passive_force[1] == 0:
            # Not airborne and attempting to jump
            force = (horizontal_stopping_force + force[0] * sqrt(self.strength), -self.strength)
        elif self.passive_force[1] != 0 and self.airsteering:
            # Airborne and actively steering, horizontal stopping force is less
            horizontal_stopping_force = - np.sign(self.velocity[0]) * np.sqrt(np.abs(self.velocity[0] / self.mass))
            force = (horizontal_stopping_force + force[0] * sqrt(self.strength), 0)
        elif self.passive_force[1] != 0 and not self.airsteering:
            # Airborne and not allowed to steer, so just let fly
            force = (0, 0)
        elif self.passive_force[1] == 0 and force[0]:
            # Actively walking along, you want the net velocity to be fixed
            force = (horizontal_stopping_force + force[0] * sqrt(self.strength), 0)
        else:
            # Not walking, should actively halt
            force = (horizontal_stopping_force, 0)

        self.physics.activeMovement(self, force)
        VGDLSprite.update(self, game)

    @classmethod
    def declare_possible_actions(cls):
        from pygame.locals import K_SPACE, K_RIGHT, K_LEFT
        actions = super().declare_possible_actions()
        actions['SPACE'] = Action(K_SPACE)
        actions['SPACE_RIGHT'] = Action(K_SPACE, K_RIGHT)
        actions['SPACE_LEFT'] = Action(K_SPACE, K_LEFT)
        return actions




# ---------------------------------------------------------------------
#     Termination criteria
# ---------------------------------------------------------------------
from .core import Termination

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


# ---------------------------------------------------------------------
#     Effect types (invoked after an event).
# ---------------------------------------------------------------------
def killSprite(sprite, partner, game):
    """ Kill command """
    game.kill_sprite(sprite)

def killBoth(sprite, partner, game):
    """ Kill command """
    game.kill_sprite(sprite)
    game.kill_sprite(partner)

def cloneSprite(sprite, partner, game):
    game.create_sprite(sprite.name, (sprite.rect.left, sprite.rect.top))

def transformTo(sprite, partner, game, stype='wall'):
    game.destroy_sprite(sprite)
    new_sprite = game.create_sprite(stype, sprite.rect.topleft, sprite.id)
    assert new_sprite, 'This really was not supposed to happen, cannot recover'
    if isinstance(sprite, OrientedSprite) and isinstance(new_sprite, OrientedSprite):
        new_sprite.orientation = sprite.orientation

def stepBack(sprite, partner, game):
    """ Revert last move. """
    sprite.rect = sprite.lastrect

def undoAll(sprite, partner, game):
    """ Revert last moves of all sprites. """
    for s in game:
        s.rect = s.lastrect

def bounceForward(sprite, partner, game):
    """ The partner sprite pushed, so if possible move in the opposite direction. """
    sprite.physics.activeMovement(sprite, unitVector(partner.lastdirection))
    game._updateCollisionDict(sprite)

def conveySprite(sprite, partner, game):
    """ Moves the partner in target direction by some step size. """
    tmp = sprite.lastrect
    v = unitVector(partner.orientation)
    sprite.physics.activeMovement(sprite, v, speed=partner.strength)
    sprite.lastrect = tmp
    game._updateCollisionDict(sprite)

def windGust(sprite, partner, game):
    """ Moves the partner in target direction by some step size, but stochastically
    (step, step-1 and step+1 are equally likely) """
    s = game.random_generator.choice([partner.strength, partner.strength + 1, partner.strength - 1])
    if s != 0:
        tmp = sprite.lastrect.copy()
        v = unitVector(partner.orientation)
        sprite.physics.activeMovement(sprite, v, speed=s)
        sprite.lastrect = tmp
        game._updateCollisionDict(sprite)

def slipForward(sprite, partner, game, prob=0.5):
    """ Slip forward in the direction of the current orientation, sometimes."""
    if prob > game.random_generator.random():
        tmp = sprite.lastrect
        v = unitVector(sprite.orientation)
        sprite.physics.activeMovement(sprite, v, speed=1)
        sprite.lastrect = tmp
        game._updateCollisionDict(sprite)

def attractGaze(sprite, partner, game, prob=0.5):
    """ Turn the orientation to the value given by the partner. """
    if prob > game.random_generator.random():
        sprite.orientation = partner.orientation

def turnAround(sprite, partner, game):
    sprite.rect = sprite.lastrect
    sprite.lastmove = sprite.cooldown
    sprite.physics.activeMovement(sprite, DOWN)
    sprite.lastmove = sprite.cooldown
    sprite.physics.activeMovement(sprite, DOWN)
    reverseDirection(sprite, partner, game)
    game._updateCollisionDict(sprite)

def reverseDirection(sprite, partner, game):
    sprite.orientation = (-sprite.orientation[0], -sprite.orientation[1])

def flipDirection(sprite, partner, game):
    sprite.orientation = game.random_generator.choice(BASEDIRS)

def bounceDirection(sprite, partner, game, friction=0):
    """ The centers of the objects determine the direction"""
    stepBack(sprite, partner, game)
    inc = sprite.orientation
    snorm = unitVector((-sprite.rect.centerx + partner.rect.centerx,
                        - sprite.rect.centery + partner.rect.centery))
    dp = snorm[0] * inc[0] + snorm[1] * inc[1]
    sprite.orientation = (-2 * dp * snorm[0] + inc[0], -2 * dp * snorm[1] + inc[1])
    sprite.speed *= (1. - friction)

def wallBounce(sprite, partner, game, friction=0):
    """ Bounce off orthogonally to the wall. """
    if not oncePerStep(sprite, game, 'lastbounce'):
        return
    sprite.speed *= (1. - friction)
    stepBack(sprite, partner, game)
    if abs(sprite.rect.centerx - partner.rect.centerx) > abs(sprite.rect.centery - partner.rect.centery):
        sprite.orientation = (-sprite.orientation[0], sprite.orientation[1])
    else:
        sprite.orientation = (sprite.orientation[0], -sprite.orientation[1])

def wallStop(sprite, partner, game, friction=0):
    """ Stop just in front of the wall, removing that velocity component,
    but possibly sliding along it. """
    if not oncePerStep(sprite, game, 'laststop'):
        return

    # We will revise the velocity used for the last movement
    old_delta = Vector2(sprite.rect.topleft) - Vector2(sprite.lastrect.topleft)

    # Horizontal collision
    if abs(sprite.rect.centerx - partner.rect.centerx) > abs(sprite.rect.centery - partner.rect.centery):
        # velocity = (0, sprite.velocity[1] * (1. - friction))
        if sprite.velocity[0] > 0:
            x_clip = partner.rect.left - sprite.rect.right
        else:
            x_clip = partner.rect.right - sprite.rect.left

        rescale = (old_delta.x + x_clip) / old_delta.x
        new_delta = old_delta * rescale

        sprite.passive_force = (0, sprite.passive_force[1])
        velocity = (0, sprite.velocity[1])
    else:
        # Downward motion, so downward collision
        if sprite.velocity[1] > 0:
            y_clip = partner.rect.top - sprite.rect.bottom
        else:
            y_clip = partner.rect.bottom - sprite.rect.top

        rescale = (old_delta.y + y_clip) / old_delta.y
        new_delta = old_delta * rescale

        # Counter-act passive movement that has been applied earlier
        sprite.passive_force = (sprite.passive_force[0], 0)
        velocity = (sprite.velocity[0], 0)

    sprite.rect = sprite.lastrect.move(new_delta)
    sprite.update_velocity(velocity)

def killIfSlow(sprite, partner, game, limitspeed=1):
    """ Take a decision based on relative speed. """
    if sprite.is_static:
        relspeed = partner.speed
    elif partner.is_static:
        relspeed = sprite.speed
    else:
        relSpeed = np.linalg.norm(sprite.velocity - partner.velocity)
    if relspeed < limitspeed:
        killSprite(sprite, partner, game)

def killIfFromAbove(sprite, partner, game):
    """ Kills the sprite, only if the other one is higher and moving down. """
    if (sprite.lastrect.top > partner.lastrect.top
        and partner.rect.top > partner.lastrect.top):
        killSprite(sprite, partner, game)

def killIfAlive(sprite, partner, game):
    """ Perform the killing action, only if no previous collision effect has removed the partner. """
    if partner not in game.kill_list:
        killSprite(sprite, partner, game)

def collectResource(sprite, partner, game):
    """ Adds/increments the resource type of sprite in partner """
    assert isinstance(sprite, Resource)
    r = sprite.resourceType
    partner.resources[r] = max(0, min(partner.resources[r]+sprite.value, game.resources_limits[r]))

def changeResource(sprite, partner, game, resource, value=1):
    """ Increments a specific resource type in sprite """
    sprite.resources[resource] = max(0, min(sprite.resources[resource]+value, game.resources_limits[resource]))

def spawnIfHasMore(sprite, partner, game, resource, stype, limit=1):
    """ If 'sprite' has more than a limit of the resource type given, it spawns a sprite of 'stype'. """
    if sprite.resources[resource] >= limit:
        game.create_sprite(stype, (sprite.rect.left, sprite.rect.top))

def killIfHasMore(sprite, partner, game, resource, limit=1):
    """ If 'sprite' has more than a limit of the resource type given, it dies. """
    if sprite.resources[resource] >= limit:
        killSprite(sprite, partner, game)

def killIfOtherHasMore(sprite, partner, game, resource, limit=1):
    """ If 'partner' has more than a limit of the resource type given, sprite dies. """
    if partner.resources[resource] >= limit:
        killSprite(sprite, partner, game)

def killIfHasLess(sprite, partner, game, resource, limit=1):
    """ If 'sprite' has less than a limit of the resource type given, it dies. """
    if sprite.resources[resource] <= limit:
        killSprite(sprite, partner, game)

def killIfOtherHasLess(sprite, partner, game, resource, limit=1):
    """ If 'partner' has less than a limit of the resource type given, sprite dies. """
    if partner.resources[resource] <= limit:
        killSprite(sprite, partner, game)

def wrapAround(sprite, partner, game, offset=0):
    """ Move to the edge of the screen in the direction the sprite is coming from.
    Plus possibly an offset. """
    if sprite.orientation[0] > 0:
        sprite.rect.left = offset * sprite.rect.size[1]
    elif sprite.orientation[0] < 0:
        sprite.rect.left = game.screensize[0] - sprite.rect.size[0] * (1 + offset)
    if sprite.orientation[1] > 0:
        sprite.rect.top = offset * sprite.rect.size[1]
    elif sprite.orientation[1] < 0:
        sprite.rect.top = game.screensize[1] - sprite.rect.size[1] * (1 + offset)
    sprite.lastmove = 0

def pullWithIt(sprite, partner, game):
    """ The partner sprite adds its movement to the sprite's. """
    if not oncePerStep(sprite, game, 'lastpull'):
        return
    tmp = sprite.lastrect
    v = unitVector(partner.lastdirection)
    sprite._updatePos(v, partner.speed * sprite.physics.gridsize[0])
    if isinstance(sprite.physics, ContinuousPhysics):
        sprite.speed = partner.speed
        sprite.orientation = partner.lastdirection
    sprite.lastrect = tmp

def teleportToExit(sprite, partner, game):
    e = game.random_generator.choice(game.sprite_groups[partner.stype])
    sprite.rect = e.rect
    sprite.lastmove = 0

# this allows us to determine whether the game has stochastic elements or not
stochastic_effects = [teleportToExit, windGust, slipForward, attractGaze, flipDirection]

# this allows is to determine which effects might kill a sprite
kill_effects = [killSprite, killIfSlow, transformTo, killIfOtherHasLess, killIfOtherHasMore, killIfHasMore, killIfHasLess,
                killIfFromAbove, killIfAlive]
