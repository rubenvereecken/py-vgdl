import itertools
import logging
from typing import NewType, Optional, Union, Dict, List, Tuple

import pygame
from pygame.math import Vector2

from vgdl.core import VGDLSprite, Resource
from vgdl.tools import oncePerStep


# __all__ = []


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
    from vgdl.ontology.sprites import OrientedSprite
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
    assert False, 'TODO'
    sprite.physics.activeMovement(sprite, unitVector(partner.lastdirection))
    game._updateCollisionDict(sprite)

def conveySprite(sprite, partner, game):
    """ Moves the partner in target direction by some step size. """
    tmp = sprite.lastrect
    assert False, 'TODO'
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
    if not oncePerStep(sprite, game, 't_lastbounce'):
        return
    sprite.speed *= (1. - friction)
    stepBack(sprite, partner, game)
    if abs(sprite.rect.centerx - partner.rect.centerx) > abs(sprite.rect.centery - partner.rect.centery):
        sprite.orientation = (-sprite.orientation[0], sprite.orientation[1])
    else:
        sprite.orientation = (sprite.orientation[0], -sprite.orientation[1])

def wallStop(sprite, partner, game, friction=0):
    """
    It is important both horizontal and vertical collisions are resolved.
    Vertical collisions keep gravity from building up.
    """
    # if not oncePerStep(sprite, game, 'laststop'):
    #     return

    # We will revise the velocity used for the last movement
    old_delta = Vector2(sprite.rect.topleft) - Vector2(sprite.lastrect.topleft)
    collision_vec = Vector2(partner.rect.center) - Vector2(sprite.rect.center)
    lastcollision_vec = Vector2(partner.rect.center) - Vector2(sprite.lastrect.center)

    # Probably a duplicate, because delta is never 0 with a collision
    if old_delta == Vector2(0,0):
        return

    same_vertical = partner.rect.left < sprite.rect.right < partner.rect.right or \
        partner.rect.left < sprite.rect.left < partner.rect.right

    # Horizontal collision
    # Assume you need horizontal velocity to effect a horizontal collision
    if abs(lastcollision_vec.x) > abs(lastcollision_vec.y):
        if not oncePerStep(sprite, game, 't_last_horizontal_stop'):
            return

        # velocity = (0, sprite.velocity[1] * (1. - friction))
        if sprite.velocity[0] > 0:
            x_clip = partner.rect.left - sprite.rect.right
        else:
            x_clip = partner.rect.right - sprite.rect.left

        # TODO clean up unused factors
        rescale = (old_delta.x + x_clip) / old_delta.x
        # new_delta = old_delta * rescale
        new_delta = old_delta + (x_clip, 0)

        sprite.passive_force = (0, sprite.passive_force[1])
        velocity = (0, sprite.velocity[1])
        y_clip = None
    else:
        if not oncePerStep(sprite, game, 't_last_vertical_stop'):
            return
        # Downward motion, so downward collision
        if sprite.velocity[1] > 0:
            y_clip = partner.rect.top - sprite.rect.bottom
        else:
            y_clip = partner.rect.bottom - sprite.rect.top

        rescale = (old_delta.y + y_clip) / old_delta.y
        # new_delta = old_delta.elementwise() * (1, rescale)
        # new_delta = old_delta * rescale
        new_delta = old_delta + (0, y_clip)

        # Counter-act passive movement that has been applied earlier
        sprite.passive_force = (sprite.passive_force[0], 0)
        velocity = (sprite.velocity[0], 0)
        # TODO
        x_clip = None

    sprite.rect = sprite.lastrect.move(new_delta)
    sprite.velocity = velocity


def killIfSlow(sprite, partner, game, limitspeed=1):
    """ Take a decision based on relative speed. """
    if sprite.is_static:
        relspeed = partner.speed
    elif partner.is_static:
        relspeed = sprite.speed
    else:
        relSpeed = (sprite.velocity - partner.velocity).length()
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
    if not oncePerStep(sprite, game, 't_lastpull'):
        return
    tmp = sprite.lastrect
    v = unitVector(partner.lastdirection)
    sprite._updatePos(v, partner.speed * sprite.physics.gridsize[0])
    if isinstance(sprite.physics, ContinuousPhysics):
        sprite.speed = partner.speed
        sprite.orientation = partner.lastdirection
    sprite.lastrect = tmp

def teleportToExit(sprite, partner, game):
    e = game.random_generator.choice(game.sprite_registry.group(partner.stype))
    sprite.rect = e.rect
    sprite.lastmove = 0
