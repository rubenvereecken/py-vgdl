'''
Video game description language -- parser, framework and core game classes.

@author: Tom Schaul
'''

import pygame
import random
from .tools import Node, indentTreeParser
from collections import defaultdict
from .tools import roundedPoints
import math
import os
import sys

VGDL_GLOBAL_IMG_LIB = {}

class BasicGame:
    """
    Heavily integrated with pygame for both game mechanics
    and visualisation both.

    This regroups all the components of a game's dynamics, after parsing.
    """
    MAX_SPRITES = 10000

    default_mapping = {'w': ['wall'],
                       'A': ['avatar'],
                       }

    seed = 123
    block_size = 10
    frame_rate = 25
    render_sprites = True
    load_save_enabled = False

    notable_sprites = []
    notable_resources = []

    def __init__(self, **kwargs):
        from .ontology import Immovable, DARKGRAY, MovingAvatar, GOLD
        for name, value in kwargs.items():
            if hasattr(self, name):
                self.__dict__[name] = value
            else:
                print("WARNING: undefined parameter '%s' for game! "%(name))

        # contains mappings to constructor (just a few defaults are known)
        self.sprite_constr = {'wall': (Immovable, {'color': DARKGRAY}, ['wall']),
                              'avatar': (MovingAvatar, {}, ['avatar']),
                              }
        # z-level of sprite types (in case of overlap)
        self.sprite_order  = ['wall',
                              'avatar',
                              ]
        # contains instance lists
        self.sprite_groups = defaultdict(list)
        # which sprite types (abstract or not) are singletons?
        self.singletons = []
        # collision effects (ordered by execution order)
        self.collision_eff = []
        # for reading levels
        self.char_mapping = {}
        # termination criteria
        self.terminations = [Termination()]
        # resource properties
        self.resources_limits = defaultdict(lambda: 2)
        self.resources_colors = defaultdict(lambda: GOLD)

        self.random_generator = random.Random(self.seed)
        self.is_stochastic = False
        self._lastsaved = None
        self.reset()

        if not self.notable_sprites:
            self.notable_sprites = self.sprite_groups


    def buildLevel(self, lstr):
        from .ontology import stochastic_effects
        lines = [l for l in lstr.split("\n") if len(l)>0]
        lengths = list(map(len, lines))
        assert min(lengths)==max(lengths), "Inconsistent line lengths."
        self.width = lengths[0]
        self.height = len(lines)
        # assert self.width > 1 and self.height > 1, "Level too small."

        self.screensize = (self.width*self.block_size, self.height*self.block_size)

        # set up resources
        for res_type, (sclass, args, _) in self.sprite_constr.items():
            if issubclass(sclass, Resource):
                if 'res_type' in args:
                    res_type = args['res_type']
                if 'color' in args:
                    self.resources_colors[res_type] = args['color']
                if 'limit' in args:
                    self.resources_limits[res_type] = args['limit']

        # create sprites
        for row, l in enumerate(lines):
            for col, c in enumerate(l):
                if c in self.char_mapping:
                    pos = (col*self.block_size, row*self.block_size)
                    self._createSprite(self.char_mapping[c], pos)
                elif c in self.default_mapping:
                    pos = (col*self.block_size, row*self.block_size)
                    self._createSprite(self.default_mapping[c], pos)
        self.kill_list=[]
        for _, _, effect, _ in self.collision_eff:
            if effect in stochastic_effects:
                self.is_stochastic = True

        # guarantee that avatar is always visible
        self.sprite_order.remove('avatar')
        self.sprite_order.append('avatar')


    def initScreen(self, headless, zoom=1, title=None):
        self.headless = headless
        self.zoom = zoom
        self.display_size = (self.screensize[0] * zoom, self.screensize[1] * zoom)

        # The screen surface will be used for drawing on
        # It will be displayed on the `display` surface, possibly magnified
        # The background is currently solely used for clearing away sprites
        self.background = pygame.Surface(self.screensize)
        if headless:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            self.screen = pygame.display.set_mode((1,1))
            self.display = None
        else:
            self.screen = pygame.Surface(self.screensize)
            self.screen.fill((0,0,0))
            self.background = self.screen.copy()
            self.display = pygame.display.set_mode(self.display_size, pygame.RESIZABLE, 32)
            if title:
                pygame.display.set_caption(title)
            # TODO there will probably be need for a separate background surface
            # once dirty optimisation is back in


    def reset(self):
        self.score = 0
        self.time = 0
        self.ended = False
        self.sprite_groups = defaultdict(list)
        self.num_sprites = 0
        # TODO should use kill_list more consistently
        # Is there a case where it makes sense to have a sprite in kill_list
        # without having cleared it right away?
        self.kill_list=[]
        #self.random_generator = random.Random(self.seed)


    # Returns a list of empty grid cells
    def emptyBlocks(self):
        alls = [s for s in self]
        res = []
        for col in range(self.width):
            for row in range(self.height):
                r = pygame.Rect((col*self.block_size, row*self.block_size),
                                (self.block_size, self.block_size))
                free = True
                for s in alls:
                    if r.colliderect(s.rect):
                        free = False
                        break
                if free:
                    res.append((col*self.block_size, row*self.block_size))
        return res


    def randomizeAvatar(self):
        if len(self.getAvatars()) == 0:
            self._createSprite(['avatar'], self.random_generator.choice(self.emptyBlocks()))


    def _createSprite(self, keys, pos):
        res = []
        for key in keys:
            if self.num_sprites > self.MAX_SPRITES:
                print("Sprite limit reached.")
                return
            sclass, args, stypes = self.sprite_constr[key]
            # verify the singleton condition
            anyother = False
            for pk in stypes[::-1]:
                if pk in self.singletons:
                    if self.numSprites(pk) > 0:
                        anyother = True
                        break
            if anyother:
                continue
            s = sclass(pos=pos, size=(self.block_size, self.block_size),
                name=key, random_generator=self.random_generator, **args)
            s.stypes = stypes
            self.sprite_groups[key].append(s)
            self.num_sprites += 1
            if s.is_stochastic:
                self.is_stochastic = True
            res.append(s)
        return res


    def __iter__(self):
        """ Iterator over all sprites (ordered) """
        for key in self.sprite_order:
            if key not in self.sprite_groups:
                # abstract type
                continue
            for s in self.sprite_groups[key]:
                yield s

    def numSprites(self, key):
        """ Abstract sprite groups are computed on demand only """
        deleted = len([s for s in self.kill_list if key in s.stypes])
        if key in self.sprite_groups:
            return len(self.sprite_groups[key])-deleted
        else:
            return len([s for s in self if key in s.stypes])-deleted

    def getSprites(self, key):
        if key in self.sprite_groups:
            return [s for s in self.sprite_groups[key] if s not in self.kill_list]
        else:
            # TODO I don't actually know where this would be used
            # This gets all sprites of a certain type
            return [s for s in self if key in s.stypes and s not in self.kill_list]

    def getAvatars(self):
        """ The currently alive avatar(s) """
        res = []
        for ss in self.sprite_groups.values():
            if ss and isinstance(ss[0], Avatar):
                res.extend([s for s in ss if s not in self.kill_list])
        return res


    # TODO: move or remove this, it's really part of ontology
    ignoredattributes = ['stypes',
                         'name',
                         'lastmove',
                         'color',
                         'lastrect',
                         'resources',
                         'physicstype',
                         'physics',
                         'rect',
                         'alternate_keys',
                         'res_type',
                         'stype',
                         'ammo',
                         'draw_arrow',
                         'shrink_factor',
                         'prob',
                         'is_stochastic',
                         'cooldown',
                         'total',
                         'is_static',
                         'noiseLevel',
                         'angle_diff',
                         'only_active',
                         'airsteering',
                         'strength',
                         'img',
                         'image',
                         'scale_image',
                         'randomtiling',
                         ]


    def __getstate__(self):
        assert len(self.kill_list) == 0
        objects = {}
        for sprite_type, sprites in self.sprite_groups.items():
            objects[sprite_type] = [sprite.__getstate__() for sprite in sprites]

        state = {
            'score': self.score,
            'ended': self.ended,
            'objects': objects,
        }
        return state


    def __setstate__(self):
        pass


    def getBoundingBoxes(self):
        boxes = []
        for s_types in list(self.notable_sprites):
            for s in self.getSprites(s_types):
                box = [ float(s.rect.x),
                        float(s.rect.y),
                        float(s.rect.w),
                        float(s.rect.h) ]
                boxes.append(box)
        return boxes

    # Returns gamestate in observation format
    def getObservation(self):
        #from .ontology import Avatar, Immovable, Missile, Portal, RandomNPC, ResourcePack
        state = []

        sprites_list = list(self.notable_sprites)
        num_classes = len(sprites_list)
        resources_list = self.notable_resources

        for i, key in enumerate(sprites_list):
            class_one_hot = [float(j==i) for j in range(num_classes)]
            for s in self.getSprites(key):
                position = [ float(s.rect.y)/self.block_size,
                             float(s.rect.x)/self.block_size ]
                if hasattr(s, 'orientation'):
                    orientation = [float(a) for a in s.orientation]
                else:
                    orientation = [0.0, 0.0]

                resources = [ float(s.resources[r]) for r in resources_list ]

                object_att = position + orientation + class_one_hot + resources

                state.append(object_att)
        return state

    def lenObservation(self):
        return 2 + 2 + (len(self.notable_sprites)) + len(self.notable_resources)

    def getFeatures(self):
        avatars = self.getAvatars()
        l = len(avatars)
        if l is not 0:
            a = avatars[0]
            avatar_pos = [float(a.rect.x)/self.block_size, float(a.rect.y)/self.block_size]
            resources = [float(a.resources[r]) for r in self.notable_resources]
            speed = [a.speed]
        else:
            avatar_pos = [.0, .0]
            resources = [.0 for r in self.notable_resources]
            speed = [.0]

        sprite_distances = []
        for key in self.notable_sprites:
            dist = 100
            if l is not 0:
              for s in self.getSprites(key):
                dist = min(self._getDistance(a, s)/self.block_size, dist)
            sprite_distances.append(dist)



        features = avatar_pos + speed + sprite_distances + resources
        return features

    def _getDistance(self, s1, s2):
        return math.hypot(s1.rect.x - s2.rect.x, s1.rect.y - s2.rect.y)

    def lenFeatures(self):
        return 2 + 1 + len(self.notable_sprites) + len(self.notable_resources)


    # Clears sprite from screen and removes dead sprites
    def _clearAll(self, onscreen=True):
        for s in set(self.kill_list):
            if onscreen:
                s._clear(self.screen, self.background, double=True)
            self.sprite_groups[s.name].remove(s)
        if onscreen:
            for s in self:
                s._clear(self.screen, self.background)
        self.kill_list = []

    def _drawAll(self):
        for s in self:
            s._draw(self)

    def _updateCollisionDict(self, changedsprite):
        for key in changedsprite.stypes:
            if key in self.lastcollisions:
                del self.lastcollisions[key]

    def _eventHandling(self):
        self.lastcollisions = {}
        ss = self.lastcollisions
        for g1, g2, effect, kwargs in self.collision_eff:
            # build the current sprite lists (if not yet available)
            for g in [g1, g2]:
                if g not in ss:
                    if g in self.sprite_groups:
                        tmp = self.sprite_groups[g]
                    else:
                        tmp = []
                        for key in self.sprite_groups:
                            v = self.sprite_groups[key]
                            if v and g in v[0].stypes:
                                tmp.extend(v)
                    ss[g] = (tmp, len(tmp))

            # special case for end-of-screen
            if g2 == "EOS":
                ss1, l1 = ss[g1]
                for s1 in ss1:
                    if not pygame.Rect((0,0), self.screensize).contains(s1.rect):
                        effect(s1, None, self, **kwargs)
                continue

            # iterate over the shorter one
            ss1, l1 = ss[g1]
            ss2, l2 = ss[g2]
            if l1 < l2:
                shortss, longss, switch = ss1, ss2, False
            else:
                shortss, longss, switch = ss2, ss1, True

            # score argument is not passed along to the effect function
            score = 0
            if 'scoreChange' in kwargs:
                kwargs = kwargs.copy()
                score = kwargs['scoreChange']
                del kwargs['scoreChange']

            # do collision detection
            for s1 in shortss:
                for ci in s1.rect.collidelistall(longss):
                    s2 = longss[ci]
                    if s1 == s2:
                        continue
                    # deal with the collision effects
                    if score:
                        self.score += score
                    if switch:
                        # CHECKME: this is not a bullet-proof way, but seems to work
                        if s2 not in self.kill_list:
                            effect(s2, s1, self, **kwargs)
                    else:
                        # CHECKME: this is not a bullet-proof way, but seems to work
                        if s1 not in self.kill_list:
                            effect(s1, s2, self, **kwargs)


    def getPossibleActions(self):
        return self.getAvatars()[0].declare_possible_actions()


    def tick(self, action, render = True):

        # This is required for game-updates to work properly
        self.time += 1

        # Flush events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Update Keypresses
        # Agents are updated during the update routine in their ontology files, this demends on BasicGame.keystate
        self.keystate = [0]* len(pygame.key.get_pressed())
        self.keystate[action] = 1

        # Iterate Over Termination Criteria
        for t in self.terminations:
            self.ended, win = t.isDone(self)
            if self.ended:
                break

        # Update Sprites
        for s in self:
            s.update(self)

        # Handle Collision Effects
        self._eventHandling()

        # Clean up dead sprites
        # NOTE(ruben): This used to be before event handling in original code
        self._clearAll()

        if not self.headless:
            self._drawAll()
            pygame.transform.scale(self.screen, self.display_size, self.display)
            pygame.display.update()
            # TODO once dirtyrects are back in, reset them here


class VGDLSprite:
    """ Base class for all sprite types. """
    name = None
    COLOR_DISC = [20,80,140,200]
    dirtyrects = []

    is_static= False
    only_active =False
    is_avatar= False
    is_stochastic = False
    color    = None
    cooldown = 0 # pause ticks in-between two moves
    speed    = None
    mass     = 1
    physicstype=None
    shrinkfactor=0

    def __init__(self, pos, size=(10,10), color=None, speed=None, cooldown=None, physicstype=None, random_generator=None, **kwargs):
        from .ontology import GridPhysics
        self.rect = pygame.Rect(pos, size)
        self.lastrect = self.rect
        self.physicstype = physicstype or self.physicstype or GridPhysics
        self.physics = self.physicstype()
        self.physics.gridsize = size
        self.speed = speed or self.speed
        self.cooldown = cooldown or self.cooldown
        self.img = 0
        self.color = color or self.color or (random_generator.choice(self.COLOR_DISC), random_generator.choice(self.COLOR_DISC), random_generator.choice(self.COLOR_DISC))

        for name, value in kwargs.items():
            try:
                self.__dict__[name] = value
            except:
                print("WARNING: undefined parameter '%s' for sprite '%s'! "%(name, self.__class__.__name__))
        # how many timesteps ago was the last move?
        self.lastmove = 0

        # management of resources contained in the sprite
        self.resources = defaultdict(lambda: 0)

        # TODO: Load images into a central dictionary to save loading a separate image for each object
        if self.img:
            if VGDL_GLOBAL_IMG_LIB.get(self.img) is None:
                import pkg_resources
                sprites_path = pkg_resources.resource_filename('vgdl', 'sprites')
                pth = os.path.join(sprites_path, self.img + '.png')
                img = pygame.image.load(pth)
                VGDL_GLOBAL_IMG_LIB[self.img] = img
            self.image = VGDL_GLOBAL_IMG_LIB[self.img]
            self.scale_image = pygame.transform.scale(self.image, (int(size[0] * (1-self.shrinkfactor)), int(size[1] * (1-self.shrinkfactor))))#.convert_alpha()


    def update(self, game):
        """ The main place where subclasses differ. """
        self.lastrect = self.rect
        # no need to redraw if nothing was updated
        self.lastmove += 1
        if not self.is_static and not self.only_active:
            self.physics.passiveMovement(self)

    def _updatePos(self, orientation, speed=None):
        if speed is None:
            speed = self.speed
        if not(self.cooldown > self.lastmove or abs(orientation[0])+abs(orientation[1])==0):
            self.rect = self.rect.move((orientation[0]*speed, orientation[1]*speed))
            self.lastmove = 0

    def _velocity(self):
        """ Current velocity vector. """
        if self.speed is None or self.speed==0 or not hasattr(self, 'orientation'):
            return (0,0)
        else:
            return (self.orientation[0]*self.speed, self.orientation[1]*self.speed)

    @property
    def lastdirection(self):
        return (self.rect[0]-self.lastrect[0], self.rect[1]-self.lastrect[1])


    def _draw(self, game):
        screen = game.screen
        if self.shrinkfactor != 0:
            shrunk = self.rect.inflate(-self.rect.width*self.shrinkfactor,
                                       -self.rect.height*self.shrinkfactor)
        else:
            shrunk = self.rect

        # uncomment for debugging
        #from .ontology import LIGHTGREEN
        #rounded = roundedPoints(self.rect)
        #pygame.draw.lines(screen, self.color, True, rounded, 2)

        if self.img and game.render_sprites:
            screen.blit(self.scale_image, shrunk)
        else:
            screen.fill(self.color, shrunk)
        if self.resources:
            self._drawResources(game, screen, shrunk)
        #r = self.rect.copy()
        #VGDLSprite.dirtyrects.append(r)

    def _drawResources(self, game, screen, rect):
        """ Draw progress bars on the bottom third of the sprite """
        from .ontology import BLACK
        tot = len(self.resources)
        barheight = rect.height/3.5/tot
        offset = rect.top+2*rect.height/3.
        for r in sorted(self.resources.keys()):
            wiggle = rect.width/10.
            prop = max(0,min(1,self.resources[r] / float(game.resources_limits[r])))
            if prop != 0:
                filled = pygame.Rect(rect.left+wiggle/2, offset, prop*(rect.width-wiggle), barheight)
                rest   = pygame.Rect(rect.left+wiggle/2+prop*(rect.width-wiggle), offset, (1-prop)*(rect.width-wiggle), barheight)
                screen.fill(game.resources_colors[r], filled)
                screen.fill(BLACK, rest)
                offset += barheight

    def _clear(self, screen, background, double=True):
        pass
        r = screen.blit(background, self.rect, self.rect)
        #VGDLSprite.dirtyrects.append(r)
        if double:
            r = screen.blit(background, self.lastrect, self.lastrect)
        #    VGDLSprite.dirtyrects.append(r)

    def __repr__(self):
        return self.name+" at (%s,%s)"%(self.rect.left, self.rect.top)


class Avatar:
    """ Abstract superclass of all avatars. """
    shrinkfactor=0.15

    def __init__(self):
        self.actions = self.declare_possible_actions()

class Resource(VGDLSprite):
    """ A special type of object that can be present in the game in two forms, either
    physically sitting around, or in the form of a counter inside another sprite. """
    value=1
    limit=2
    res_type = None

    @property
    def resourceType(self):
        if self.res_type is None:
            return self.name
        else:
            return self.res_type

class Termination:
    """ Base class for all termination criteria. """
    def isDone(self, game):
        """ returns whether the game is over, with a win/lose flag """
        from pygame.locals import K_ESCAPE, QUIT
        if game.keystate[K_ESCAPE] or pygame.event.peek(QUIT):
            return True, False
        else:
            return False, None
