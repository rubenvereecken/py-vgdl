'''
Video game description language -- parser, framework and core game classes.

@author: Tom Schaul
'''

import pygame
import random
from .tools import Node, indentTreeParser
from collections import defaultdict, UserDict
from .tools import roundedPoints
import math
import os
import sys
import copy
import logging
from typing import NewType, Optional, Union, Dict, List, Tuple

VGDL_GLOBAL_IMG_LIB: Dict[str, str] = {}

class SpriteRegistry:
    """
    A registry that knows of all types of sprites from a game description,
    and all the sprite instances during the life of a game.

    TODO: may need to split this into separate class and instance registries.
    """
    def __init__(self):
        # Sprite classes, a class is identified by its key
        self.classes: Dict[str, type] = {}
        self.class_args: Dict[str, List] = {}
        self.stypes: Dict[str, List] = {}
        self.sprite_keys: List[str] = []

        # Sprite instances, each has a unique id
        self._sprites_by_key = defaultdict(list)
        self._sprite_by_id = {}

    def reset(self):
        self._sprites_by_key.clear()
        self._sprite_by_id.clear()

    def register_sprite_class(self, key, cls, args, stypes):
        assert not key in self.sprite_keys, 'Sprite key already registered'
        self.classes[key] = cls
        self.class_args[key] = args
        self.stypes[key] = stypes
        self.sprite_keys.append(key)

    def get_sprite_defs(self):
        for key in self.sprite_keys:
            yield key, self.get_sprite_def(key)

    def get_sprite_def(self, key):
        return self.classes[key], self.class_args[key], self.stypes[key]

    def request_id_number(self, key):
        count = len(self._sprites_by_key[key])
        return count + 1

    def request_id(self, key):
        n = self.request_id_number(key)
        return '{}.{}'.format(key, n)

    def create_sprite(self, key, id=None, **kwargs):
        # TODO fix rng
        sclass, args, stypes = self.get_sprite_def(key)
        id = id or self.request_id(key)

        sprite = sclass(key=key, id=id, **{**args, **kwargs})
        sprite.stypes = stypes

        self._sprites_by_key[key].append(sprite)
        self._sprite_by_id[id] = sprite

        return sprite

    def kill_sprite(self, sprite: 'VGDLSprite'):
        sprite.alive = False

    def sprite_groups(self, include_dead=False) -> Dict[str, List['VGDLSprite']]:
        if include_dead:
            return self._sprites_by_key
        else:
            return { k: [sprite for sprite in sprites if sprite.alive] \
                     for k, sprites in self._sprites_by_key.items()}

    def get_state(self) -> dict:
        def _sprite_state(sprite):
            return dict(
                id=sprite.id,
                state=sprite.getGameState()
            )

        sprite_states = {}
        for sprite_type, sprites in self._sprites_by_key.items():
            sprite_states[sprite_type] = [_sprite_state(sprite) for sprite in sprites]

        return sprite_states

    def set_state(self, state: dict):
        """
        - Overwrite the state of matching sprites (id-wise)
        - Remove sprites that are not in the new state
        - Add sprites that are in the new state

        Overwriting really is an unnecessary optimisation. All it gives us is
        last references to objects that do not get added, such as the avatar.
        """
        assert set(self.sprite_keys) == set(state.keys()), \
            'Known sprite keys should match'

        other_ids = set([sprite['id'] for sprites in state.values() for sprite in sprites])
        known_ids = set(self._sprite_by_id.keys())
        deleted_ids = known_ids.difference(other_ids)
        added_ids = other_ids.difference(known_ids)

        if len(deleted_ids) > 0:
            print('To be deleted')
            print(deleted_ids)
            for key in self.sprite_keys:
                self._sprites_by_key[key] = [sprite for sprite in self._sprites_by_key[key] \
                                             if not sprite.id in deleted_ids]

        if len(added_ids) > 0:
            print('To be added')
            print(added_ids)

        for key, sprite_states in state.items():
            for sprite_state in sprite_states:
                id = sprite_state['id']
                if id in self._sprite_by_id:
                    self._sprite_by_id[id].setGameState(sprite_state['state'])
                else:
                    # Including pos here because I don't like allowing position-less sprites
                    sprite = self.create_sprite(key, id, pos=sprite_state['state']['pos'])
                    sprite.setGameState(sprite_state['state'])



# Currently an action is a pygame.key press, an index into pygame.key.get_pressed()
# This may not fly anymore with actions that require multiple simultaneous key presses
Action = NewType('Action', int)
Color = NewType('Color', Tuple[int, int, int])
Direction = NewType('Direction', Tuple[int, int])

class SpriteState(UserDict):
    def __repr__(self):
        def _clean_value(v):
            if isinstance(v, defaultdict):
                return dict(v)
            return v
        attributes = ', '.join('{}={}'.format(k, _clean_value(v)) for k, v in self.data.items())
        return 'SpriteState({})'.format(attributes)

class GameState(UserDict):
    @property
    def avatar_state(self):
        return self.data['sprites']['avatar'][0]

    def ended(self):
        return self.data['ended']

    def __eq__(self, other):
        """ Game state equality, should ignore time etc """
        return self.data['sprites'] == other.data['sprites']

    def __hash__(self):
        """ Game state equality is based on sprite state """
        from .tools import freeze_dict
        return hash(freeze_dict(self.data['sprites']))

    def __lt__(self, other):
        # return self.data['time'] < other.data['time']
        return self.avatar_state['state']['pos'] < other.avatar_state['state']['pos']

    def __repr__(self):
        """ Assume single-avatar """
        avatar_state = self.avatar_state['state']
        return ('GameState(time={time}, score={score}, ended={ended}, '
               'avatar=(pos={pos}, alive={alive}))').format(**self.data, **avatar_state)


class BasicGame:
    """
    Heavily integrated with pygame for both game mechanics
    and visualisation both.

    This regroups all the components of a game's dynamics, after parsing.
    """
    MAX_SPRITES = 10000

    title = None
    seed = 123
    block_size = 10
    frame_rate = 25
    render_sprites = True
    load_save_enabled = False

    notable_resources: List[str] = []

    def __init__(self, sprite_registry, **kwargs):
        from .ontology import Immovable, DARKGRAY, MovingAvatar, GOLD
        for name, value in kwargs.items():
            if name in ['notable_resources', 'notable_sprites']:
                logging.warning('DEPRECATED BasicGame arg will be ignored: %s=%s', name, value)
            if hasattr(self, name):
                self.__dict__[name] = value
            else:
                print("WARNING: undefined parameter '%s' for game! "%(name))

        self.sprite_registry = sprite_registry

        # z-level of sprite types (in case of overlap)
        self.sprite_order  = ['wall',
                              'avatar',
                              ]
        # which sprite types (abstract or not) are singletons?
        self.singletons = []
        # used for erasing dead sprites
        self.kill_list = []
        # collision effects (ordered by execution order)
        self.collision_eff = []
        # for reading levels
        self.char_mapping = {}
        # termination criteria
        self.terminations = [Termination()]
        # resource properties, used to draw resource bar on the avatar sprite
        self.resources_limits = defaultdict(lambda: 2)
        self.resources_colors = defaultdict(lambda: GOLD)

        self.random_generator = random.Random(self.seed)
        self.is_stochastic = False
        self._lastsaved = None
        self.reset()


    def __repr__(self):
        if not self.title is None:
            return '{} `{}`'.format(self.__class__.__name__, self.title)
        else:
            return '{}'.format(self.__class__.__name__)


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
        self.notable_resources = []
        for res_type, (sclass, args, _) in self.sprite_registry.get_sprite_defs():
            if issubclass(sclass, Resource):
                if 'res_type' in args:
                    res_type = args['res_type']
                if 'color' in args:
                    self.resources_colors[res_type] = args['color']
                if 'limit' in args:
                    self.resources_limits[res_type] = args['limit']
                self.notable_resources.append(res_type)

        # create sprites
        for row, l in enumerate(lines):
            for col, c in enumerate(l):
                if c in self.char_mapping:
                    pos = (col*self.block_size, row*self.block_size)
                    self.create_sprites(self.char_mapping[c], pos)
        for _, _, effect, _ in self.collision_eff:
            if effect in stochastic_effects:
                self.is_stochastic = True

        # Used only for determining whether sprites should be erased
        self.kill_list=[]

        # guarantee that avatar is always visible
        self.sprite_order.remove('avatar')
        self.sprite_order.append('avatar')


    def initScreen(self, headless, zoom=5, title=None):
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
        self.kill_list.clear()
        self.sprite_registry.reset()
        # TODO rng?


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
            self.create_sprite('avatar', self.random_generator.choice(self.emptyBlocks()))


    @property
    def num_sprites(self):
        return sum(len(sprite_list) for sprite_list in self.sprite_registry.sprite_groups().values())


    def create_sprite(self, key, pos) -> Optional['VGDLSprite']:
        assert self.num_sprites < self.MAX_SPRITES, 'Sprite limit reached'

        sclass, args, stypes = self.sprite_registry.get_sprite_def(key)

        # TODO port this to registry
        anyother = any(self.numSprites(pk) > 0 for pk in stypes[::-1] if pk in self.singletons)
        if anyother:
            return None

        sprite = self.sprite_registry.create_sprite(key, pos=pos,
                                           size=(self.block_size, self.block_size),
                                           rng=self.random_generator)
        self.is_stochastic = self.is_stochastic or sprite.is_stochastic

        return sprite

    def create_sprites(self, keys, pos) -> List['VGDLSprite']:
        # Splitting it makes mypy happy
        filter_nones = lambda l: filter(lambda el: el, l)
        return list(filter_nones(self.create_sprite(key, pos) for key in keys))

    def kill_sprite(self, sprite):
        self.kill_list.append(sprite)
        self.sprite_registry.kill_sprite(sprite)


    def __iter__(self):
        """ Iterator over all sprites (ordered) """
        for key in self.sprite_order:
            if key not in self.sprite_registry.sprite_groups():
                # abstract type
                continue
            for s in self.sprite_registry.sprite_groups()[key]:
                yield s

    def numSprites(self, key):
        """ Abstract sprite groups are computed on demand only """
        deleted = len([s for s in self.kill_list if key in s.stypes])
        assert len(self.kill_list) == 0, 'Deprecated behaviour'
        if key in self.sprite_registry.sprite_groups():
            return len(self.sprite_registry.sprite_groups()[key])-deleted
        else:
            return len([s for s in self if key in s.stypes])-deleted

    def getSprites(self, key):
        assert len(self.kill_list) == 0, 'Deprecated behaviour'
        if key in self.sprite_registry.sprite_groups():
            return [s for s in self.sprite_registry.sprite_groups()[key] if s not in self.kill_list]
        else:
            # TODO I don't actually know where this would be used
            # This gets all sprites of a certain type
            return [s for s in self if key in s.stypes and s not in self.kill_list]

    def getAvatars(self):
        """ The currently alive avatar(s) """
        res = []
        assert len(self.kill_list) == 0, 'Deprecated behaviour'
        for ss in self.sprite_registry.sprite_groups(include_dead=True).values():
            if ss and isinstance(ss[0], Avatar):
                res.extend([s for s in ss if s not in self.kill_list])
        return res


    def __getstate__(self):
        assert len(self.kill_list) == 0, 'Deprecated behaviour'
        objects = {}
        for sprite_type, sprites in self.sprite_registry.sprite_groups().items():
            objects[sprite_type] = [sprite.__getstate__() for sprite in sprites]

        state = {
            'score': self.score,
            'ended': self.ended,
            'objects': objects,
        }
        return state


    def getGameState(self, include_random_state=False) -> GameState:
        assert len(self.kill_list) == 0, 'Kill list not empty'
        # sprite_states = {}

        # def _sprite_state(sprite):
        #     return dict(
        #         position=(sprite.rect.left, sprite.rect.top),
        #         state=sprite.getGameState()
        #     )

        # for sprite_type, sprites in self.sprite_registry.sprite_groups().items():
        #     sprite_states[sprite_type] = [_sprite_state(sprite) for sprite in sprites]

        state = {
            'score': self.score,
            'time': self.time,
            'ended': self.ended,
            'sprites': self.sprite_registry.get_state(),
        }
        assert 'sprites' in state
        return GameState(state)


    def setGameState(self, state: GameState):
        """
        Rebuilds all sprites and resets game state
        TODO: Keep a sprite registry and even keep dead sprites around,
        just overwrite their state when setting game state.
        This has the advantage of keeping the Python objects intact.
        """
        # for sprite_type, sprite_states in state['sprites'].items():
        #     # Discard all the other sprites
        #     self.sprite_groups[sprite_type] = []

        #     for sprite_state in sprite_states:
        #         sprites = self._createSprite(sprite_type, sprite_state['position'])
        #         sprite = sprites[0]; assert len(sprites) == 1, '... but how'
        #         sprite.setGameState(sprite_state['state'])

        state = copy.deepcopy(state)
        self.sprite_registry.set_state(state.pop('sprites'))
        for k, v in state.items():
            setattr(self, k, v)

    # Returns gamestate in observation format
    def getObservation(self):
        #from .ontology import Avatar, Immovable, Missile, Portal, RandomNPC, ResourcePack
        state = []

        notable_sprites = self.sprite_registry.sprite_groups()
        sprites_list = list(notable_sprites)
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
        return 2 + 2 + (len(notable_sprites)) + len(self.notable_resources)

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
        for key in self.sprite_registry.sprite_groups():
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
        return 2 + 1 + len(self.sprite_registry.sprite_groups()) + len(self.notable_resources)


    def _clearAll(self, onscreen=True):
        """ Clears dead sprites from screen """
        for s in set(self.kill_list):
            if onscreen:
                s._clear(self.screen, self.background, double=True)
        if onscreen:
            for s in self:
                s._clear(self.screen, self.background)
        self.kill_list.clear()

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
                    if g in self.sprite_registry.sprite_groups():
                        tmp = self.sprite_registry.sprite_groups()[g]
                    else:
                        tmp = []
                        for key, v in self.sprite_registry.sprite_groups().items():
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


    def getPossibleActions(self) -> Dict[str, Action]:
        avatar_cls, _, _ = self.sprite_registry.get_sprite_def('avatar')
        return avatar_cls.declare_possible_actions()


    def tick(self, action: Action):
        assert action in self.getPossibleActions().values(), \
          'Illegal action %s, expected one of %s' % (action, self.getPossibleActions())

        if self.ended:
            logging.warning('Action performed while game ended')
            return

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

        # Update Sprites
        for s in self:
            s.update(self)

        # Handle Collision Effects
        self._eventHandling()

        # Clean up dead sprites
        # NOTE(ruben): This used to be before event handling in original code
        self._clearAll()

        # Iterate Over Termination Criteria
        for t in self.terminations:
            self.ended, win = t.isDone(self)
            if self.ended:
                break

        if not self.headless:
            self._drawAll()
            pygame.transform.scale(self.screen, self.display_size, self.display)
            pygame.display.update()
            # TODO once dirtyrects are back in, reset them here


class VGDLSprite:
    """ Base class for all sprite types. """
    COLOR_DISC    = [20,80,140,200]

    key: str
    is_static     = False
    only_active   = False
    is_avatar     = False
    is_stochastic = False
    color         = None # type: Optional[Color]
    cooldown      = 0 # pause ticks in-between two moves
    speed         = None # type: Optional[int]
    mass          = 1
    physicstype   = None # type: type
    shrinkfactor  = 0.

    state_attributes = ['alive', 'resources', 'speed']

    def __init__(self, pos=None, size=(10,10), color=None, speed=None, cooldown=None, physicstype=None, random_generator=None, **kwargs):
        if pos is None:
            import ipdb; ipdb.set_trace()
        from .ontology import GridPhysics
        self.alive            = True
        self.rect             = pygame.Rect(pos, size)
        self.lastrect         = self.rect
        self.physicstype      = physicstype or self.physicstype or GridPhysics
        self.physics          = self.physicstype()
        self.physics.gridsize = size
        self.speed            = speed or self.speed
        self.cooldown         = cooldown or self.cooldown
        self.img              = 0
        self.color            = color or self.color or (random_generator.choice(self.COLOR_DISC), random_generator.choice(self.COLOR_DISC), random_generator.choice(self.COLOR_DISC))

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


    def getGameState(self) -> SpriteState:
        state = { attr_name: getattr(self, attr_name) for attr_name in self.state_attributes \
                 if hasattr(self, attr_name)}
        state['pos'] = self.rect.topleft
        # The alternative is to have each class define _just_ its state attrs,
        # flatten(c.state_attributes for c in inspect.getmro(self.__class__) \
        #   if c.hasattr('state_attributes'))
        return SpriteState(state)

    def setGameState(self, state: SpriteState):
        self.rect.topleft = state.pop('pos')

        for k, v in state.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                logging.warning('Unknown sprite state attribute `%s`', k)

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
        if double:
            r = screen.blit(background, self.lastrect, self.lastrect)

    def __repr__(self):
        return self.name+" at (%s,%s)"%(self.rect.left, self.rect.top)


class Avatar:
    """ Abstract superclass of all avatars. """
    shrinkfactor=0.15

    def __init__(self):
        self.actions = Avatar.declare_possible_actions()

class Resource(VGDLSprite):
    """ A special type of object that can be present in the game in two forms, either
    physically sitting around, or in the form of a counter inside another sprite. """
    value=1
    limit=2
    res_type = None

    state_attributes = VGDLSprite.state_attributes + ['limit']

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
