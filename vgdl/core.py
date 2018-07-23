'''
Video game description language -- parser, framework and core game classes.

@author: Tom Schaul
'''

import pygame
from pygame.math import Vector2
import random
from .tools import Node, indentTreeParser, PrettyDict
from collections import defaultdict, UserDict
from .tools import roundedPoints
import math
import numpy as np
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
        # Sprites are grouped by their primary stype, but they can have more
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

    def generate_id_number(self, key):
        count = len(self._sprites_by_key[key])
        return count + 1

    def generate_id(self, key):
        n = self.generate_id_number(key)
        return '{}.{}'.format(key, n)

    def create_sprite(self, key, id=None, **kwargs):
        # TODO fix rng
        sclass, args, stypes = self.get_sprite_def(key)
        id = id or self.generate_id(key)

        sprite = sclass(key=key, id=id, **{**args, **kwargs})
        sprite.stypes = stypes

        self._sprites_by_key[key].append(sprite)
        self._sprite_by_id[id] = sprite

        return sprite

    def kill_sprite(self, sprite: 'VGDLSprite'):
        """ Kills a sprite while keeping track of it """
        assert sprite is self._sprite_by_id[sprite.id], \
            'Unknown sprite %s' % sprite
        sprite.alive = False

    def destroy_sprite(self, sprite):
        """
        Destroys sprite based on id, erase all traces.
        Potentially useful when replacing a sprite
        """
        if sprite.id in self._sprite_by_id:
            del self._sprite_by_id[sprite.id]
            self._sprites_by_key[sprite.key] = [s for s in self._sprites_by_key[sprite.key] if not s.id == sprite.id]
            return True
        return False

    def groups(self, include_dead=False) -> Dict[str, List['VGDLSprite']]:
        assert isinstance(include_dead, bool), 'include_dead must be bool'
        if include_dead:
            return self._sprites_by_key
        else:
            return { k: [sprite for sprite in sprites if sprite.alive] \
                     for k, sprites in self._sprites_by_key.items()}

    def group(self, key, include_dead=False) -> List['VGDLSprite']:
        if include_dead:
            return self._sprites_by_key[key]
        else:
            return [ sprite for sprite in self._sprites_by_key[key] if sprite.alive ]

    def with_stype(self, stype):
        if stype in self.groups():
            return self.group(stype)
        else:
            return [s for sprites in self.groups().values() for s in sprites if stype in s.stypes]

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
# Action = NewType('Action', int)
Color = NewType('Color', Tuple[int, int, int])
Direction = NewType('Direction', Tuple[int, int])

class SpriteState(PrettyDict, UserDict):
    pass

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


class Action:
    """
    Actions are based on pygame keys, even though we do not actually
    use those keys. They're a handy leftover to vectorise actions
    and make thinking with them and programming game dynamics easier.
    """
    def __init__(self, *args):
        self.keys = tuple(sorted(args))

    def as_force(self):
        """
        Directional keys are used to encode directions.
        Opposite directions cancel eachother out.
        """
        from pygame.locals import K_LEFT, K_RIGHT, K_UP, K_DOWN
        return ( -1 * (K_LEFT in self.keys) + 1 * (K_RIGHT in self.keys),
                 -1 * (K_UP in self.keys) + 1 * (K_DOWN in self.keys) )


    def __repr__(self):
        import pygame.key
        _key_name = lambda k: pygame.key.name(k) if pygame.key.name(k) != 'unknown key' else k
        key_rep = ','.join(_key_name(k) for k in self.keys)
        return 'Action({})'.format(key_rep or 'noop')

    def __eq__(self, other):
        return self.keys == other.keys


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
    render_sprites = True

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

        # z-level of sprite types (in case of overlap), populated by parser
        self.sprite_order = []
        # which sprite types (abstract or not) are singletons? By parser
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
        self.kill_list.clear()

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


    def _resize_display(self, target_size):
        # Doesn't actually work on quite a few systems
        # https://github.com/pygame/pygame/issues/201
        w_factor = target_size[0] / self.display_size[0]
        h_factor = target_size[1] / self.display_size[1]
        factor = min(w_factor, h_factor)

        self.display_size = (int(self.display_size[0] * factor),
                             int(self.display_size[1] * factor))
        self.display = pygame.display.set_mode(self.display_size, pygame.RESIZABLE, 32)

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
        return sum(len(sprite_list) for sprite_list in self.sprite_registry.groups().values())


    def create_sprite(self, key, pos, id=None) -> Optional['VGDLSprite']:
        assert self.num_sprites < self.MAX_SPRITES, 'Sprite limit reached'

        sclass, args, stypes = self.sprite_registry.get_sprite_def(key)

        # TODO port this to registry
        anyother = any(self.numSprites(pk) > 0 for pk in stypes[::-1] if pk in self.singletons)
        if anyother:
            return None

        sprite = self.sprite_registry.create_sprite(key, pos=pos, id=id,
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

    def destroy_sprite(self, sprite):
        self.kill_list.append(sprite)
        self.sprite_registry.destroy_sprite(sprite)

    def __iter__(self):
        """ Iterator over all sprites (ordered) """
        for key in self.sprite_order:
            if key not in self.sprite_registry.groups():
                # abstract type
                continue
            for s in self.sprite_registry.groups()[key]:
                yield s

    def numSprites(self, key):
        """ Abstract groups are computed on demand only """
        deleted = len([s for s in self.kill_list if key in s.stypes])
        assert len(self.kill_list) == 0, 'Deprecated behaviour'
        if key in self.sprite_registry.groups():
            return len(self.sprite_registry.groups()[key])-deleted
        else:
            return len([s for s in self if key in s.stypes])-deleted

    def getSprites(self, key):
        assert len(self.kill_list) == 0, 'Deprecated behaviour'
        if key in self.sprite_registry.groups():
            return [s for s in self.sprite_registry.groups()[key] if s not in self.kill_list]
        else:
            # TODO I don't actually know where this would be used
            # This gets all sprites of a certain type
            return [s for s in self if key in s.stypes and s not in self.kill_list]

    def getAvatars(self):
        """ The currently alive avatar(s) """
        res = []
        assert len(self.kill_list) == 0, 'Deprecated behaviour'
        for ss in self.sprite_registry.groups(include_dead=True).values():
            if ss and isinstance(ss[0], Avatar):
                res.extend([s for s in ss if s not in self.kill_list])
        return res


    def __getstate__(self):
        assert len(self.kill_list) == 0, 'Deprecated behaviour'
        objects = {}
        for sprite_type, sprites in self.sprite_registry.groups().items():
            objects[sprite_type] = [sprite.__getstate__() for sprite in sprites]

        state = {
            'score': self.score,
            'ended': self.ended,
            'objects': objects,
        }
        return state


    def getGameState(self, include_random_state=False) -> GameState:
        assert len(self.kill_list) == 0, 'Kill list not empty'

        state = {
            'score': self.score,
            'time': self.time,
            'ended': self.ended,
            'sprites': self.sprite_registry.get_state(),
        }

        return GameState(state)


    def setGameState(self, state: GameState):
        """
        Rebuilds all sprites and resets game state
        TODO: Keep a sprite registry and even keep dead sprites around,
        just overwrite their state when setting game state.
        This has the advantage of keeping the Python objects intact.
        """
        state = copy.deepcopy(state)
        self.sprite_registry.set_state(state.pop('sprites'))
        for k, v in state.items():
            setattr(self, k, v)


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
        self.lastcollisions: Dict[str, Tuple['VGDLSprite',int]] = {}
        ss = self.lastcollisions
        for g1, g2, effect, kwargs in self.collision_eff:
            # build the current sprite lists (if not yet available)
            for g in [g1, g2]:
                if g not in ss:
                    sprites = self.sprite_registry.with_stype(g)
                    ss[g] = (sprites, len(sprites))

            # special case for end-of-screen
            if g2 == "EOS":
                ss1, l1 = ss[g1]
                for s1 in ss1:
                    if not pygame.Rect((0,0), self.screensize).contains(s1.rect):
                        effect(s1, None, self, **kwargs)
                continue

            # TODO care about efficiency again sometime, test short sprite list vs long
            # Can do this by sorting first?
            sprites, _ = ss[g1]
            others, _ = ss[g2]

            # score argument is not passed along to the effect function
            score = 0
            if 'scoreChange' in kwargs:
                kwargs = kwargs.copy()
                score = kwargs['scoreChange']
                del kwargs['scoreChange']

            for sprite in sprites:
                for collision_i in sprite.rect.collidelistall(others):
                    other = others[collision_i]

                    if sprite is other:
                        continue
                    elif sprite == other:
                        assert False, "Huh, interesting"

                    if score:
                        self.score += score
                    if sprite not in self.kill_list:
                        effect(sprite, other, self, **kwargs)


    def getPossibleActions(self) -> Dict[str, Action]:
        """ Assume actions don't change """
        from vgdl.core import Avatar
        avatar_cls = next(cls for cls in self.sprite_registry.classes.values() if issubclass(cls, Avatar))
        return avatar_cls.declare_possible_actions()


    def tick(self, action: Union[Action, int]):
        """
        Actions are currently communicated to the rest of the program
        through game.keystate, which mimics pygame.key.get_pressed().
        Maybe one beautiful day we can step away from that.
        It's a leftover but it works and it focuses on designing
        games that are human playable. Key presses are easy to reason about,
        even if we do not actually use them.
        """
        if isinstance(action, int):
            action = Action(action)
        assert action in self.getPossibleActions().values(), \
          'Illegal action %s, expected one of %s' % (action, self.getPossibleActions())
        if isinstance(action, int):
            action = Action(action)

        if self.ended:
            logging.warning('Action performed while game ended')
            return

        # This is required for game-updates to work properly
        self.time += 1

        # Flush events
        # Getting events like this keeps things rolling, otherwise use pygame.event.pump
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.VIDEORESIZE:
                self._resize_display(event.size)

        # Update Keypresses
        # Agents are updated during the update routine in their ontology files, this demends on BasicGame.keystate
        self.keystate = [0]* len(pygame.key.get_pressed())
        for key in action.keys:
            self.keystate[key] = 1

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
            self._update_display()
            # TODO once dirtyrects are back in, reset them here


    def _update_display(self):
        pygame.transform.scale(self.screen, self.display_size, self.display)
        pygame.display.update()


class VGDLSprite:
    """ Base class for all sprite types. """
    COLOR_DISC    = [20,80,140,200]

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

    def __init__(self, key, id, pos, size=(10,10), color=None, speed=None, cooldown=None, physicstype=None, random_generator=None, **kwargs):
        # Every sprite must have a key, an id, and a position
        self.key: str = key
        self.id: str  = id
        self.rect     = pygame.Rect(pos, size)
        self.lastrect = self.rect
        self.alive    = True

        from .ontology import GridPhysics
        self.physicstype      = physicstype or self.physicstype or GridPhysics
        self.physics          = self.physicstype()
        self.physics.gridsize = size
        self.speed            = speed or self.speed
        self.cooldown         = cooldown or self.cooldown
        self.img              = 0
        # TODO rng
        self.color = color or self.color
        # self.color            = color or self.color or (random_generator.choice(self.COLOR_DISC), random_generator.choice(self.COLOR_DISC), random_generator.choice(self.COLOR_DISC))

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
        # TODO use self.speed etc
        if speed is None:
            velocity = Vector2(orientation) * self.speed
        else:
            velocity = Vector2(self.orientation) * speed
        # if not(self.cooldown > self.lastmove or abs(orientation[0])+abs(orientation[1])==0):
        if not(self.cooldown > self.lastmove):
            # TODO use self.velocity
            self.rect = self.rect.move(velocity)
            self.lastmove = 0

    @property
    def velocity(self) -> 'Vector2':
        if self.speed is None or self.speed==0 or not hasattr(self, 'orientation'):
            return Vector2(0,0)
        else:
            return Vector2(self.orientation) * self.speed


    def update_velocity(self, v):
        assert len(v) == 2
        v = Vector2(v)
        self.speed = v.length()
        # Orientation is of unit length except when it isn't
        if self.speed == 0:
            self.orientation = Vector2(0,0)
        else:
            self.orientation = v.normalize()
        if any(np.isnan(self.orientation)):
            print("NAN ALERT")
            import ipdb; ipdb.set_trace()


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
        return "{} `{}` at ({}, {})".format(self.key, self.id, *self.rect.topleft)


class Avatar:
    """ Abstract superclass of all avatars. """
    shrinkfactor=0.15

    def __init__(self):
        assert false
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
            return self.key
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
