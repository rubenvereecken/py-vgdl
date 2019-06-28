import itertools
import numpy as np
import logging
logger = logging.getLogger(__name__)
from time import time
from typing import Union, List, Dict, Tuple


class PygletKeys:
    RETURN = 0xff00 + 13
    SPACE = 32
    ESCAPE = 0xff00 + 27

    LEFT = 0xff00 + 81
    UP = 0xff00 + 82
    RIGHT = 0xff00 + 83
    DOWN = 0xff00 + 84


KEYS_TO_ATARI_ACTION = {
        tuple(): 'NOOP',
        (PygletKeys.SPACE,): 'FIRE',
        (PygletKeys.RIGHT,): 'RIGHT', (PygletKeys.LEFT,): 'LEFT',
        (PygletKeys.DOWN,): 'DOWN', (PygletKeys.UP,): 'UP',
        (PygletKeys.UP, PygletKeys.RIGHT): 'UPRIGHT',
        (PygletKeys.UP, PygletKeys.LEFT): 'UPLEFT',
        (PygletKeys.DOWN, PygletKeys.RIGHT): 'DOWNRIGHT',
        (PygletKeys.DOWN, PygletKeys.LEFT): 'DOWNLEFT',
}

for k, v in list(KEYS_TO_ATARI_ACTION.items()):
    KEYS_TO_ATARI_ACTION[frozenset(k)] = v
    # Append FIRE to all the possible combinations
    if v != 'FIRE':
        KEYS_TO_ATARI_ACTION[frozenset(k + (PygletKeys.SPACE,))] = v + 'FIRE'


ATARI_ACTION_TO_KEYS = { v: k for k, v in KEYS_TO_ATARI_ACTION.items() }


class Controls:
    def __init__(self):
        self.action = 0
        self.restart = False
        self.pause = False
        self.debug = False


    def capture_key_presses(self):
        """
        There are two models of processing key presses, depending on framework.
        Pyglet uses event based, PyGame seems to favor manually checking what's pressed.
        """


    def on_key_press(self, key, mod):
        if key == self.return_key:
            self.restart = True
        elif key == self.pause_key:
            print('PAUSE?')
            self.pause = not self.pause
        elif key == self.debug_key:
            print('DEBUG')
            self.debug = not self.debug
        elif key in self.activated:
            self.activated[key] = True


    def on_key_release(self, key, mod):
        if key in self.activated:
            self.activated[key] = False


    @property
    def current_action(self):
        active_keys = [key for key, active in self.activated.items() if active]
        # Up to 3 buttons active at a time, at least 0,
        # so see if there is any known combination for an Atari action
        for num_keys in range(max(3, len(active_keys)), -1, -1):
            for key_combo in itertools.combinations(active_keys, num_keys):
                key_combo = frozenset(key_combo)
                if key_combo in self.keys_to_action:
                    return self.keys_to_action[key_combo]
        logger.debug('Combo not recognized:', active_keys)
        return self.perform_noop()


class AtariControls(Controls):
    def __init__(self, action_selection):
        super().__init__()
        # action_selection should be list of keys corresponding to AtariActions
        self.available_actions = dict(zip(action_selection, range(len(action_selection))))
        # Maps a key combo to a Gym action (an action index)
        self.keys_to_action = { ATARI_ACTION_TO_KEYS[name]: code for name, code in \
                self.available_actions.items()}

        # Dict to keep tracked of pressed keys; key code -> bool
        self.activated = { key: False for name, key in vars(PygletKeys).items() \
                if not name.startswith('__') or name in ['RETURN', 'ESCAPE'] }

        self.escape_key = PygletKeys.ESCAPE
        self.return_key = PygletKeys.RETURN


    def perform_noop(self):
        return self.available_actions['NOOP']


import pygame

# TODO do we really need these? Is this so we don't track too many?
class PygameKeys:
    RETURN = pygame.K_RETURN
    ESCAPE = pygame.K_ESCAPE
    PAUSE = pygame.K_PAUSE

    LEFT = pygame.K_LEFT
    UP = pygame.K_UP
    RIGHT = pygame.K_RIGHT
    DOWN = pygame.K_DOWN
    SPACE = pygame.K_SPACE


KEYS_TO_VGDL_ACTION = {
    # Note how this noop is spelled different from the Atari one
    tuple(): 'NO_OP',
    (pygame.K_SPACE,): 'SPACE',
    (pygame.K_RIGHT,): 'RIGHT', (pygame.K_LEFT,): 'LEFT',
    (pygame.K_DOWN,): 'DOWN', (pygame.K_UP,): 'UP',
    (pygame.K_SPACE, pygame.K_RIGHT): 'SPACE_RIGHT',
    (pygame.K_SPACE, pygame.K_LEFT): 'SPACE_LEFT',
}

def _expand_vgdl_keys():
    for num in range(ord('a'), ord('z')):
        char = chr(num)
        pygame_key = getattr(pygame, 'K_%s' % char)
        KEYS_TO_VGDL_ACTION[(pygame_key,)] = char
        setattr(PygameKeys, char, pygame_key)

_expand_vgdl_keys()

# for k, v in list(KEYS_TO_VGDL_ACTION.items()):
#     KEYS_TO_VGDL_ACTION[frozenset(k)] = v

# VGDL_ACTION_TO_KEYS = { v: k for k, v in KEYS_TO_VGDL_ACTION.items() }
VGDL_ACTION_TO_KEYS = { k: frozenset(k) for k in KEYS_TO_VGDL_ACTION.keys() }

class VGDLControls(Controls):
    def __init__(self, action_selection: List[str]):
        super().__init__()
        # action_selection should be list of keys corresponding to VGDL actions
        self.available_actions: Dict[str, int] \
            = dict(zip(action_selection, range(len(action_selection))))
        # Maps a key combo to a Gym action (an action index)
        self.keys_to_action: Dict[Tuple, int] = \
            { VGDL_ACTION_TO_KEYS[name]: code \
             for name, code in self.available_actions.items()}
        # print(self.available_actions)
        # print(self.keys_to_action)

        # Dict to keep tracked of pressed keys
        self.activated = { key: False for name, key in vars(PygameKeys).items() \
                if not name.startswith('__') or name in ['RETURN', 'ESCAPE', 'PAUSE'] }

        self.pause_key = pygame.K_PAUSE
        self.return_key = pygame.K_RETURN
        self.debug_key = pygame.K_ESCAPE
        self.escape_key = pygame.K_ESCAPE
        self.special_keys = [self.pause_key, self.escape_key, self.return_key, self.debug_key]

        # There is a serious need for throttling in Pygame that is not present in Pyglet
        # Make sure a special button can only be pressed once every `throttle` seconds
        self.last_special_press = 0
        self.special_throttle_time = .25


    def capture_key_presses(self):
        keys = pygame.key.get_pressed()
        # Need to flush events while paused, doesn't seem to harm vgdl.core events
        pygame.event.pump()

        for name, k in vars(PygameKeys).items():
            if name.startswith('__'): continue
            if k in self.special_keys and keys[k]:
                # Release a special key within throttle period
                # to avoid invoking it multiple times
                special_elapsed = time() - self.last_special_press
                # logger.debug(special_elapsed > self.special_throttle_time, special_elapsed)
                self.last_special_press = time()
                if special_elapsed < self.special_throttle_time:
                    self.on_key_release(k, None)
                    continue

            if not keys[k]:
                self.on_key_release(k, None)
            else:
                self.on_key_press(k, None)


    def perform_noop(self):
        return self.available_actions['NO_OP']


class ReplayVGDLControls(VGDLControls):
    def __init__(self, action_selection, replay_actions: List['Action']):
        super().__init__(action_selection)

        self.replay_actions = replay_actions
        self.action_idx = 0


    def capture_key_presses(self):
        # CAREFUL this is not idempotent

        # Capture special key presses, unset other ones
        super().capture_key_presses()
        # We don't want to go through the trace while paused
        if self.pause:
            return

        self.activated = {k: False for k in self.activated.keys()}

        if self.action_idx >= len(self.replay_actions):
            # Quit
            self.restart = True
            return

        current_action = self.replay_actions[self.action_idx]
        for key in current_action.keys:
            self.activated[key] = True
        self.action_idx += 1
