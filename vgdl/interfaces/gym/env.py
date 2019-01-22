import os
from collections import OrderedDict
import gym
from gym import spaces
import vgdl
from vgdl.state import StateObserver
import numpy as np
from .list_space import list_space


class VGDLEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 25
    }

    def __init__(self,
                 game_file = None,
                 level_file = None,
                 obs_type='image',
                 **kwargs):
        # For rendering purposes only
        self.render_block_size = kwargs.pop('block_size')

        # Variables
        self._obs_type = obs_type
        self.viewer = None
        self.game_args = kwargs
        self.notable_sprites = kwargs.get('notable_sprites', None)

        # Load game description and level description
        if game_file is not None:
            with open (game_file, "r") as myfile:
                game_desc = myfile.read()
            with open (level_file, "r") as myfile:
                level_desc = myfile.read()
            self.level_name = os.path.basename(level_file).split('.')[0]
            self.loadGame(game_desc, level_desc)


    def loadGame(self, game_desc, level_desc, **kwargs):

        self.game_desc = game_desc
        self.level_desc = level_desc
        self.game_args.update(kwargs)

        # Need to build a sample level to get the available actions and screensize....
        domain = vgdl.VGDLParser().parse_game(self.game_desc, **self.game_args)
        self.game = domain.build_level(self.level_desc)

        self.score_last = self.game.score

        # Set action space and observation space
        self._action_set = OrderedDict(self.game.get_possible_actions())
        self.action_space = spaces.Discrete(len(self._action_set))

        self.screen_width, self.screen_height = self.game.screensize

        if self._obs_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255,
                    shape=(self.screen_height, self.screen_width, 3) )
        elif self._obs_type == 'objects':
            from .state import NotableSpritesObserver
            self.observer = NotableSpritesObserver(self.game, self.notable_sprites)
            self.observation_space = list_space( spaces.Box(low=-100, high=100,
                    shape=self.observer.observation_shape) )
        elif self._obs_type == 'features':
            from .state import AvatarOrientedObserver
            self.observer = AvatarOrientedObserver(self.game)
            self.observation_space = spaces.Box(low=0, high=100,
                    shape=self.observer.observation_shape)
        elif isinstance(self._obs_type, type) and issubclass(self._obs_type, StateObserver):
            self.observer = self._obs_type(self.game)
            # TODO vgdl.StateObserver should report some space
            self.observation_space = spaces.Box(low=0, high=100,
                                        shape=self.observer.observation_shape)
        else:
            raise Exception('Unknown obs_type `{}`'.format(self._obs_type))

        # For rendering purposes, will be initialised by first `render` call
        self.renderer = None


    @property
    def _n_actions(self):
        return len(self._action_set)

    @property
    def _action_keys(self):
        return list(self._action_set.values())

    def get_action_meanings(self):
        # In the spirit of the Atari environment, describe actions with strings
        return list(self._action_set.keys())

    def _get_obs(self):
        if self._obs_type == 'image':
            return self.renderer.get_image()
        else:
            return self.observer.get_observation().as_array()

    def step(self, a):
        # if not self.mode_initialised:
        #     raise Exception('Please call `render` at least once for initialisation')
        self.game.tick(self._action_keys[a])
        state = self._get_obs()
        reward = self.game.score - self.score_last
        self.score_last = self.game.score
        terminal = self.game.ended
        return state, reward, terminal, {}

    def reset(self):
        # TODO improve the reset with the new domain split
        self.game.reset()
        # self.game = self.game.domain.build_level(self.level_desc)
        self.score_last = self.game.score
        state = self._get_obs()
        return state

    def render(self, mode='human', close=False):
        headless = mode != 'human'

        if self.renderer is None:
            from vgdl.render import PygameRenderer
            self.renderer = PygameRenderer(self.game, self.render_block_size)
            self.renderer.init_screen(headless)

        self.renderer.draw_all()
        self.renderer.update_display()

        if close:
            self.renderer.close()
        if mode == 'rgb_array':
            img = self.renderer.get_image()
            return img
        elif mode == 'human':
            return True

    def close(self):
        self.renderer.close()



class Padlist(gym.ObservationWrapper):
    def __init__(self, env=None, max_objs=200):
        self.max_objects = max_objs
        super(Padlist, self).__init__(env)
        env_shape = self.observation_space.shape
        env_shape[0] = self.max_objects
        self.observation_space = gym.spaces.Box(low=-100, high=100, shape=env_shape)

    def _observation(self, obs):
        return Padlist.process(obs, self.max_objects)

    @staticmethod
    def process(input_list, to_len):
        max_len = to_len
        item_len = len(input_list)
        if item_len < max_len:
          padded = np.pad(
              np.array(input_list,dtype=np.float32),
              ((0,max_len-item_len),(0,0)),
              mode='constant')
          return padded
        else:
          return np.array(input_list, dtype=np.float32)[:max_len]


