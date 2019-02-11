import sys
import time
import itertools
import numpy as np
import importlib
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

import gym


class HumanController:
    def __init__(self, env_name, trace_path=None, fps=15):
        self.env_name = env_name
        self.env = gym.make(env_name)
        if not env_name.startswith('vgdl'):
            logger.debug('Assuming Atari env, enable AtariObservationWrapper')
            from .wrappers import AtariObservationWrapper
            self.env = AtariObservationWrapper(self.env)
        # if trace_path is not None and importlib.util.find_spec('gym_recording') is not None:
        #     from gym_recording.wrappers import TraceRecordingWrapper
        #     self.env = TraceRecordingWrapper(self.env, trace_path)
        # elif trace_path is not None:
        #     logger.warn('trace_path provided but could not find the gym_recording package')
        if trace_path is not None:
            trace_path = Path(trace_path)
            trace_path.mkdir(parents=True, exist_ok=True)

        self.trace_path = trace_path
        self.fps = fps
        self.cum_reward = 0

        self.rewards = []
        self.actions = []
        self.observations = []


    def on_start(self):
        obs = self.env.unwrapped.observer.get_observation()
        self.observations.append(obs)


    def when_done(self):
        self.save_traces()


    def play(self, pause_on_finish=False):
        self.env.reset()

        for step_i in itertools.count():
            # Only does something for VGDL because Atari's Pyglet is event-based
            self.controls.capture_key_presses()

            obs, reward, done, info = self.env.step(self.controls.current_action)
            if reward:
                logger.debug("reward %0.3f" % reward)

            self.cum_reward += reward
            window_open = self.env.render()

            self.after_step(self.env.unwrapped.game.time)

            if not window_open:
                logger.debug('Window closed')
                return False

            if done:
                logger.debug('===> Done!')
                if pause_on_finish:
                    self.controls.pause = True
                    pause_on_finish = False
                else:
                    break

            if self.controls.restart:
                logger.info('Requested restart')
                self.controls.restart = False
                break

            if self.controls.debug:
                self.controls.debug = False
                self.debug()
                continue

            while self.controls.pause:
                self.controls.capture_key_presses()
                self.env.render()
                time.sleep(1. / self.fps)

            time.sleep(1. / self.fps)

        self.when_done()


    def debug(self, *args, **kwargs):
        # Convenience debug breakpoint
        env = self.env.unwrapped
        game = env.game
        observer = env.observer
        obs = env.observer.get_observation()
        sprites = game.sprite_registry
        state = game.get_game_state()
        all = dict(
            env=env, game=game, observer=observer,
            obs=obs, sprites=sprites, state=state
        )
        print(all)

        import ipdb; ipdb.set_trace()


    def after_step(self, step):
        env = self.env.unwrapped
        game = self.env.unwrapped.game
        action = env.unwrapped._action_keys[self.controls.current_action]
        reward = game.last_reward
        obs = env.observer.get_observation()
        self.rewards.append(reward)
        self.actions.append(action)
        self.observations.append(obs)


    def save_traces(self):
        if self.trace_path is None:
            return
        import pickle
        time_str = time.strftime("%d-%m-%Y-%H-%M-%S", time.gmtime())
        with self.trace_path.joinpath(time_str).with_suffix('.pkl').open('wb') as f:
            pickle.dump({
                'observations': self.observations,
                'rewards': self.rewards,
                'actions': self.actions,
            }, f)


class HumanAtariController(HumanController):
    def __init__(self, env_name, *args):
        super().__init__(env_name, *args)

        from .controls import AtariControls
        self.controls = AtariControls(self.env.unwrapped.get_action_meanings())

        # Render once to initialize the viewer
        self.env.render(mode='human')
        self.window = self.env.unwrapped.viewer.window
        self.window.on_key_press = self.controls.on_key_press
        self.window.on_key_release = self.controls.on_key_release



class HumanVGDLController(HumanController):
    def __init__(self, env_name, *args):
        super().__init__(env_name, *args)

        from .controls import VGDLControls
        self.controls = VGDLControls(self.env.unwrapped.get_action_meanings())
        self.env.render(mode='human')


class ReplayVGDLController(HumanController):
    def __init__(self, env_name, replay_actions, spy_func=None, *args, **kwargs):
        super().__init__(env_name, *args, **kwargs)
        self.replay_actions = replay_actions
        self.spy_func = spy_func

        from .controls import ReplayVGDLControls
        self.controls = ReplayVGDLControls(self.env.unwrapped.get_action_meanings(),
                                     replay_actions)
        self.env.render(mode='human')


    def after_step(self, step):
        if self.spy_func is not None:
            actual_action = self.env.unwrapped._action_keys[self.controls.current_action]
            self.spy_func(self.env.unwrapped, step, actual_action)


def determine_controller(env_name):
    if env_name.startswith('vgdl'):
        return HumanVGDLController
    else:
        return HumanAtariController
