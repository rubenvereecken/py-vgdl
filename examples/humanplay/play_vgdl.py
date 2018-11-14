#!/usr/bin/env python

import os
import sys
import time
import itertools
import numpy as np
import argparse
import logging

import gym
from gym_recording.wrappers import TraceRecordingWrapper

import vgdl.interfaces.gym
from .human import HumanVGDLController

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def register_vgdl_env(domain_file, level_file, observer=None, blocksize=None):
    from gym.envs.registration import register, registry
    level_name = '.'.join(os.path.basename(level_file).split('.')[:-1])
    env_name = 'vgdl_{}-v0'.format(level_name)

    register(
        id=env_name,
        entry_point='vgdl.interfaces.gym:VGDLEnv',
        kwargs={
            'game_file': domain_file,
            'level_file': level_file,
            'block_size': blocksize or 24,
            'obs_type': observer or 'features',
        },
        timestep_limit=10000,
        nondeterministic=True
    )

    return env_name


def main():
    parser = argparse.ArgumentParser(
        description='Allows human play of VGDL domain and level files, ' + \
        'optionally loading additional ontology classes'
    )
    parser.add_argument('levelfile', type=str)
    parser.add_argument('--domainfile', '-d', type=str, default=None)
    parser.add_argument('--ontology', '-m', type=str, help="Python module for the game description")
    parser.add_argument('--observer', '-s', type=str,
                        help="A vgdl.StateObserver class, format pkg.module.Class")
    parser.add_argument('--reps', default=1, type=int)
    parser.add_argument('--blocksize', '-b', type=int)
    parser.add_argument('--tracedir', type=str)
    parser.add_argument('--pause_on_finish', dest='pause_on_finish',
                        action='store_true', default=False)

    args = parser.parse_args()

    if args.ontology is not None:
        import importlib
        module = importlib.import_module(args.ontology)
        import vgdl
        vgdl.registry.register_all(module)

    if args.observer is not None:
        import importlib
        name_bits = args.observer.split('.')
        module_name = '.'.join(name_bits[:-1])
        class_name = name_bits[-1]
        module = importlib.import_module(module_name)
        observer_cls = getattr(module, class_name)
    else:
        observer_cls = None

    if args.domainfile is None:
        # rely on naming convention to figure out domain file
        args.domainfile = os.path.join(os.path.dirname(args.levelfile),
                                       os.path.basename(args.levelfile).split('_')[0] + '.txt')

    env_name = register_vgdl_env(args.domainfile, args.levelfile, observer_cls,
                                 args.blocksize)
    # env_name = '.'.join(os.path.basename(args.levelfile).split('.')[:-1])

    logging.basicConfig(format='%(levelname)s:%(name)s %(message)s',
            level=logging.DEBUG)

    if args.tracedir is None:
        args.tracedir = os.path.join(THIS_DIR, '..', 'traces')
    args.tracedir = os.path.join(args.tracedir, env_name)
    os.makedirs(args.tracedir, exist_ok=True)

    controller_cls = HumanVGDLController
    controller = controller_cls(env_name, args.tracedir)

    for epoch_i in range(args.reps):
        window_open = controller.play(args.pause_on_finish)
        # Make sure it's reaaaally False
        if window_open is False:
            break

    controller.env.close()


if __name__ == '__main__':
    main()
