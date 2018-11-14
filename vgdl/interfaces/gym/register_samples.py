import gym
from gym.envs.registration import register
from vgdl.interfaces.gym import VGDLEnv
import os

# Location of sample games
import pkg_resources
games_path = pkg_resources.resource_filename('vgdl', 'games')

sample_games = [
    'aliens',
    'boulderdash',
    'chase',
    'frogs',
    'missilecommand',
    'portals',
    'survivezombies',
    'zelda' ]

# A list of relevant classes for each sample game
classes = {
    'aliens':         ['avatar', 'alien', 'base', 'bomb', 'sam'],
    'boulderdash':    ['avatar', 'boulder', 'butterfly', 'crab', 'diamond',
                       'exitdoor', 'wall'],
    'chase':          ['avatar', 'angry', 'carcass', 'scared', 'wall'],
    'frogs':          ['avatar', 'goal', 'log', 'truck', 'wall'],
    'missilecommand': ['avatar', 'city', 'explosion', 'incoming'],
    'portals':        ['avatar',  'goal', 'portalentry', 'portalexit', 'random',
                       'straight', 'wall' ],
    'survivezombies': ['avatar', 'bee', 'flower', 'hell', 'honey', 'zombie'],
    'zelda':          ['avatar', 'enemy', 'goal', 'key', 'wall']
}

# A list of relevant resources for each sample game
resources = {
    'aliens':         [],
    'boulderdash':    ['diamond'],
    'chase':          [],
    'frogs':          [],
    'missilecommand': [],
    'portals':        [],
    'survivezombies': ['honey'],
    'zelda':          []
}

suffixes = {
    'image':    "",
    'objects':  "_objects",
    'features': "_features",
}

# Register the sample games
def register_sample_games():
    try:
        for game in sample_games:
            for obs_type, suffix in suffixes.items():
                name='vgdl_{}{}-v0'.format(game, suffix)
                register(
                    id=name,
                    entry_point='vgdl.interfaces.gym:VGDLEnv',
                    kwargs={
                        'game_file': os.path.join(games_path, game + '.txt'),
                        'level_file': os.path.join(games_path, game + '_lvl0.txt'),
                        'obs_type': obs_type,
                        'notable_sprites': classes[game],
                        'notable_resources': resources[game],
                        # Use 24 (size of sprites) to render the full sprites
                        'block_size': 5 if obs_type == 'image' else 10
                    },
                    timestep_limit=1000,
                    nondeterministic=True,
                )
    except gym.error.Error as e:
        import logging
        logging.warning('Failed to register sample games, likely you are trying to import'
                        ' two versions of gym_vgdl')
