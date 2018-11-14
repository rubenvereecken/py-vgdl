from gym.envs.registration import register
from .env import VGDLEnv
from .register_samples import register_sample_games

# Register samples
register_sample_games()

# register(
#     id='vgdl_generic-v0',
#     entry_point='gym_vgdl:VGDLEnv',
#     kwargs={
#         'block_size': 10
#     },
#     nondeterministic=True,
# )
