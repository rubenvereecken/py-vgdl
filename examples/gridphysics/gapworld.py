import pygame
import os

import vgdl
import vgdl.mdp
from vgdl.ontology import MovingAvatar, RIGHT
import vgdl.interfaces
from vgdl.interfaces.pybrain import VGDLPybrainEnvironment, VGDLPybrainTask
from vgdl.state import AbsoluteObserver


class RightMovingJumpingAvatar(MovingAvatar):
    """
    Only moves and jumps to the right
    """

    def declare_possible_actions(self):
        from pygame.locals import K_RIGHT, K_SPACE
        actions = {}
        actions["RIGHT"] = K_RIGHT
        actions["SPACE"] = K_SPACE
        actions["NO_OP"] = 0
        return actions


    def update(self, game):
        from vgdl.core import VGDLSprite
        from pygame.locals import K_SPACE

        VGDLSprite.update(self, game)

        if game.keystate[K_SPACE]:
            x = self.rect.x / game.block_size
            # Jump up to 2 far, but may be less if near end of corridor
            jump_size = min(2, game.width - x - 1)
            self.physics.activeMovement(self, RIGHT, jump_size)
        else:
            action = self._readAction(game)
            self.physics.activeMovement(self, action)


def load_gapworld_game_and_level():
    with open(os.path.join(os.path.dirname(__file__), 'gapworld.txt')) as f:
        gamefile = f.read()
    with open(os.path.join(os.path.dirname(__file__), 'gapworld_lvl0.txt')) as f:
        levelfile = f.read()
    game = vgdl.VGDLParser().parseGame(gamefile)
    game.buildLevel(levelfile)
    return game


def test_gapworld():
    # Register the avatar first
    vgdl.registry.register_class(RightMovingJumpingAvatar)
    game = load_gapworld_game_and_level()
    env = VGDLPybrainEnvironment(game, AbsoluteObserver(game))
    task = VGDLPybrainTask(env)
    mapper = vgdl.mdp.MDPConverter(task)
    T, R = mapper.convert_task_to_mdp()
    for action_i in range(T.shape[0]):
        print('Action {}:'.format(env.action_set[action_i]))
        print(T[action_i])
    print('Rewards:')
    print(R)
    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT


if __name__ == '__main__':
    test_gapworld()

