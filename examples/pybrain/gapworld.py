import pygame
import os

import vgdl
import vgdl.mdp
from vgdl.ontology import MovingAvatar, RIGHT
import vgdl.interfaces
from vgdl.state import AbsoluteObserver, KeyValueObservation


class RightMovingJumpingAvatar(MovingAvatar):
    """
    Only moves and jumps to the right
    """

    @classmethod
    def declare_possible_actions(cls):
        # TODO port
        from vgdl.core import Action
        from pygame.locals import K_RIGHT, K_SPACE
        actions = {}
        actions["RIGHT"] = Action(K_RIGHT)
        actions["SPACE"] = Action(K_SPACE)
        actions["NO_OP"] = Action()
        return actions


    def update(self, game):
        from vgdl.core import VGDLSprite
        from pygame.locals import K_SPACE

        VGDLSprite.update(self, game)

        if K_SPACE in game.active_keys:
            x = self.rect.x / game.block_size
            # Jump up to 2 far, but may be less if near end of corridor
            jump_size = min(2, game.width - x - 1)
            self.physics.activeMovement(self, RIGHT, jump_size)
        else:
            action = self._readAction(game)
            self.physics.activeMovement(self, action)


class GapworldObserver(AbsoluteObserver):
    """
    All we need is a 1d x variable to represent state uniquely
    """
    def get_observation(self):
        avatar = self._game.getAvatars()[0]
        position = self._rect_to_pos(avatar.rect)
        obs = KeyValueObservation(x=position[0])
        return obs


def load_gapworld_game_and_level():
    with open(os.path.join(os.path.dirname(__file__), 'gapworld.txt')) as f:
        gamefile = f.read()
    with open(os.path.join(os.path.dirname(__file__), 'gapworld_lvl0.txt')) as f:
        levelfile = f.read()
    game = vgdl.VGDLParser().parseGame(gamefile)
    game.buildLevel(levelfile)
    return game


def test_gapworld():
    from vgdl.interfaces.pybrain import VGDLPybrainEnvironment, VGDLPybrainTask
    # Register the avatar first
    vgdl.registry.register_class(RightMovingJumpingAvatar)
    game = load_gapworld_game_and_level()
    env = VGDLPybrainEnvironment(game, GapworldObserver(game))
    task = VGDLPybrainTask(env)
    mapper = vgdl.mdp.MDPConverter(task)
    T, R = mapper.convert_task_to_mdp()
    print('Known states:')
    print(mapper.get_observations())
    for action_i in range(T.shape[0]):
        print('Action {}:'.format(env.action_set[action_i]))
        print(T[action_i])
    print('Rewards:')
    print(R)

    from pybrain.rl.learners.modelbased import policyIteration, trueValues
    # policy is S x A
    policy, optimal_T = policyIteration(T, R, discountFactor=.9)
    # So this seems wrong whether we allow transitions from absorbing states
    # or not, but it's a good indication
    V = trueValues(optimal_T, R, discountFactor=.9)

    print('Optimal policy:')
    print(policy)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    test_gapworld()

