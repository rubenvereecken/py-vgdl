from .constants import *
from .physics import *
from .sprites import *
from .avatars import *
from .terminations import *
from .effects import *

# this allows us to determine whether the game has stochastic elements or not
stochastic_effects = [windGust, slipForward, attractGaze, flipDirection]

# this allows is to determine which effects might kill a sprite
kill_effects = [killSprite, killIfSlow, transformTo, killIfOtherHasLess, killIfOtherHasMore, killIfHasMore, killIfHasLess,
                killIfFromAbove, killIfAlive]

