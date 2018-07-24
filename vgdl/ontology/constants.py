from pygame.math import Vector2
from vgdl.core import Action, Color

GREEN = Color((0, 200, 0))
BLUE = Color((0, 0, 200))
RED = Color((200, 0, 0))
GRAY = Color((90, 90, 90))
WHITE = Color((250, 250, 250))
BROWN = Color((140, 120, 100))
BLACK = Color((0, 0, 0))
ORANGE = Color((250, 160, 0))
YELLOW = Color((250, 250, 0))
PINK = Color((250, 200, 200))
GOLD = Color((250, 212, 0))
LIGHTRED = Color((250, 50, 50))
LIGHTORANGE = Color((250, 200, 100))
LIGHTBLUE = Color((50, 100, 250))
LIGHTGREEN = Color((50, 250, 50))
LIGHTGRAY = Color((150, 150, 150))
DARKGRAY = Color((30, 30, 30))
DARKBLUE = Color((20, 20, 100))

UP = Vector2(0, -1)
DOWN = Vector2(0, 1)
LEFT = Vector2(-1, 0)
RIGHT = Vector2(1, 0)

BASEDIRS = [UP, LEFT, DOWN, RIGHT]

NOOP = Action()
