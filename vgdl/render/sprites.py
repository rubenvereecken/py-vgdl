import pygame

from pathlib import Path
import pkg_resources

sprites_root = Path(pkg_resources.resource_filename('vgdl', 'sprites'))

class SpriteLibrary:
    default_instance = None

    def __init__(self, sprites_path):
        self.sprites_path = Path(sprites_path)
        if not self.sprites_path.exists():
            raise Exception(f'{sprites_path} does not exist')

        self.cache = {}


    def sprite_path(self, name):
        stem = Path(name).with_suffix('.png')
        sprite_path = self.sprites_path.joinpath(stem)
        return sprite_path


    def get_sprite(self, name):
        if name not in self.cache:
            path = self.sprite_path(name)
            img = pygame.image.load(str(path))
            self.cache[name] = img

        return self.cache[name]


    def load_all(self):
        # Mostly for debug purposes, don't use this often
        names = [s.relative_to(sprites_root) for s in self.sprites_path.glob('**/*.png')]

        for name in names:
            self.get_sprite(name)


    @classmethod
    def default(cls):
        # Singleton gets instantiated on first call
        if cls.default_instance is None:
            cls.default_instance = SpriteLibrary(sprites_root)
        return cls.default_instance

