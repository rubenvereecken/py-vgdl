import pygame


class PygameRenderer:
    def __init__(self, game, block_size):
        self.game = game
        # In pixels
        self.block_size = block_size
        self.screen_dims = (game.width * self.block_size, game.height * self.block_size)


    def init_screen(self, zoom, title=None):
        self.zoom = zoom
        self.display_dims = (self.screen_dims[0] * zoom, self.screen_dims[1] * zoom)

        # The screen surface will be used for drawing on
        # It will be displayed on the `display` surface, possibly magnified
        # The background is currently solely used for clearing away sprites
        # self.background = pygame.Surface(self.screensize)
        # if headless:
        #     os.environ['SDL_VIDEODRIVER'] = 'dummy'
        #     self.screen = pygame.display.set_mode((1,1))
        #     self.display = None
        # else:
        self.screen = pygame.Surface(self.screen_dims)
        self.screen.fill((0,0,0))
        self.background = self.screen.copy()
        self.display = pygame.display.set_mode(self.display_dims, pygame.RESIZABLE, 32)
        title_prefix = 'VGDL'
        title = title_prefix + ' ' + title if title else title_prefix
        if title:
            pygame.display.set_caption(title)


    def draw_all(self):
        # TODO could be better
        for s in self.game.sprite_registry.sprites():
            self.render_sprite(s)


    def update_display(self):
        pygame.transform.scale(self.screen, self.display_dims, self.display)
        pygame.display.update()


    def render_sprite(self, sprite):
        if sprite.shrinkfactor != 0:
            sprite_rect = sprite.rect.inflate(-sprite.rect.width*sprite.shrinkfactor,
                                       -sprite.rect.height*sprite.shrinkfactor)
        else:
            sprite_rect = sprite.rect

        # TODO sprites
        # if sprite.img and game.render_sprites:
        #     self.screen.blit(sprite.scale_image, sprite_rect)
        self.screen.fill(sprite.color, sprite_rect)
        # TODO resources


    def clear_sprite(self, sprite):
        # TODO if you get anything weird look at that 'double blitting'
        # TODO double
        self.screen.blit(self.background, sprite.rect, sprite.rect)


    def clear(self):
        # TODO properly draw background
        # self.screen.blit()
        self.screen.fill((0,0,0))


    def force_display(self):
        self.clear()
        self.draw_all()
        self.update_display()
