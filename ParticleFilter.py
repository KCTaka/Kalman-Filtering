import pygame

from CarEnvironment import CarEnvironment
from Observers import Observer

class NewCarEnv(CarEnvironment):
    def __init__(self):
        super().__init__()
        
    def optional_render(self):
        # Calculate and display FPS
        font = pygame.font.Font(None, 36)
        fps = self.clock.get_fps()
        fps_text = font.render(f"FPS: {fps:.2f}", True, pygame.Color('black'))
        self.screen.blit(fps_text, (10, 10))

if __name__ == "__main__":
    # Create simulation instance
    sim = NewCarEnv()
    sim.run()