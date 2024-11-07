import threading
import time
import pygame

class Controller():
    def __init__(self, update_interval: float) -> None:
        self.update_interval = update_interval
        self.angle_velocity = 0
        self.acceleration = 0
        self.running = True
        self.update_thread = threading.Thread(target=self.get_input)
        
    def start(self) -> None:
        self.update_thread.start()
    
    def get_input(self) -> None:
        """Get input from user"""
        while self.running:
            if not pygame.get_init():
                break
            
            keys = pygame.key.get_pressed()
            
            self.angle_velocity = 0
            self.acceleration = 0
            
            if keys[pygame.K_LEFT]:
                self.angle_velocity = -3
            if keys[pygame.K_RIGHT]:
                self.angle_velocity = 3
            if keys[pygame.K_UP]:
                self.acceleration = 200
            if keys[pygame.K_DOWN]:
                self.acceleration = -200
            
            time.sleep(self.update_interval)
    
    def stop(self) -> None:
        """Stop the input thread"""
        self.running = False
        self.update_thread.join()

# Example usage
if __name__ == "__main__":
    pygame.init()
    controller = Controller(0.1)
    # Run your main loop here
    # When done, stop the controller
    controller.stop()
    pygame.quit()