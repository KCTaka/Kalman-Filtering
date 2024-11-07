import numpy as np
import pygame
import threading
import time

from Observers import Observer
from Controller import Controller
from helper import normalize_angle

class CarEnvironment():
    def __init__(self,
                 controller: Controller,
                 window_size: tuple[int, int] = (800, 600),
                 initial_position: tuple[float, float] = (400, 300),
                 observers: list[Observer] = [],
                 ):
        """
        Initialize the car simulation environment
        
        Args:
            window_size: Tuple of (width, height) for the window
            initial_position: Starting position of the car
        """
                
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("2D Car Simulation")
        self.clock = pygame.time.Clock()
        
        # Car properties
        self.position = np.array(initial_position, dtype=float)
        self.velocity = np.array([0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0])
        self.angle = 0.0  # in radians
        self.car_size = (40, 20)  # width, height
        
        # Physics parameters
        self.acceleration_rate = 200.0 # pixel per second squared
        self.friction = 1          # Ns/m
        self.rotation_speed = 3.0  # radians per second
        
        # Controller
        if not controller:
            raise ValueError("Controller must be provided")
        self.controller = controller
        self.controller.start()
        
        # Observers
        self.observers = observers
        self.observers_time_tracker = {observer: time.time() for observer in self.observers}
        self.len_observers = len(self.observers)
        self.min_update_interval = min([observer.update_interval for observer in self.observers])/10 if observers else 1# Try to update the observer as much as possible
        
        # Control flags
        self.running = False
        self.update_thread = None
        
        # Initialize inputs
        self.angular_velocity_input = 0
        self.linear_acceleration_input = 0
        
    def _update_observers_amount(self):
        """ Update the amount of observers"""
        if self.len_observers != len(self.observers):
            self.observers_time_tracker = {observer: time.time() for observer in self.observers}
            self.len_observers = len(self.observers)
            self.min_update_interval = min([observer.update_interval for observer in self.observers])/10
            
    def _observer_publisher(self):
        """Continuously update observers at specified intervals"""
        start_time = time.time()
        while self.running:
            # Check if all attributes available
            has_all_attributes = all([hasattr(self, attr) for observer in self.observers for attr in observer.attributes])
            if not has_all_attributes:
                print(f"Waiting for all attributes to be available {time.time() - start_time:.2f}")
                continue
            
            self._update_observers_amount()
            for observer in self.observers:
                last_updated = self.observers_time_tracker[observer]
                time_interval = time.time() - last_updated
                if time_interval >= observer.update_interval:
                    self.observers_time_tracker[observer] = time.time()
                    attributes = observer.attributes
                    
                    measurements = []
                    for attr in attributes:
                        measurements.append(getattr(self, attr))
                    
                    observer.update(*measurements)
                    
            time.sleep(self.min_update_interval)
            
    
    def handle_input(self, dt: float):
        """Handle keyboard input and update car state"""
        ang_vel = self.controller.angle_velocity
        lin_acc = self.controller.acceleration
        
        # Rotation
        self.angle += ang_vel * dt
        self.angle = normalize_angle(self.angle)
            
        # Forward/Backward movement
        self.acceleration = np.array([lin_acc * np.cos(self.angle), lin_acc * np.sin(self.angle)])
    
    def update(self, dt: float):
        """Update car physics"""
        # Update velocity
        friction_force = -self.velocity * self.friction
        self.velocity += (self.acceleration + friction_force) * dt
            
        # Update position
        self.position += self.velocity * dt
        
        # Keep car within bounds
        self.position[0] = np.clip(self.position[0], 0, self.screen.get_width())
        self.position[1] = np.clip(self.position[1], 0, self.screen.get_height())
        
    def draw_car(self, position: tuple[int, int], angle: float, color: tuple[int, int, int], alpha: int = 255):
        """Draw car on the surface"""
        surface = pygame.Surface(self.car_size, pygame.SRCALPHA)
        surface.set_alpha(alpha)
        
        pygame.draw.rect(surface, color, (0, 0, *self.car_size))
        angle_deg = np.degrees(angle)
        rotated_surface = pygame.transform.rotate(surface, -angle_deg)
        position = (position[0] - rotated_surface.get_width() / 2, position[1] - rotated_surface.get_height() / 2)
        self.screen.blit(rotated_surface, position)
        
    def optional_render(self):
        pass
    
    def optional_value_update(self):
        pass
    
    def render(self):
        """Render the car and environment"""
        
        # Clear screen
        self.screen.fill((255, 255, 255))
        
        self.optional_render()
        
        # Draw car
        self.draw_car(self.position, self.angle, (255, 0, 0))

        # Update display
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        self.running = True
        
        # Start position printing thread
        self.update_thread = threading.Thread(target=self._observer_publisher)
        self.update_thread.start()
        
        try:
            while self.running:
                self.optional_value_update()
                
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    
                dt = self.clock.tick(60) / 1000.0  # Convert to seconds
                
                self.handle_input(dt)
                self.update(dt)
                self.render()
                
        finally:
            self.running = False
            if self.update_thread:
                self.update_thread.join()
            pygame.quit()
            
if __name__ == "__main__":
    controller = Controller(0.0001)
    env = CarEnvironment(controller=controller)
    env.run()