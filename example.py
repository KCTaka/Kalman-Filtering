import numpy as np
import pygame
from typing import Callable, Optional, List
import threading
import time
from abc import ABC, abstractmethod

class CarObserver(ABC):
    """Abstract base class for car position observers"""
    @abstractmethod
    def update(self, position: np.ndarray, angle: float):
        """Called when car position updates"""
        pass

class PositionPrinter(CarObserver):
    """Observer that prints car position to console"""
    def update(self, position: np.ndarray, angle: float):
        print(f"Car position: ({position[0]:.2f}, {position[1]:.2f}), angle: {np.degrees(angle):.2f}°")

class PositionAccessor(CarObserver):
    """Observer that stores car position for external access"""
    def __init__(self):
        self.current_position = np.array([0.0, 0.0])
        self.current_angle = 0.0
        
    def update(self, position: np.ndarray, angle: float):
        self.current_position = position.copy()
        self.current_angle = angle
        
    def get_position(self) -> tuple[float, float]:
        """Get the last known position of the car"""
        return tuple(self.current_position)
    
    def get_angle(self) -> float:
        """Get the last known angle of the car"""
        return self.current_angle

class CarSim():
    def __init__(self, window_size: tuple[int, int] = (800, 600),
                 initial_position: tuple[float, float] = (400, 300),
                 update_interval: float = 1.0):
        """
        Initialize the car simulation environment
        
        Args:
            window_size: Tuple of (width, height) for the window
            initial_position: Starting position of the car
            update_interval: Time interval (in seconds) for updating observers
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
        self.max_speed = 300.0  # pixels per second
        self.acceleration_rate = 200.0
        self.friction = 0.95
        self.rotation_speed = 3.0  # radians per second
        
        # Observer pattern implementation
        self.observers: List[CarObserver] = []
        self.update_interval = update_interval
        self.observer_thread = None
        
        # Control flags
        self.running = False
        
    def add_observer(self, observer: CarObserver):
        """Add a new observer to the simulation"""
        self.observers.append(observer)
        
    def remove_observer(self, observer: CarObserver):
        """Remove an observer from the simulation"""
        if observer in self.observers:
            self.observers.remove(observer)
    
    def _notify_observers(self):
        """Continuously notify observers at specified intervals"""
        while self.running:
            for observer in self.observers:
                observer.update(self.position, self.angle)
            time.sleep(self.update_interval)
    
    def handle_input(self, dt: float):
        """Handle keyboard input and update car state"""
        keys = pygame.key.get_pressed()
        
        # Rotation
        if keys[pygame.K_LEFT]:
            self.angle += self.rotation_speed * dt
        if keys[pygame.K_RIGHT]:
            self.angle -= self.rotation_speed * dt
            
        # Forward/Backward movement
        if keys[pygame.K_UP]:
            self.acceleration = np.array([
                np.cos(self.angle) * self.acceleration_rate,
                np.sin(self.angle) * self.acceleration_rate
            ])
        elif keys[pygame.K_DOWN]:
            self.acceleration = np.array([
                -np.cos(self.angle) * self.acceleration_rate,
                -np.sin(self.angle) * self.acceleration_rate
            ])
        else:
            self.acceleration = np.array([0.0, 0.0])
    
    def update(self, dt: float):
        """Update car physics"""
        # Update velocity
        self.velocity += self.acceleration * dt
        self.velocity *= self.friction
        
        # Limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity * (self.max_speed / speed)
            
        # Update position
        self.position += self.velocity * dt
        
        # Keep car within bounds
        self.position[0] = np.clip(self.position[0], 0, self.screen.get_width())
        self.position[1] = np.clip(self.position[1], 0, self.screen.get_height())
    
    def render(self):
        """Render the car and environment"""
        # Clear screen
        self.screen.fill((255, 255, 255))
        
        # Create car surface
        car_surface = pygame.Surface(self.car_size, pygame.SRCALPHA)
        pygame.draw.rect(car_surface, (255, 0, 0), (0, 0, *self.car_size))
        
        # Rotate car surface
        rotated_surface = pygame.transform.rotate(car_surface, np.degrees(self.angle))
        
        # Calculate position for rotated surface
        pos_x = self.position[0] - rotated_surface.get_width() // 2
        pos_y = self.position[1] - rotated_surface.get_height() // 2
        
        # Draw car
        self.screen.blit(rotated_surface, (pos_x, pos_y))
        
        # Update display
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        self.running = True
        
        # Start observer notification thread
        self.observer_thread = threading.Thread(target=self._notify_observers)
        self.observer_thread.start()
        
        try:
            while self.running:
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
            if self.observer_thread:
                self.observer_thread.join()
            pygame.quit()


# Example usage
if __name__ == "__main__":
    # Create simulation instance
    sim = CarSim(update_interval=0.5)  # Update every 0.5 seconds
    
    # Create and add position printer
    # printer = PositionPrinter()
    # sim.add_observer(printer)
    
    # Create and add position accessor
    accessor = PositionAccessor()
    sim.add_observer(accessor)
    
    # Create a flag for the access thread
    access_thread_running = True
    
    # Example of how to access position from another thread
    def access_position():
        while access_thread_running:
            try:
                pos = accessor.get_position()
                angle = accessor.get_angle()
                print(f"External access - Position: {pos}, angle: {np.degrees(angle):.2f}°")
                time.sleep(1.0)  # Check position every second
            except Exception as e:
                print(f"Error accessing position: {e}")
    
    # Start position access thread
    access_thread = threading.Thread(target=access_position)
    access_thread.daemon = True  # Make thread daemon so it exits when main program exits
    access_thread.start()
    
    # Run simulation
    try:
        sim.run()
    finally:
        # Clean up
        access_thread_running = False
        time.sleep(1.1)  # Give access thread time to complete its last iteration