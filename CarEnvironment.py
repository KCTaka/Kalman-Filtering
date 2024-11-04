import numpy as np
import pygame
from typing import Callable, Optional
import threading
import time
from abc import ABC, abstractmethod

from KalmanFilter import KalmanFilter

def normalize_angle(angle: float):
    """Normalize angle to range [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

class Observer(ABC):
    """Abstract base class for landmark observers"""
    def __init__(self) -> None:
        self.update_interval = 1.0
    
    @abstractmethod
    def update(self, landmark_sensor):
        """Called when car position updates"""
        pass
    
class LandmarkSensor(Observer):
    """Observer that prints car position to console"""
    def __init__(self, update_interval) -> None:
        super().__init__()
        self.landmark_measurements = []
        self.update_interval = update_interval
        
    def print_landmark_measurements(self):
        print("---------------------------------")
        for i, (distance, angle) in enumerate(self.landmark_measurements):
            print(f"Landmark {i+1}: distance={distance:.2f}, angle={np.degrees(angle):.2f}°")
        print("---------------------------------")
    
    def update(self, landmark_measurements):
        self.landmark_measurements = landmark_measurements
        #self.print_landmark_measurements()

class CarEnvironment():
    def __init__(self, landmark_positions: list[tuple[int, int]],
                 window_size: tuple[int, int] = (800, 600),
                 initial_position: tuple[float, float] = (400, 300),
                 print_interval: float = 1.0,
                 observers: list[Observer] = []):
        """
        Initialize the car simulation environment
        
        Args:
            window_size: Tuple of (width, height) for the window
            initial_position: Starting position of the car
            print_interval: Time interval (in seconds) for printing car position
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
        
        # Landmark/Obstacle positions
        self.landmark_positions = [np.array(landmark_position) for landmark_position in landmark_positions]
        self.landmark_size = (5, 5)
        
        # Observers
        self.observers = observers
        
        # Control flags
        self.running = False
        self.print_interval = print_interval
        self.update_thread = None
        
        # Initialize inputs
        self.angular_velocity_input = 0
        self.linear_acceleration_input = 0
            
    def _update_observers(self):
        """Continuously update observers at specified intervals"""
        while self.running:
            max_update_interval = 0
            for observer in self.observers:
                landmark_measurements = self.get_landmark_measurements()
                observer.update(landmark_measurements)
                max_update_interval = max(max_update_interval, observer.update_interval)
                
            time.sleep(max_update_interval)
    
    def get_landmark_measurements(self):
        """Get landmark measurement"""
        
        # Calculate distances to landmarks
        distances = [np.linalg.norm(self.position - landmark) for landmark in self.landmark_positions]
        
        # Calculate the angle to each landmark
        angles = [normalize_angle(np.arctan2(landmark[1] - self.position[1], landmark[0] - self.position[0]) - self.angle) for landmark in self.landmark_positions]
        
        return list(zip(distances, angles))
    
    def handle_input(self, dt: float):
        """Handle keyboard input and update car state"""
        keys = pygame.key.get_pressed()
        
        self.angular_velocity_input = 0
        self.linear_acceleration_input = 0
        
        # Rotation
        if keys[pygame.K_LEFT]:
            self.angle -= self.rotation_speed * dt
            self.angle = normalize_angle(self.angle)
            self.angular_velocity_input = -self.rotation_speed
        if keys[pygame.K_RIGHT]:
            self.angle += self.rotation_speed * dt
            self.angle = normalize_angle(self.angle)
            self.angular_velocity_input = self.rotation_speed
            
        # Forward/Backward movement
        if keys[pygame.K_UP]:
            self.acceleration = np.array([
                np.cos(self.angle) * self.acceleration_rate,
                np.sin(self.angle) * self.acceleration_rate
            ])
            self.linear_acceleration_input = self.acceleration_rate
        elif keys[pygame.K_DOWN]:
            self.acceleration = np.array([
                -np.cos(self.angle) * self.acceleration_rate,
                -np.sin(self.angle) * self.acceleration_rate
            ])
            self.linear_acceleration_input = -self.acceleration_rate
        else:
            self.acceleration = np.array([0.0, 0.0])
    
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
    
    def render(self):
        """Render the car and environment"""
        # Clear screen
        self.screen.fill((255, 255, 255))
        
        # Create car surface
        car_surface = pygame.Surface(self.car_size, pygame.SRCALPHA)
        pygame.draw.rect(car_surface, (255, 0, 0), (0, 0, *self.car_size))
        
        # Rotate car surface
        rotated_surface = pygame.transform.rotate(car_surface, -np.degrees(self.angle))
        
        # Calculate position for rotated surface
        pos_x = self.position[0] - rotated_surface.get_width() // 2
        pos_y = self.position[1] - rotated_surface.get_height() // 2
        
        # Draw car
        self.screen.blit(rotated_surface, (pos_x, pos_y))
        
        # Create landmarks
        for landmark in self.landmark_positions:
            landmark_surface = pygame.Surface(self.landmark_size, pygame.SRCALPHA)
            pygame.draw.rect(landmark_surface, (0, 0, 255), (0, 0, *self.landmark_size))
            self.screen.blit(landmark_surface, landmark)
            
        if hasattr(self, 'kf_data') and len(self.kf_data) > 0:
            total_states = len(self.kf_data)
            for i, est_data in enumerate(self.kf_data):
                if i != total_states - 1:
                    continue

                est_position = est_data[:2]
                est_angle = est_data[4]
                
                # Calculate alpha value based on how old the state is
                # Newer states are more opaque, older states are more transparent
                alpha = max(30, int(255 * (i / total_states)))
                
                car_surface = pygame.Surface(self.car_size, pygame.SRCALPHA)
                pygame.draw.rect(car_surface, (0, 255, 0), (0, 0, *self.car_size))
                
                # Set the alpha value
                car_surface.set_alpha(alpha)
                
                rotated_surface = pygame.transform.rotate(car_surface, -np.degrees(est_angle))
                
                pos_x = est_position[0] - rotated_surface.get_width() / 2
                pos_y = est_position[1] - rotated_surface.get_height() / 2
                
                self.screen.blit(rotated_surface, (pos_x, pos_y))

        # Update display
        pygame.display.flip()
    
    def run(self, data=[[], []]):
        """Main game loop"""
        self.running = True
        
        # Start position printing thread
        self.update_thread = threading.Thread(target=self._update_observers)
        self.update_thread.start()
        
        self.kf_data = data
        
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
            if self.update_thread:
                self.update_thread.join()
            pygame.quit()
            
class CarEnvironmentWithNoise(CarEnvironment):
    def __init__(self, landmark_positions: list[tuple[int, int]],
                 window_size: tuple[int, int] = (800, 600),
                 initial_position: tuple[float, float] = (400, 300),
                 print_interval: float = 1.0,
                 observers: list[Observer] = [],
                 noise_std: float = 0.1):
        super().__init__(landmark_positions, window_size, initial_position, print_interval, observers)
        self.noise_std = noise_std
        
    def get_landmark_measurements(self):
        """Get landmark measurement with added noise"""
        measurements = super().get_landmark_measurements()
        return [(distance + np.random.normal(0, self.noise_std), angle + np.random.normal(0, self.noise_std)) for distance, angle in measurements]

class CarSim():
    def __init__(self, landmark_positions: list[tuple[int, int]],
                 window_size: tuple[int, int] = (800, 600),
                 initial_position: tuple[float, float] = (400, 300),
                 sensor_interval: float = 1.0,
                 noise_std: float = 0.1):
        
        self.x = [np.array([*initial_position, *(0, 0), 0])]
        self.P = [np.zeros_like(np.eye(self.x[-1].shape[0]))]
        
        self.landmark_sensor = LandmarkSensor(sensor_interval)
        
        self.car_env = CarEnvironmentWithNoise(landmark_positions, window_size, initial_position, sensor_interval, [self.landmark_sensor], noise_std)
        self.kf = KalmanFilter(get_A_k = self.get_A_k,
                                get_B_k = self.get_B_k,
                                get_D_k1 = self.get_D_k1,

                                f = self.f,
                                h = self.h,
                                )
        
        self.dt = sensor_interval
        self.landmark_positions = self.car_env.landmark_positions # Must use measured, not true landmark positions
    
    def catesian_to_polar(self, x):
        return np.linalg.norm(x), np.arctan2(x[1], x[0])
        
    def get_A_k(self, x_k1k: np.ndarray, u_k: np.ndarray) -> np.ndarray:
        A_k = np.array([[1, 0, self.dt, 0, 0],
                        [0, 1, 0, self.dt, 0],
                        [0, 0, (1-self.car_env.friction), 0, 0],
                        [0, 0, 0, (1-self.car_env.friction), 0],
                        [0, 0, 0, 0, 1]])
        return A_k
    
    def get_B_k(self, x_k1k: np.ndarray, u_k: np.ndarray) -> np.ndarray:
        B_k = np.array([[0, 0],
                        [0, 0],
                        [self.dt*np.cos(x_k1k[4]), 0],
                        [self.dt*np.sin(x_k1k[4]), 0],
                        [0, self.dt]])
        return B_k
    
    def get_D_k1(self, x_k1k: np.ndarray) -> np.ndarray: # Must correct to adjust to a calculated landpos
        D_k1 = []
        for landmark_position in self.landmark_positions:
            rho = np.linalg.norm(landmark_position - x_k1k[:2])
            rho_dot_x = -(landmark_position[0] - x_k1k[0])/rho 
            rho_dot_y = -(landmark_position[1] - x_k1k[1])/rho
            
            phi_dot_x = (landmark_position[1] - x_k1k[1])/(rho**2) 
            phi_dot_y = -(landmark_position[0] - x_k1k[0])/(rho**2)
            
            D_k1 += [[rho_dot_x, rho_dot_y, 0, 0, 0], [phi_dot_x, phi_dot_y, 0, 0, -1]]

        return np.array(D_k1)
    
    def f(self, x_k: np.ndarray, u_k: np.ndarray) -> np.ndarray:
        A = np.array([[1, 0, self.dt, 0, 0],
                [0, 1, 0, self.dt, 0],
                [0, 0, (1-self.car_env.friction), 0, 0],
                [0, 0, 0, (1-self.car_env.friction), 0],
                [0, 0, 0, 0, 1]])
        
        B = np.array([[0, 0],
                        [0, 0],
                        [self.dt*np.cos(x_k[4]), 0],
                        [self.dt*np.sin(x_k[4]), 0],
                        [0, self.dt]])
        
        x_k1 = A @ x_k + B @ u_k
        x_k1[4] = normalize_angle(x_k1[4])
        
        return x_k1
    
    def h(self, x_k1k: np.ndarray) -> np.ndarray:
        h = []
        for landmark_position in self.landmark_positions:
            rho = np.linalg.norm(landmark_position - x_k1k[:2])
            phi = normalize_angle(np.arctan2(landmark_position[1] - x_k1k[1], landmark_position[0] - x_k1k[0]) - x_k1k[4])
            h += [[rho, phi]]
        
        return np.array(h).flatten()
        
    def run(self):
        self.car_env.run(self.x)
        
    def simulate(self):
        Qk = np.array(0.1*np.eye(5))
        Rk1 = np.array(0.1*np.eye(len(self.landmark_positions)*2))
        Nk = np.array(0.1*np.eye(2))
        
        while True:
            x_k = self.x[-1]
            P_k = self.P[-1]
            
            input_angular_vel = self.car_env.angular_velocity_input
            input_linear_acc = self.car_env.linear_acceleration_input
            u_k = np.array([input_linear_acc, input_angular_vel])
            z_k1 = np.concatenate(self.landmark_sensor.landmark_measurements)
            
            x_k1, P_k1 = self.kf.update(x_k, u_k, z_k1, P_k, Qk, Rk1, Nk)
            
            self.x.append(x_k1)
            self.P.append(P_k1)
            
            # Set x to only have the last 10 elements
            if len(self.x) > 50:
                self.x.pop(0)
                self.P.pop(0)
                
            real_position = np.array(self.car_env.position)
            real_angle = self.car_env.angle
            
            error = np.linalg.norm(x_k1[:2] - real_position)
            print(f"Error: {error:.2f}")
            
            time.sleep(self.dt)
        
if __name__ == "__main__":
    # Generate random landmark positions
    landmark_positions = [(np.random.randint(0, 200), np.random.randint(0, 600)) for _ in range(5)]
    sim = CarSim(landmark_positions,
                    window_size=(800, 600),
                    initial_position=(400, 300),
                    sensor_interval=0.1,
                    noise_std=0.1)
    
    def delay_simulation():
        time.sleep(1)
        sim.simulate()
    
    simulation_thread = threading.Thread(target=delay_simulation, daemon=True)
    simulation_thread.start()
    
    sim.run()
    
    