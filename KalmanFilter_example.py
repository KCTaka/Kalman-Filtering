import time
import threading
import numpy as np
import pygame

from KalmanFilter import KalmanFilter
from CarEnvironment import CarEnvironment
from Observers import Observer, LandmarkSensor, SpeedSensor
from Controller import Controller

from helper import normalize_angle

class KalmanFilter(KalmanFilter):
    def state_estimation(self, x_k, u_k, z_k1):
        x_k1k = self.f(x_k, u_k)
        z_k1k = self.h(x_k1k)
        s_k1 = z_k1 - z_k1k
        # Normalize the angle residual
        s_zeta = s_k1[:-1]
        s_zeta[1::2] = normalize_angle(s_zeta[1::2])
        s_k1[:-1] = s_zeta
        x_k1 = x_k1k + self.W_k1 @ s_k1
        return x_k1

class CarEnvironmentWithNoise(CarEnvironment):
    def __init__(self, landmark_positions: list[tuple[int, int]],
                 controller: Controller,
                 window_size: tuple[int, int] = (800, 600),
                 initial_position: tuple[float, float] = (400, 300),
                 observers: list[Observer] = [],
                 noise_std: float = 0.1):
        
        super().__init__(controller, window_size, initial_position, observers)
        self.noise_std = noise_std
        self.kf_data = None
        
        # Landmark/Obstacle positions
        self.landmark_positions = [np.array(landmark_position) for landmark_position in landmark_positions]
        self.landmark_size = (5, 5)
    
    def get_landmark_measurements(self):
        """Get landmark measurement"""
        
        # Calculate distances to landmarks
        distances = [np.linalg.norm(self.position - landmark) for landmark in self.landmark_positions]
        
        # Calculate the angle to each landmark
        angles = [normalize_angle(np.arctan2(landmark[1] - self.position[1], landmark[0] - self.position[0]) - self.angle) for landmark in self.landmark_positions]
        
        self.landmark_measurements = list(zip(distances, angles))
        self.landmark_measurements = [(distance + np.random.normal(0, self.noise_std), normalize_angle(angle + np.deg2rad(np.random.normal(0, self.noise_std)))) for distance, angle in self.landmark_measurements]
        
    def get_speed_measurement(self):
        """Get speed measurement"""
        self.speed = np.linalg.norm(self.velocity)
        self.speed += np.random.normal(0, self.noise_std)
        
    def optional_value_update(self):
        self.get_landmark_measurements()
        self.get_speed_measurement()
    
    def optional_render(self):
        # Add tracers for Kalman Filter data
        if hasattr(self, 'kf_data') and self.kf_data:
            total_states = len(self.kf_data)
            for i, est_data in enumerate(self.kf_data):
                if i != total_states - 1:
                    continue
                
                est_position = est_data[:2]
                est_angle = est_data[4]
                
                # Calculate alpha value based on how old the state is
                # Newer states are more opaque, older states are more transparent
                alpha = max(30, int(255 * (i / total_states)))
                
                # Draw ghost car
                self.draw_car(est_position, est_angle, (0, 255, 0), alpha)
                
                
        # Calculate and display FPS
        font = pygame.font.Font(None, 36)
        fps = self.clock.get_fps()
        fps_text = font.render(f"FPS: {fps:.2f}", True, pygame.Color('black'))
        self.screen.blit(fps_text, (10, 10))
        
        # Create landmarks
        for landmark_pos in self.landmark_positions:
            landmark_pos = (landmark_pos[0] - self.landmark_size[0]/2, landmark_pos[1] - self.landmark_size[1]/2)
            
            landmark_surface = pygame.Surface(self.landmark_size, pygame.SRCALPHA)
            pygame.draw.rect(landmark_surface, (0, 0, 255), (0, 0, *self.landmark_size))
            self.screen.blit(landmark_surface, landmark_pos)

class CarSim():
    def __init__(self, landmark_positions: list[tuple[int, int]],
                 controller: Controller,
                 window_size: tuple[int, int] = (800, 600),
                 initial_position: tuple[float, float] = (400, 300),
                 sensor_interval: float = 1.0,
                 noise_std: float = 0.1):
        
        self.x = [np.array([*initial_position, *(0, 0), 0],)]
        self.P = [np.zeros_like(np.eye(self.x[-1].shape[0],))]
        
        self.landmark_sensor = LandmarkSensor(sensor_interval)
        self.speed_sensor = SpeedSensor(sensor_interval)
        observers = [self.landmark_sensor, self.speed_sensor]
        
        self.car_env = CarEnvironmentWithNoise(landmark_positions, controller, window_size, initial_position, observers, noise_std)
        self.car_env.kf_data = self.x
        self.kf = KalmanFilter(get_A_k = self.get_A_k,
                                get_B_k = self.get_B_k,
                                get_D_k1 = self.get_D_k1,

                                f = self.f,
                                h = self.h,
                                )
        
        self.dt = sensor_interval
        self.landmark_positions = self.car_env.landmark_positions # Must use measured, not true landmark positions
        
        # Controller to see inputs:
        self.controller = controller
        
    def get_A_k(self, x_k1k: np.ndarray, u_k: np.ndarray) -> np.ndarray:
        A_k = np.array([[1, 0, self.dt, 0, 0],
                        [0, 1, 0, self.dt, 0],
                        [0, 0, (1-self.car_env.friction), 0, -self.dt*np.sin(x_k1k[4])*u_k[0]],
                        [0, 0, 0, (1-self.car_env.friction), self.dt*np.cos(x_k1k[4])*u_k[0]],
                        [0, 0, 0, 0, 1]],)
        return A_k
    
    def get_B_k(self, x_k1k: np.ndarray, u_k: np.ndarray) -> np.ndarray:
        B_k = np.array([[0, 0],
                        [0, 0],
                        [self.dt*np.cos(x_k1k[4]), 0],
                        [self.dt*np.sin(x_k1k[4]), 0],
                        [0, self.dt]],)
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
            
        speed_dot_x_dot = x_k1k[2]/(np.linalg.norm(x_k1k[2:4]) + 1e-6)
        speed_dot_y_dot = x_k1k[3]/(np.linalg.norm(x_k1k[2:4]) + 1e-6)
        D_k1 += [[0, 0, speed_dot_x_dot, speed_dot_y_dot, 0]]
        
        return np.array(D_k1,)
    
    def f(self, x_k: np.ndarray, u_k: np.ndarray) -> np.ndarray:
        A = np.array([[1, 0, self.dt, 0, 0],
                [0, 1, 0, self.dt, 0],
                [0, 0, (1-self.car_env.friction*self.dt), 0, 0],
                [0, 0, 0, (1-self.car_env.friction*self.dt), 0],
                [0, 0, 0, 0, 1]],)
        
        B = np.array([[0, 0],
                        [0, 0],
                        [self.dt*np.cos(x_k[4]), 0],
                        [self.dt*np.sin(x_k[4]), 0],
                        [0, self.dt]],)
        
        x_k1 = A @ x_k + B @ u_k
        x_k1[4] = normalize_angle(x_k1[4])
        return x_k1
    
    def h(self, x_k1k: np.ndarray) -> np.ndarray:
        z_k1k = []
        for landmark_position in self.landmark_positions:
            rho = np.linalg.norm(landmark_position - x_k1k[:2])
            phi = normalize_angle(np.arctan2(landmark_position[1] - x_k1k[1], landmark_position[0] - x_k1k[0]) - x_k1k[4])
            z_k1k += [[rho, phi]]
            
        z_k1k = np.array(z_k1k,).flatten()
        speed = np.linalg.norm(x_k1k[2:4])
        z_k1k = np.concatenate([z_k1k, [speed]])
        
        return z_k1k
        
    def simulate(self):
        noise_var = self.car_env.noise_std**2
        Qk = np.array(noise_var*np.eye(5),)
        Rk1 = np.array(noise_var*np.eye(len(self.landmark_positions)*2 + 1),)
        Nk = np.array(0*np.eye(2),)
        
        while True:
            x_k = self.x[-1]
            P_k = self.P[-1]
            
            input_angular_vel = self.controller.angle_velocity
            input_linear_acc = self.controller.acceleration

            landmark_measurements = np.array(self.car_env.landmark_measurements).flatten()
            u_k = np.array([input_linear_acc, input_angular_vel],)
            z_k1 = np.concatenate([landmark_measurements, [self.speed_sensor.speed]])
            
            x_k1, P_k1 = self.kf.update(x_k, u_k, z_k1, P_k, Qk, Rk1, Nk)
            
            self.x.append(x_k1)
            self.P.append(P_k1)
            
            # Set x to only have the last 10 elements
            if len(self.x) > 10:
                self.x.pop(0)
                self.P.pop(0)
                
            real_position = np.array(self.car_env.position)
            real_angle = self.car_env.angle
            
            positional_error = np.linalg.norm(real_position - x_k1[:2])
            angular_error = np.abs(normalize_angle(real_angle - x_k1[4]))
            print(f"Positional Error: {positional_error:.2f}\tAngular Error: {np.degrees(angular_error):.2f}")
            #print(f'Predicted Position: {x_k1[:2]}\tCovariance: {P_k1[:2, :2]}\n')
            
            time.sleep(self.dt)
            
    def run(self):
        def delay_simulation():
            time.sleep(1)
            self.simulate()
        
        simulation_thread = threading.Thread(target=delay_simulation, daemon=True)
        simulation_thread.start()
        
        self.car_env.run()
        
if __name__ == "__main__":
    # Generate random landmark positions
    #Set seed
    
    # Slight lag is likely due to threading
    np.random.seed(0)
    landmark_positions = [(np.random.randint(0, 200), np.random.randint(0, 600)) for _ in range(3)]
    controller = Controller(0.1)
    
    sim = CarSim(landmark_positions,
                controller = controller,
                window_size=(800, 600),
                initial_position=(400, 300),
                sensor_interval=0.1,
                noise_std=1)
    
    sim.run()