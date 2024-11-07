import numpy as np
from typing import Callable

class KalmanFilter():
    def __init__(self, 
                 get_A_k: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                 get_B_k: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                 get_D_k1: Callable[[np.ndarray], np.ndarray], 

                 f: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                 h: Callable[[np.ndarray], np.ndarray]):
        
        self.get_A_k = get_A_k
        self.get_B_k = get_B_k
        self.get_D_k1 = get_D_k1
        
        self.f = f
        self.h = h
        
        self.W_k1 = None
        
    def state_covariance_estimation(self, A_k, B_k, D_k1, P_k, Q_k, R_k1, N_k):      
        P_k1k = A_k @ P_k @ A_k.T + B_k @ N_k @ B_k.T + Q_k
        S_k1 = D_k1 @ P_k1k @ D_k1.T + R_k1
        
        if np.all(S_k1 == 0):
            self.W_k1 = np.zeros((1, 1))
            return P_k1k
        
        self.W_k1 = P_k1k @ D_k1.T @ np.linalg.inv(S_k1)
        P_k1 = P_k1k - self.W_k1 @ S_k1 @ self.W_k1.T
        return P_k1
             
    def state_estimation(self, x_k, u_k, z_k1):
        x_k1k = self.f(x_k, u_k)
        z_k1k = self.h(x_k1k)
        s_k1 = z_k1 - z_k1k
        x_k1 = x_k1k + self.W_k1 @ s_k1
        return x_k1
    
    def update(self, x_k, u_k, z_k1, P_k, Q_k, R_k1, N_k):
        A_k = self.get_A_k(x_k, u_k)
        B_k = self.get_B_k(x_k, u_k)
        
        x_k1k = self.f(x_k, u_k)
        D_k1 = self.get_D_k1(x_k1k)
        
        P_k1 = self.state_covariance_estimation(A_k, B_k, D_k1, P_k, Q_k, R_k1, N_k)
        x_k1 = self.state_estimation(x_k, u_k, z_k1)
        
        return x_k1, P_k1

