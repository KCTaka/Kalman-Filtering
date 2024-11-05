import numpy as np

def normalize_angle(angle: float):
    """Normalize angle to range [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi