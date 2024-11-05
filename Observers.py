import numpy as np
from abc import ABC, abstractmethod
from typing import List, Any

class Observer(ABC):
    """Abstract base class for landmark observers"""
    def __init__(self) -> None:
        self._update_interval: float = 1.0
        self._attributes: List[str] = []
    
    @property
    @abstractmethod
    def update_interval(self) -> float:
        """Required update interval property"""
        pass
    
    @update_interval.setter
    @abstractmethod
    def update_interval(self, value: float) -> None:
        pass
    
    @property
    @abstractmethod
    def attributes(self) -> List[str]:
        """Required attributes property"""
        pass
    
    @attributes.setter
    @abstractmethod
    def attributes(self, value: List[str]) -> None:
        pass
    
    @abstractmethod
    def update(self, *args: Any) -> None:
        """Called when car position updates"""
        pass

class LandmarkSensor(Observer):
    """Observer that prints car position to console"""
    def __init__(self, update_interval: float, attributes: List[str] = ["landmark_measurements"]) -> None:
        super().__init__()
        self.landmark_measurements = []
        self._update_interval = update_interval
        self._attributes = attributes
    
    @property
    def update_interval(self) -> float:
        return self._update_interval
    
    @update_interval.setter
    def update_interval(self, value: float) -> None:
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("update_interval must be a positive number")
        self._update_interval = float(value)
    
    @property
    def attributes(self) -> List[str]:
        return self._attributes
    
    @attributes.setter
    def attributes(self, value: List[str]) -> None:
        if not isinstance(value, list) or not all(isinstance(attr, str) for attr in value):
            raise ValueError("attributes must be a list of strings")
        self._attributes = value
    
    def print_landmark_measurements(self) -> None:
        print("---------------------------------")
        for i, (distance, angle) in enumerate(self.landmark_measurements):
            print(f"Landmark {i+1}: distance={distance:.2f}, angle={np.degrees(angle):.2f}°")
        print("---------------------------------")
    
    def update(self, *landmark_measurements: Any) -> None:
        self.landmark_measurements = landmark_measurements[0]