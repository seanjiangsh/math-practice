import math
import sympy as sp
from typing import Tuple


def degrees_to_radians(degrees: float, symbolic=False) -> float:
    """
    Convert an angle in degrees to radians.
    
    Args:
        degrees (float): The angle in degrees.
    
    Returns:
        float: The angle in radians.
    """
    if symbolic:
        return degrees * sp.pi / 180
    else:
        return degrees * math.pi / 180.0


def radians_to_degrees(radians: any, symbolic=False) -> float:
    """
    Convert an angle in radians to degrees.
    
    Args:
        radians (float): The angle in radians (ignore the pi part).
    
    Returns:
        float: The angle in degrees.
    """
    if symbolic:
        return radians * (180 / sp.pi)
    else:
        return radians * (180.0 / math.pi)


def degrees_to_DMS(degrees: float) -> Tuple[int, int, float]:
    """
    Convert an angle in degrees to degrees, minutes, and seconds.
    
    Args:
        degrees (float): The angle in degrees.
    
    Returns:
        Tuple[int, int, float]: The angle in degrees, minutes, and seconds.
    """
    degrees_int = int(degrees)
    minutes_float = (degrees - degrees_int) * 60
    minutes_int = int(minutes_float)
    seconds = (minutes_float - minutes_int) * 60

    return degrees_int, minutes_int, seconds


def DMS_to_degrees(DMS: Tuple[int, int, float]) -> float:
    """
    Convert an angle in degrees, minutes, and seconds to degrees.
    
    Args:
        DMS (Tuple[int, int, float]): The angle in degrees, minutes, and seconds.
    
    Returns:
        float: The angle in degrees.
    """
    degrees = DMS[0] + DMS[1] / 60 + DMS[2] / 3600
    return degrees


def print_coterminal_angle(angle: float):
    remain = angle % 360
    rounds = angle // 360
    print(f'The angle {angle} is equivalent to {remain} after {rounds} full rotations.')


def print_coterminal_radian(radian: any):
    remain = radian % (sp.pi * 2)
    rounds = radian // (sp.pi * 2)
    print(f'The angle {radian} is equivalent to {remain} after {rounds} full rotations.')


def find_coterminal_angle_in_range(angle: float, bounds: Tuple[float, float]):
    lower_bound, upper_bound = bounds
    while angle < lower_bound:
        angle += 360
    while angle > upper_bound:
        angle -= 360
    return angle


def find_coterminal_radian_in_range(angle: float, bounds: Tuple[float, float]):
    lower_bound, upper_bound = bounds
    while angle < lower_bound:
        angle += 2 * sp.pi
    while angle > upper_bound:
        angle -= 2 * sp.pi
    return angle
