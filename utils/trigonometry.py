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


def radian_to_degrees(radian: any) -> float:
    return radian * (180 / sp.pi)


def find_coterminal_angle(angle: float):
    remain = angle % 360
    rounds = angle // 360
    return remain, rounds


def print_coterminal_angle(angle: float):
    remain, rounds = find_coterminal_angle(angle)
    print(f'The angle {angle} is equivalent to {remain} after {rounds} full rotations.')


def find_coterminal_radian(radian: any):
    remain = radian % (sp.pi * 2)
    rounds = radian // (sp.pi * 2)
    degrees = radian_to_degrees(remain)
    return remain, rounds, degrees


def print_coterminal_radian(radian: any):
    remain, rounds, degrees = find_coterminal_radian(radian)
    print(f'The angle {radian} is equivalent to {remain} ({degrees} degrees) after {rounds} full rotations.')


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


def find_reference_angle(angle: float):
    angle = find_coterminal_angle(angle)[0]
    # Quadrant 1
    if 0 <= angle <= 90:
        return angle
    # Quadrant 2
    elif 90 < angle <= 180:
        return 180 - angle
    # Quadrant 3
    elif 180 < angle <= 270:
        return angle - 180
    # Quadrant 4
    elif 270 < angle <= 360:
        return 360 - angle
    # Angle must be in the range of 0 to 360, so this should never happen
    else:
        raise ValueError('Angle must be in the range of 0 to 360')


def find_reference_radian(radian: any):
    radian = find_coterminal_radian(radian)[0]
    # Quadrant 1
    if 0 <= radian <= sp.pi / 2:
        return radian
    # Quadrant 2
    elif sp.pi / 2 < radian <= sp.pi:
        return sp.pi - radian
    # Quadrant 3
    elif sp.pi < radian <= 3 * sp.pi / 2:
        return radian - sp.pi
    # Quadrant 4
    elif 3 * sp.pi / 2 < radian <= 2 * sp.pi:
        return 2 * sp.pi - radian
    # Radian must be in the range of 0 to 2pi, so this should never happen
    else:
        raise ValueError('Radian must be in the range of 0 to 2pi')


def list_coterminal_angles_in_degrees(degree: float, n=10):
    angles = []
    for i in range(-n, n):
        angles.append(degree + i * 360)
    return angles


def list_coterminal_angles_in_radians(radian: any, n=10):
    angles = []
    for i in range(-n, n):
        angles.append(radian + i * 2 * sp.pi)
    return angles


def get_trig_functions_from_point(point: Tuple[float, float], print_to_output=True):
    x, y = point
    r = sp.sqrt(x**2 + y**2)
    sin = y / r
    cos = x / r
    tan = y / x
    csc = 1 / sin
    sec = 1 / cos
    cot = 1 / tan

    if print_to_output:
        print(f'r = {sp.nsimplify(r)}')
        print(f'sin = {sp.nsimplify(sin)}')
        print(f'cos = {sp.nsimplify(cos)}')
        print(f'tan = {sp.nsimplify(tan)}')
        print(f'csc = {sp.nsimplify(csc)}')
        print(f'sec = {sp.nsimplify(sec)}')
        print(f'cot = {sp.nsimplify(cot)}')

    return r, sin, cos, tan, csc, sec, cot
