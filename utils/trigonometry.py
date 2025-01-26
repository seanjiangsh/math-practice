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
    """
    Convert an angle in radians to degrees.
    
    Args:
        radian (any): The angle in radians.
    
    Returns:
        float: The angle in degrees.
    """
    return radian * (180 / sp.pi)


def find_coterminal_angle(angle: int | float | sp.Mul) -> Tuple[float, float] | Tuple[sp.Mul, sp.Mul]:
    """
    Find the coterminal angle of a given angle.
    
    Args:
        angle (int | float | sp.Mul): The angle in degrees or radians.
    
    Returns:
        Tuple[float, float] | Tuple[sp.Mul, sp.Mul]: The coterminal angle and the number of full rotations.
    """
    if isinstance(angle, (int, float)):
        remain = angle % 360
        rounds = angle // 360
    else:
        remain = angle % (2 * sp.pi)
        rounds = angle // (2 * sp.pi)
    return remain, rounds


def print_coterminal_angle(angle: int | float | sp.Mul):
    """
    Print the coterminal angle of a given angle.
    
    Args:
        angle (int | float | sp.Mul): The angle in degrees or radians.
    """
    if isinstance(angle, (int, float)):
        remain, rounds = find_coterminal_angle(angle)
        print(f'The angle {angle} is equivalent to {remain} after {rounds} full rotations.')
    else:
        remain, rounds = find_coterminal_angle(angle)
        degrees = radian_to_degrees(remain)
        print(f'The angle {angle} is equivalent to {remain} ({degrees} degrees) after {rounds} full rotations.')


def find_coterminal_angle_in_range(angle: int | float | sp.Mul,
                                   bounds: Tuple[int | float, int | float] | Tuple[sp.Mul, sp.Mul]) -> int | float | sp.Mul:
    """
    Find the coterminal angle of a given angle within specified bounds.
    
    Args:
        angle (int | float | sp.Mul): The angle in degrees or radians.
        bounds (Tuple[int | float, int | float] | Tuple[sp.Mul, sp.Mul]): The lower and upper bounds.
    
    Returns:
        int | float | sp.Mul: The coterminal angle within the specified bounds.
    """
    lower_bound, upper_bound = bounds
    if isinstance(angle, (int, float)):
        while angle < lower_bound:
            angle += 360
        while angle > upper_bound:
            angle -= 360
    else:
        while angle < lower_bound:
            angle += 2 * sp.pi
        while angle > upper_bound:
            angle -= 2 * sp.pi
    return angle


def find_reference_angle(angle: int | float | sp.Mul) -> int | float | sp.Mul:
    """
    Find the reference angle of a given angle.
    
    Args:
        angle (int | float | sp.Mul): The angle in degrees or radians.
    
    Returns:
        int | float | sp.Mul: The reference angle.
    """
    angle = find_coterminal_angle(angle)[0]
    if isinstance(angle, (int, float)):
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
    else:
        # Quadrant 1
        if 0 <= angle <= sp.pi / 2:
            return angle
        # Quadrant 2
        elif sp.pi / 2 < angle <= sp.pi:
            return sp.pi - angle
        # Quadrant 3
        elif sp.pi < angle <= 3 * sp.pi / 2:
            return angle - sp.pi
        # Quadrant 4
        elif 3 * sp.pi / 2 < angle <= 2 * sp.pi:
            return 2 * sp.pi - angle
        # Angle must be in the range of 0 to 2pi, so this should never happen
        else:
            raise ValueError('Radian must be in the range of 0 to 2pi')


def list_coterminal_angles(degree: int | float | sp.Mul, n=10):
    """
    List coterminal angles of a given angle in degrees.
    
    Args:
        degree (int | float): The angle in degrees.
        n (int, optional): The number of coterminal angles to list. Defaults to 10.
    
    Returns:
        list: A list of coterminal angles in degrees.
    """
    angles = []
    if isinstance(degree, (int, float)):
        for i in range(-n, n):
            angles.append(degree + i * 360)
    else:
        for i in range(-n, n):
            angles.append(degree + i * 2 * sp.pi)
    return angles


def get_trig_functions_from_point(point: Tuple[float, float], print_to_output=True):
    """
    Calculate trigonometric functions from a given point.
    
    Args:
        point (Tuple[float, float]): The point (x, y).
        print_to_output (bool, optional): Whether to print the results. Defaults to True.
    
    Returns:
        Tuple: The radius, sine, cosine, tangent, cosecant, secant, and cotangent values.
    """
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
