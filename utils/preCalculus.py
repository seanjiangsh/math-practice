import math
import sympy as sp


def normalize_angle(angle, symbolic=False):
    """
    Normalize an angle to its coterminal angle in the range [0, 2π).
    
    Args:
        angle (float or sympy symbol): The angle in radians
        symbolic (bool): If True, use sympy for symbolic computation
        
    Returns:
        float or sympy expr: The coterminal angle in the range [0, 2π)
    """
    if symbolic:
        return sp.Mod(angle, 2 * sp.pi)
    else:
        normalized = angle % (2 * math.pi)
        # Handle potential floating-point precision issues near 2π
        if math.isclose(normalized, 2 * math.pi, abs_tol=1e-10):
            normalized = 0.0
        return normalized


def rectangular_to_polar(x, y, symbolic=False):
    """
    Convert rectangular (Cartesian) coordinates to polar coordinates.
    
    Args:
        x (float or sympy symbol): The x-coordinate in the Cartesian plane
        y (float or sympy symbol): The y-coordinate in the Cartesian plane
        symbolic (bool): If True, use sympy for symbolic computation
        
    Returns:
        tuple: A tuple containing (r, theta) where:
            r (float or sympy expr): The radial distance from the origin
            theta (float or sympy expr): The angle in radians from the positive x-axis
                (normalized to range [0, 2π))
    """
    if symbolic:
        r = sp.sqrt(x**2 + y**2)
        theta = sp.atan2(y, x)
    else:
        r = math.sqrt(x**2 + y**2)
        theta = math.atan2(y, x)

    # Use the normalize_angle function to ensure theta is in [0, 2π)
    theta = normalize_angle(theta, symbolic)

    return r, theta


def polar_to_rectangular(r, theta, symbolic=False):
    """
    Convert polar coordinates to rectangular (Cartesian) coordinates.
    
    Args:
        r (float or sympy symbol): The radial distance from the origin
        theta (float or sympy symbol): The angle in radians from the positive x-axis
        symbolic (bool): If True, use sympy for symbolic computation
        
    Returns:
        tuple: A tuple containing (x, y) Cartesian coordinates where:
            x (float or sympy expr): The x-coordinate in the Cartesian plane
            y (float or sympy expr): The y-coordinate in the Cartesian plane
    """
    if symbolic:
        x = r * sp.cos(theta)
        y = r * sp.sin(theta)
    else:
        x = r * math.cos(theta)
        y = r * math.sin(theta)

    return x, y
