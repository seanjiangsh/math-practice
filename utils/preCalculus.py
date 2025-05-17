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


def get_coterminal_polar_point(polar_point, symbolic=False):
    """
    Find the coterminal polar point with angle in the range [0, 2π).
    
    Args:
        polar_point (tuple): A tuple containing (r, theta) where:
            r (float or sympy symbol): The radial distance from the origin
            theta (float or sympy symbol): The angle in radians
        symbolic (bool): If True, use sympy for symbolic computation
        
    Returns:
        tuple: A tuple containing (r, normalized_theta) where:
            r (float or sympy expr): The same radial distance
            normalized_theta (float or sympy expr): The coterminal angle in range [0, 2π)
    """
    r, theta = polar_point
    normalized_theta = normalize_angle(theta, symbolic)
    return r, normalized_theta


def find_equivalent_polar_points(polar_point, n_points=5, angle_range=None, symbolic=False):
    """
    Find equivalent polar points to (r, theta).
    
    Args:
        polar_point (tuple): A tuple containing (r, theta) where:
            r (float or sympy symbol): The radial distance from the origin
            theta (float or sympy symbol): The angle in radians
        n_points (int): Number of equivalent points to find
        angle_range (tuple, optional): (min_angle, max_angle) range for the returned angles.
            If None, find closest points to angle 0.
        symbolic (bool): If True, use sympy for symbolic computation
    
    Returns:
        list: List of tuples [(r1, theta1), (r2, theta2), ...] of equivalent polar points
              (excluding the original input point)
    """
    r, theta = polar_point
    pi = sp.pi if symbolic else math.pi
    two_pi = 2 * pi

    # Case 1: If angle_range is None, find closest points towards angle 0
    if angle_range is None:
        results = []
        r, theta = polar_point
        pi_val = sp.pi if symbolic else math.pi

        # Generate equivalent points by adding and subtracting multiples of pi
        k = 1
        while len(results) < n_points:
            # For odd multiples of pi, negate r
            new_r = -r if k % 2 == 1 else r

            # Add point with +k*pi
            results.append((new_r, theta + k * pi_val))

            # If we need more points, add point with -k*pi
            if len(results) < n_points:
                results.append((new_r, theta - k * pi_val))

            k += 1

        return results[:n_points]

    # Case 2: If angle_range is provided, use the existing approach
    min_angle, max_angle = angle_range

    # Step 1: Determine if theta is closer to the lower or upper bound of the angle range
    normalized_theta = normalize_angle(theta, symbolic)

    # Calculate distances to both bounds
    if symbolic:
        dist_to_min = abs(normalized_theta - min_angle)
        dist_to_max = abs(normalized_theta - max_angle)
    else:
        dist_to_min = abs(normalized_theta - min_angle)
        dist_to_max = abs(normalized_theta - max_angle)

    search_upward = dist_to_min <= dist_to_max

    results = []
    offsets = []

    if search_upward:
        k = 0
        while len(offsets) < 2 * n_points and k <= 50:
            offsets.append(k)
            if k > 0:
                offsets.append(-k)
            k += 1
    else:
        k = 0
        while len(offsets) < 2 * n_points and k <= 50:
            offsets.append(-k)
            if k > 0:
                offsets.append(k)
            k += 1

    for offset in offsets:
        new_theta = theta + offset * two_pi
        if min_angle <= new_theta <= max_angle:
            if offset != 0 or not (symbolic and new_theta == theta or not symbolic and math.isclose(new_theta, theta, abs_tol=1e-10)):
                results.append((r, new_theta))

        neg_theta = theta + pi + offset * two_pi
        if min_angle <= neg_theta <= max_angle:
            results.append((-r, neg_theta))

        if len(results) >= n_points:
            break

    results.sort(key=lambda x: x[1])
    return results[:n_points]
