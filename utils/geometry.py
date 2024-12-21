from typing import Literal


def get_sum_of_angles_in_polygon(num_sides: int, symbolic=False) -> float:
    """
    Get the sum of the angles in a polygon with a given number of sides.
    
    Args:
        num_sides (int): The number of sides in the polygon.
    
    Returns:
        float: The sum of the angles in the polygon.
    
    Raises:
        ValueError: If the number of sides is less than 3.
    """
    if num_sides < 3:
        raise ValueError("The number of sides must be at least 3.")

    if symbolic:
        return (num_sides - 2) * 180
    else:
        return (num_sides - 2) * 180.0


def get_interior_angle_of_regular_polygon(num_sides: int, symbolic=False) -> float:
    """
    Get the measure of an angle in a regular polygon with a given number of sides.
    
    Args:
        num_sides (int): The number of sides in the polygon.
    
    Returns:
        float: The measure of an angle in the regular polygon.
    
    Raises:
        ValueError: If the number of sides is less than 3.
    """
    if num_sides < 3:
        raise ValueError("The number of sides must be at least 3.")

    return get_sum_of_angles_in_polygon(num_sides, symbolic) / num_sides


def get_exterior_angle_of_regular_polygon(num_sides: int) -> float:
    """
    Get the measure of an exterior angle in a regular polygon with a given number of sides.
    
    Args:
        num_sides (int): The number of sides in the polygon.
    
    Returns:
        float: The measure of an exterior angle in the regular polygon.
    
    Raises:
        ValueError: If the number of sides is less than 3.
    """
    if num_sides < 3:
        raise ValueError("The number of sides must be at least 3.")

    return 360 / num_sides


def find_sides_by_interior_angle_of_regular_polygon(angle: float) -> int:
    """
    Find the number of sides in a regular polygon given the measure of an interior angle.
    
    Args:
        angle (float): The measure of an interior angle in the regular polygon.
    
    Returns:
        int: The number of sides in the regular polygon.
    
    Raises:
        ValueError: If the angle is less than 0 or greater than 180.
    """
    if angle <= 0 or angle >= 180:
        raise ValueError("The angle must be between 0 and 180 degrees.")

    return round(360 / (180 - angle))


def find_sides_by_exterior_angle_of_regular_polygon(angle: float) -> int:
    """
    Find the number of sides in a regular polygon given the measure of an exterior angle.
    
    Args:
        angle (float): The measure of an exterior angle in the regular polygon.
    
    Returns:
        int: The number of sides in the regular polygon.
    
    Raises:
        ValueError: If the angle is less than 0 or greater than 360.
    """
    if angle <= 0 or angle >= 360:
        raise ValueError("The angle must be between 0 and 360 degrees.")

    return round(360 / angle)
