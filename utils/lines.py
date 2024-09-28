from sympy import sqrt


def get_length_between_points(p1: list[float], p2: list[float], symbolic=False) -> float:
    """
    Get the length between two points in any number of dimensions.
    
    Args:
        p1 (list[float]): The first point.
        p2 (list[float]): The second point.
    
    Returns:
        float: The length between the two points.
    
    Raises:
        ValueError: If the points do not have the same length or if the length is less than 2.
    """
    if len(p1) != len(p2):
        raise ValueError("The points must have the same number of dimensions.")
    if len(p1) < 2:
        raise ValueError("The points must have at least 2 dimensions.")

    sum_of_squares = sum((p2[i] - p1[i])**2 for i in range(len(p1)))
    if symbolic:
        return sqrt(sum_of_squares)
    else:
        return sum_of_squares**0.5


def get_midpoint_between_points(p1: list[float], p2: list[float]) -> list[float]:
    """
    Get the midpoint between two points in any number of dimensions.
    
    Args:
        p1 (list[float]): The first point.
        p2 (list[float]): The second point.
    
    Returns:
        list[float]: The midpoint between the two points.
    
    Raises:
        ValueError: If the points do not have the same length or if the length is less than 2.
    """
    if len(p1) != len(p2):
        raise ValueError("The points must have the same number of dimensions.")
    if len(p1) < 2:
        raise ValueError("The points must have at least 2 dimensions.")

    return tuple((p1[i] + p2[i]) / 2 for i in range(len(p1)))
