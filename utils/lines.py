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


# # Example usage:
# p1 = [1.0, 2.0, 3.0]
# p2 = [4.0, 6.0, 8.0]
# length = get_length_between_points(p1, p2)
# print(f"The length between points {tuple(p1)} and {tuple(p2)} is {length}")
