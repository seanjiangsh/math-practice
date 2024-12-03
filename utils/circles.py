import sympy
import math
from sympy import symbols, Rational, simplify, sqrt


def find_circle_equation(A: float, B: float, D: float, E: float, F: float):
    """
    Finds the equation of a circle given the coefficients of the general form 
    Ax^2 + By^2 + Dx + Ey + F = 0. Assumes A == B for a circle.
    
    Parameters:
    A (int or float): Coefficient of x^2.
    B (int or float): Coefficient of y^2.
    D (int or float): Coefficient of x.
    E (int or float): Coefficient of y.
    F (int or float): Constant term.

    Returns:
    tuple: A tuple containing:
      - circle_equation (str): The equation of the circle in the form (x - h)^2 + (y - k)^2 = r^2.
      - center (tuple): The center of the circle as a tuple (h, k).
      - radius (sympy.core.mul.Mul): The radius of the circle.
    
    Example:
    # x^2 + y^2 + 24x + 10y + 160 = 0
    A, B, D, E, F = 1, 1, 24, 10, 160
    find_circle_equation(A, B, D, E, F)

    """

    # Define the variables x and y
    x, y = symbols('x y')

    # Normalize the coefficients using Rational to avoid floating-point errors
    if A != 1 or B != 1:
        D = Rational(D, A)
        E = Rational(E, B)
        F = Rational(F, A)  # Assume A == B for a circle, otherwise adjust differently

    # Completing the square for x
    h = Rational(-D, 2)
    x_term = f"(x + {-h})" if h < 0 else f"(x - {h})"

    # Completing the square for y
    k = Rational(-E, 2)
    y_term = f"(y + {-k})" if k < 0 else f"(y - {k})"

    # Radius squared
    r_squared = simplify(h**2 + k**2 - F)
    radius = sqrt(r_squared)

    # Form the equation in a more intuitive format
    circle_equation = f"{x_term}^2 + {y_term}^2 = {r_squared}"

    # Return the equation, center, and radius
    center = (h, k)
    return circle_equation, center, radius


def get_arc_length(radius: float, angle: float, symbolic=False):
    """
    Calculates the arc length of a circle given the radius and angle in degrees.
    
    Parameters:
    radius (float): The radius of the circle.
    angle (float): The angle in degrees.
    symbolic (bool): If True, returns the symbolic expression.
    
    Returns:
    float: The arc length of the circle.
    
    Formula:
    arc_length = radius * angle * (pi / 180)
    or
    arc_length = (angle/360) * (2 * pi * radius)

    Example:
    get_arc_length(5, 90)
    """

    # Convert the angle to radians
    pi = sympy.pi if symbolic else math.pi
    angle_radians = angle * (pi / 180)

    # Calculate the arc length
    arc_length = radius * angle_radians
    return arc_length


def get_angle_by_arc_length(radius: float, arc_length: float, symbolic=False):
    """
    Calculates the angle of a circle given the radius and arc length.
    
    Parameters:
    radius (float): The radius of the circle.
    arc_length (float): The arc length.
    symbolic (bool): If True, returns the symbolic expression.
    
    Returns:
    float: The angle of the circle in degrees.
    
    Formula:
    angle = (arc_length / radius) * (180 / pi)

    Example:
    get_angle_by_arc_length(5, 5)
    """

    # Convert the angle to degrees
    pi = sympy.pi if symbolic else math.pi
    angle = (arc_length / radius) * (180 / pi)
    return angle
