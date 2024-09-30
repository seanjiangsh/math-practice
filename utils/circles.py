from sympy import symbols, Rational, simplify, sqrt


def find_circle_equation(A, B, D, E, F):
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
