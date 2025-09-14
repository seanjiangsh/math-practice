import math
import numpy as np
import sympy as sp
from typing import Tuple, List, Union, Callable, Optional


def normalize_angle(angle: Union[float, sp.Expr], symbolic: bool = False) -> Union[float, sp.Expr]:
    """
    Normalize an angle to its coterminal angle in the range [0, 2π).
    
    Args:
        angle: The angle in radians
        symbolic: If True, use sympy for symbolic computation
        
    Returns:
        The coterminal angle in the range [0, 2π)
    """
    if symbolic:
        return sp.Mod(angle, 2 * sp.pi)
    else:
        normalized = angle % (2 * math.pi)
        # Handle potential floating-point precision issues near 2π
        if math.isclose(normalized, 2 * math.pi, abs_tol=1e-10):
            normalized = 0.0
        return normalized


def rectangular_to_polar(x: Union[float, sp.Expr],
                         y: Union[float, sp.Expr],
                         symbolic: bool = False) -> Tuple[Union[float, sp.Expr], Union[float, sp.Expr]]:
    """
    Convert rectangular (Cartesian) coordinates to polar coordinates.
    
    Args:
        x: The x-coordinate in the Cartesian plane
        y: The y-coordinate in the Cartesian plane
        symbolic: If True, use sympy for symbolic computation
        
    Returns:
        A tuple containing (r, theta) where:
            r: The radial distance from the origin
            theta: The angle in radians from the positive x-axis
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


def polar_to_rectangular(r: Union[float, sp.Expr],
                         theta: Union[float, sp.Expr],
                         symbolic: bool = False) -> Tuple[Union[float, sp.Expr], Union[float, sp.Expr]]:
    """
    Convert polar coordinates to rectangular (Cartesian) coordinates.
    
    Args:
        r: The radial distance from the origin
        theta: The angle in radians from the positive x-axis
        symbolic: If True, use sympy for symbolic computation
        
    Returns:
        A tuple containing (x, y) Cartesian coordinates where:
            x: The x-coordinate in the Cartesian plane
            y: The y-coordinate in the Cartesian plane
    """
    if symbolic:
        x = r * sp.cos(theta)
        y = r * sp.sin(theta)
    else:
        x = r * math.cos(theta)
        y = r * math.sin(theta)

    return x, y


def get_coterminal_polar_point(polar_point: Tuple[Union[float, sp.Expr], Union[float, sp.Expr]],
                               symbolic: bool = False) -> Tuple[Union[float, sp.Expr], Union[float, sp.Expr]]:
    """
    Find the coterminal polar point with angle in the range [0, 2π).
    
    Args:
        polar_point: A tuple containing (r, theta) where:
            r: The radial distance from the origin
            theta: The angle in radians
        symbolic: If True, use sympy for symbolic computation
        
    Returns:
        A tuple containing (r, normalized_theta) where:
            r: The same radial distance
            normalized_theta: The coterminal angle in range [0, 2π)
    """
    r, theta = polar_point
    normalized_theta = normalize_angle(theta, symbolic)
    return r, normalized_theta


def find_equivalent_polar_points(polar_point: Tuple[Union[float, sp.Expr], Union[float, sp.Expr]],
                                 n_points: int = 5,
                                 angle_range: Optional[Tuple[Union[float, sp.Expr], Union[float, sp.Expr]]] = None,
                                 symbolic: bool = False) -> List[Tuple[Union[float, sp.Expr], Union[float, sp.Expr]]]:
    """
    Find equivalent polar points to (r, theta).
    
    Args:
        polar_point: A tuple containing (r, theta) where:
            r: The radial distance from the origin
            theta: The angle in radians
        n_points: Number of equivalent points to find
        angle_range: (min_angle, max_angle) range for the returned angles.
            If None, find closest points to angle 0.
        symbolic: If True, use sympy for symbolic computation
    
    Returns:
        List of tuples [(r1, theta1), (r2, theta2), ...] of equivalent polar points
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


def find_polar_vertices(equation: Callable[[np.ndarray], np.ndarray],
                        theta_range: Tuple[float, float] = (0, 2 * np.pi),
                        n_points: int = 1000,
                        min_distance: float = 0.05) -> List[Tuple[float, float]]:
    """
    Find vertices (local maxima and minima) of a polar equation.
    
    Args:
        equation: Function that takes theta and returns r
        theta_range: Range of theta to search (default: 0 to 2π)
        n_points: Number of points to sample for numerical differentiation
        min_distance: Minimum distance between vertices (to avoid duplicates)
        
    Returns:
        List of (theta, r) tuples representing vertices
    """
    # Create theta values for numerical differentiation
    thetas = np.linspace(theta_range[0], theta_range[1], n_points)
    delta = (theta_range[1] - theta_range[0]) / n_points

    # Ensure output is always at least 1D array
    rs = np.atleast_1d(equation(thetas))
    rs_forward = np.atleast_1d(equation(thetas + delta / 2))
    rs_backward = np.atleast_1d(equation(thetas - delta / 2))
    derivatives = (rs_forward - rs_backward) / delta

    # Find where derivative changes sign (crosses zero)
    vertices = []
    for i in range(1, len(derivatives) - 1):
        if derivatives[i - 1] * derivatives[i + 1] <= 0:
            # Found a sign change - this is approximately a vertex
            theta = thetas[i]
            r = equation(theta)

            # Normalize theta to be within [0, 2π)
            theta = theta % (2 * np.pi)

            # Check if this vertex is too close to an existing one
            is_duplicate = False
            for existing_theta, existing_r in vertices:
                # Consider angles that wrap around (near 0 and 2π)
                theta_diff = min(abs(theta - existing_theta), abs(theta - existing_theta + 2 * np.pi),
                                 abs(theta - existing_theta - 2 * np.pi))
                r_diff = abs(r - existing_r)

                # Consider points the same if they're close in both theta and r
                # Use a relative threshold for r based on its magnitude
                r_threshold = min_distance * (1 + abs(r))
                if theta_diff < min_distance and r_diff < r_threshold:
                    is_duplicate = True
                    break

            # Only add non-duplicate vertices
            if not is_duplicate:
                vertices.append((theta, r))

    # Sort vertices by theta value for consistent output
    vertices.sort(key=lambda v: v[0])

    return vertices


def get_pi_fraction_string(angle: float) -> str:
    """
    Convert an angle in radians to a fraction of π as a string.
    
    Args:
        angle: Angle in radians
        
    Returns:
        String representation like "π/4", "π/2", "3π/4", etc.
    """
    # Normalize angle to [0, 2π)
    angle = angle % (2 * np.pi)

    # Special cases
    if abs(angle) < 0.01:
        return "0"
    if abs(angle - np.pi) < 0.01:
        return "π"
    if abs(angle - 2 * np.pi) < 0.01:
        return "2π"

    # Common fractions of π to check
    fractions = [(np.pi / 6, "π/6"), (np.pi / 4, "π/4"), (np.pi / 3, "π/3"), (np.pi / 2, "π/2"), (2 * np.pi / 3, "2π/3"),
                 (3 * np.pi / 4, "3π/4"), (5 * np.pi / 6, "5π/6"), (7 * np.pi / 6, "7π/6"), (5 * np.pi / 4, "5π/4"),
                 (4 * np.pi / 3, "4π/3"), (3 * np.pi / 2, "3π/2"), (5 * np.pi / 3, "5π/3"), (7 * np.pi / 4, "7π/4"),
                 (11 * np.pi / 6, "11π/6")]

    # Check if angle is close to any common fraction
    for frac_value, frac_str in fractions:
        if abs(angle - frac_value) < 0.01:
            return frac_str

    # If not a common fraction, express in terms of π with 2 decimal places
    pi_multiple = angle / np.pi
    return f"{pi_multiple:.2f}π"


def gauss_jordan_elimination(matrix: Union[List[List[float]], np.ndarray], tolerance: float = 1e-10) -> Tuple[np.ndarray, str]:
    """
    Perform Gauss-Jordan elimination to solve a system of linear equations.
    
    Args:
        matrix: Augmented matrix [A|b] where A is the coefficient matrix and b is the constants vector.
                Can be a list of lists or numpy array.
        tolerance: Numerical tolerance for considering values as zero
        
    Returns:
        Tuple containing:
            - Reduced row echelon form (RREF) matrix
            - Solution status: "unique", "no_solution", or "infinite_solutions"
    """
    # Convert to numpy array for easier manipulation
    if isinstance(matrix, list):
        A = np.array(matrix, dtype=float)
    else:
        A = matrix.astype(float)

    rows, cols = A.shape

    # Track which columns contain pivot elements
    pivot_cols = []

    for row in range(rows):
        # Find the pivot column (first non-zero element in current row)
        pivot_col = None
        for col in range(cols - 1):  # Don't include the augmented column
            if abs(A[row, col]) > tolerance:
                pivot_col = col
                break

        if pivot_col is None:
            # Current row is all zeros (except possibly the augmented part)
            if abs(A[row, -1]) > tolerance:
                # Inconsistent system: 0 = non-zero
                return A, "no_solution"
            continue

        pivot_cols.append(pivot_col)

        # Make the pivot element 1
        pivot_element = A[row, pivot_col]
        A[row, :] = A[row, :] / pivot_element

        # Eliminate all other elements in the pivot column
        for other_row in range(rows):
            if other_row != row and abs(A[other_row, pivot_col]) > tolerance:
                multiplier = A[other_row, pivot_col]
                A[other_row, :] = A[other_row, :] - multiplier * A[row, :]

    # Clean up small values that should be zero
    A[np.abs(A) < tolerance] = 0

    # Determine solution type
    num_variables = cols - 1
    num_pivots = len(pivot_cols)

    # Check for inconsistent system
    for row in range(num_pivots, rows):
        if abs(A[row, -1]) > tolerance:
            return A, "no_solution"

    if num_pivots == num_variables:
        return A, "unique"
    else:
        return A, "infinite_solutions"


def solve_linear_system(matrix: Union[List[List[float]], np.ndarray],
                        variable_names: Optional[List[str]] = None,
                        tolerance: float = 1e-10) -> dict:
    """
    Solve a system of linear equations and return the solution in a readable format.
    
    Args:
        matrix: Augmented matrix [A|b]
        variable_names: Names for the variables (e.g., ['x', 'y', 'z'])
        tolerance: Numerical tolerance for considering values as zero
        
    Returns:
        Dictionary containing:
            - 'rref': The reduced row echelon form matrix
            - 'status': Solution status
            - 'solution': The solution (if unique) or description of solution set
            - 'free_variables': List of free variables (if infinite solutions)
    """
    rref_matrix, status = gauss_jordan_elimination(matrix, tolerance)

    rows, cols = rref_matrix.shape
    num_variables = cols - 1

    # Generate default variable names if not provided
    if variable_names is None:
        if num_variables <= 3:
            variable_names = ['x', 'y', 'z'][:num_variables]
        else:
            variable_names = [f'x{i+1}' for i in range(num_variables)]

    result = {'rref': rref_matrix, 'status': status, 'solution': None, 'free_variables': []}

    if status == "no_solution":
        result['solution'] = "No solution exists (inconsistent system)"
        return result

    if status == "unique":
        # Extract the unique solution
        solution = {}
        for i, var_name in enumerate(variable_names):
            # Find the row with pivot in column i
            for row in range(rows):
                if abs(rref_matrix[row, i]) > tolerance and abs(rref_matrix[row, i] - 1) < tolerance:
                    # Check if this is indeed a pivot (all other elements in row are 0 except augmented part)
                    is_pivot = True
                    for col in range(num_variables):
                        if col != i and abs(rref_matrix[row, col]) > tolerance:
                            is_pivot = False
                            break
                    if is_pivot:
                        solution[var_name] = rref_matrix[row, -1]
                        break
            if var_name not in solution:
                solution[var_name] = 0  # Free variable set to 0

        result['solution'] = solution
        return result

    if status == "infinite_solutions":
        # Identify pivot and free variables
        pivot_vars = []
        free_vars = []

        # Find pivot columns
        pivot_columns = []
        for row in range(rows):
            for col in range(num_variables):
                if abs(rref_matrix[row, col] - 1) < tolerance:
                    # Check if this is a pivot
                    is_pivot = True
                    for other_col in range(col):
                        if abs(rref_matrix[row, other_col]) > tolerance:
                            is_pivot = False
                            break
                    if is_pivot and col not in pivot_columns:
                        pivot_columns.append(col)
                        pivot_vars.append(variable_names[col])
                        break

        for i, var_name in enumerate(variable_names):
            if i not in pivot_columns:
                free_vars.append(var_name)

        result['free_variables'] = free_vars

        # Express pivot variables in terms of free variables
        solution_expressions = {}
        for row in range(len(pivot_columns)):
            pivot_col = pivot_columns[row]
            pivot_var = variable_names[pivot_col]

            # Build expression: pivot_var = constant + sum(coefficient * free_var)
            constant = rref_matrix[row, -1]
            coefficients = {}

            for col in range(num_variables):
                if col != pivot_col and abs(rref_matrix[row, col]) > tolerance:
                    var_name = variable_names[col]
                    coefficients[var_name] = -rref_matrix[row, col]  # Negative because we move to RHS

            expression = f"{pivot_var} = {constant:.3f}"
            for var_name, coeff in coefficients.items():
                if coeff >= 0:
                    expression += f" + {coeff:.3f}*{var_name}"
                else:
                    expression += f" - {abs(coeff):.3f}*{var_name}"

            solution_expressions[pivot_var] = expression

        result['solution'] = solution_expressions
        return result
