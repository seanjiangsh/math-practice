import math
import sympy as sp
from typing import Tuple, Dict, Union, Optional, List


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


def print_circular_function_values(radian: float, round_to=4):
    """
    Print the values of circular functions at a given angle in radians.

    Args:
        radian (float): The angle in radians.
    """

    sin_s = math.sin(radian)
    cos_s = math.cos(radian)
    tan_s = math.tan(radian)
    csc_s = 1 / sin_s
    sec_s = 1 / cos_s
    cot_s = 1 / tan_s

    print(f"$\\sin{{{round(radian, round_to)}}} \\approx {round(sin_s, round_to)}$")
    print(f"$\\cos{{{round(radian, round_to)}}} \\approx {round(cos_s, round_to)}$")
    print(f"$\\tan{{{round(radian, round_to)}}} \\approx {round(tan_s, round_to)}$")
    print(f"$\\csc{{{round(radian, round_to)}}} \\approx {round(csc_s, round_to)}$")
    print(f"$\\sec{{{round(radian, round_to)}}} \\approx {round(sec_s, round_to)}$")
    print(f"$\\cot{{{round(radian, round_to)}}} \\approx {round(cot_s, round_to)}$")


def _print_triangle_solution(triangle: Dict[str, float], is_radians: bool, symbolic: bool = False):
    """Helper function to print triangle solution in a consistent format."""
    if symbolic:
        print(f"Side a: {sp.nsimplify(triangle['a'])}")
        print(f"Side b: {sp.nsimplify(triangle['b'])}")
        print(f"Side c: {sp.nsimplify(triangle['c'])}")

        if not is_radians:
            print(f"Angle A: {sp.nsimplify(triangle['A'])}°")
            print(f"Angle B: {sp.nsimplify(triangle['B'])}°")
            print(f"Angle C: {sp.nsimplify(triangle['C'])}°")
        else:
            print(f"Angle A: {sp.nsimplify(triangle['A'])} rad")
            print(f"Angle B: {sp.nsimplify(triangle['B'])} rad")
            print(f"Angle C: {sp.nsimplify(triangle['C'])} rad")
    else:
        print(f"Side a: {triangle['a']:.4f}")
        print(f"Side b: {triangle['b']:.4f}")
        print(f"Side c: {triangle['c']:.4f}")

        if not is_radians:
            print(f"Angle A: {triangle['A']:.4f}°")
            print(f"Angle B: {triangle['B']:.4f}°")
            print(f"Angle C: {triangle['C']:.4f}°")
        else:
            print(f"Angle A: {triangle['A']:.4f} rad")
            print(f"Angle B: {triangle['B']:.4f} rad")
            print(f"Angle C: {triangle['C']:.4f} rad")


def solve_triangle_SAA_ASA(known_side: float,
                           known_angles: Tuple[float, float],
                           side_label: str = 'a',
                           angle_labels: Tuple[str, str] = ('A', 'B'),
                           is_radians: bool = False,
                           print_result: bool = True,
                           symbolic: bool = False) -> Dict[str, Union[float, sp.Expr]]:
    """
    Solve a triangle using the law of sines for SAA (Side-Angle-Angle) or ASA (Angle-Side-Angle) cases.
    
    Args:
        known_side (float): The known side length of the triangle.
        known_angles (Tuple[float, float]): Two known angles of the triangle.
        side_label (str, optional): Label of the known side. Defaults to 'a'.
        angle_labels (Tuple[str, str], optional): Labels of the known angles. Defaults to ('A', 'B').
        is_radians (bool, optional): Whether the angles are in radians. Defaults to False.
        print_result (bool, optional): Whether to print the results. Defaults to True.
        symbolic (bool, optional): Whether to use symbolic calculations. Defaults to False.
    
    Returns:
        Dict[str, Union[float, sp.Expr]]: A dictionary containing all sides and angles of the triangle.
    """
    # Convert angles to radians if needed
    angle1, angle2 = known_angles
    if not is_radians:
        angle1 = degrees_to_radians(angle1, symbolic)
        angle2 = degrees_to_radians(angle2, symbolic)

    # Calculate the third angle
    pi_val = sp.pi if symbolic else math.pi
    angle3 = pi_val - angle1 - angle2

    # Create a dictionary to store all sides and angles
    triangle = {'A': 0, 'B': 0, 'C': 0, 'a': 0, 'b': 0, 'c': 0}

    # Assign known angles
    triangle[angle_labels[0]] = angle1
    triangle[angle_labels[1]] = angle2

    # Find the third angle label
    all_angle_labels = ['A', 'B', 'C']
    third_angle_label = [label for label in all_angle_labels if label not in angle_labels][0]
    triangle[third_angle_label] = angle3

    # Assign the known side
    triangle[side_label] = known_side

    # Calculate the other sides using the law of sines
    all_side_labels = ['a', 'b', 'c']
    sin_func = sp.sin if symbolic else math.sin

    for i, side_lbl in enumerate(all_side_labels):
        if side_lbl != side_label:
            angle_lbl = all_angle_labels[i]  # Corresponding angle has same index
            opposite_to_known = all_angle_labels[all_side_labels.index(side_label)]
            triangle[side_lbl] = known_side * sin_func(triangle[angle_lbl]) / sin_func(triangle[opposite_to_known])

    # Convert angles back to degrees if input was in degrees
    if not is_radians:
        triangle['A'] = radians_to_degrees(triangle['A'], symbolic)
        triangle['B'] = radians_to_degrees(triangle['B'], symbolic)
        triangle['C'] = radians_to_degrees(triangle['C'], symbolic)

    if print_result:
        print("Triangle Solution:")
        _print_triangle_solution(triangle, is_radians, symbolic)

    return triangle


def solve_triangle_SSA(side_a: float,
                       side_b: float,
                       angle_A: float,
                       side_labels: Tuple[str, str] = ('a', 'b'),
                       angle_label: str = 'A',
                       is_radians: bool = False,
                       print_result: bool = True,
                       symbolic: bool = False) -> List[Dict[str, Union[float, sp.Expr]]]:
    """
    Solve the "ambiguous case" of SSA (Side-Side-Angle) triangles.
    This function handles the situation where given two sides and an angle opposite to one of them,
    there could be 0, 1, or 2 possible triangles.
    
    Args:
        side_a (float): The side opposite to the known angle.
        side_b (float): Another known side.
        angle_A (float): The known angle (opposite to side_a).
        side_labels (Tuple[str, str], optional): Labels of the known sides. Defaults to ('a', 'b').
        angle_label (str, optional): Label of the known angle. Defaults to 'A'.
        is_radians (bool, optional): Whether the angle is in radians. Defaults to False.
        print_result (bool, optional): Whether to print the results. Defaults to True.
        symbolic (bool, optional): Whether to use symbolic calculations. Defaults to False.
    
    Returns:
        List[Dict[str, Union[float, sp.Expr]]]: A list containing dictionaries for each possible triangle solution.
    """
    # Convert angle to radians if needed
    if not is_radians:
        angle_A_rad = degrees_to_radians(angle_A, symbolic)
    else:
        angle_A_rad = angle_A

    # Get the labels for the opposite side and angles
    side_a_label, side_b_label = side_labels

    # Find the remaining labels
    all_side_labels = ['a', 'b', 'c']
    all_angle_labels = ['A', 'B', 'C']

    side_c_label = [label for label in all_side_labels if label not in side_labels][0]
    angle_B_label = all_angle_labels[all_side_labels.index(side_b_label)]
    angle_C_label = all_angle_labels[all_side_labels.index(side_c_label)]

    # Setup math functions based on symbolic flag
    sin_func = sp.sin if symbolic else math.sin
    cos_func = sp.cos if symbolic else math.cos
    asin_func = sp.asin if symbolic else math.asin
    pi_val = sp.pi if symbolic else math.pi

    # Calculate sin(B) using the law of sines: sin(B)/b = sin(A)/a
    sin_B = (side_b * sin_func(angle_A_rad)) / side_a

    # Check which case we have
    solutions = []

    # Case 1: No possible triangles (sin(B) > 1)
    if not symbolic and sin_B > 1 and not math.isclose(sin_B, 1):
        if print_result:
            print("No possible triangles can be formed with the given measurements.")
        return solutions

    # Case 2: One triangle (right triangle) when sin(B) ≈ 1
    if not symbolic and math.isclose(sin_B, 1):
        triangle = {'A': 0, 'B': 0, 'C': 0, 'a': 0, 'b': 0, 'c': 0}
        triangle[angle_label] = angle_A_rad
        triangle[angle_B_label] = pi_val / 2
        triangle[angle_C_label] = pi_val / 2 - angle_A_rad
        triangle[side_a_label] = side_a
        triangle[side_b_label] = side_b
        triangle[side_c_label] = side_a * cos_func(angle_A_rad)

        # Convert angles back to degrees if input was in degrees
        if not is_radians:
            triangle[angle_label] = angle_A
            triangle[angle_B_label] = radians_to_degrees(triangle[angle_B_label], symbolic)
            triangle[angle_C_label] = radians_to_degrees(triangle[angle_C_label], symbolic)

        solutions.append(triangle)

        if print_result:
            print("One possible triangle (right triangle):")
            _print_triangle_solution(triangle, is_radians, symbolic)

        return solutions

    # Case 3 & 4: One or two triangles (sin(B) < 1)

    # Ensure sin_B is at most 1 for numeric case (handles potential floating-point errors)
    if not symbolic and sin_B > 1:
        sin_B = 1.0

    # First triangle solution with acute angle B
    angle_B_rad1 = asin_func(sin_B)
    angle_C_rad1 = pi_val - angle_A_rad - angle_B_rad1
    side_c1 = side_a * sin_func(angle_C_rad1) / sin_func(angle_A_rad)

    # Create first triangle solution
    triangle1 = {'A': 0, 'B': 0, 'C': 0, 'a': 0, 'b': 0, 'c': 0}
    triangle1[angle_label] = angle_A_rad
    triangle1[angle_B_label] = angle_B_rad1
    triangle1[angle_C_label] = angle_C_rad1
    triangle1[side_a_label] = side_a
    triangle1[side_b_label] = side_b
    triangle1[side_c_label] = side_c1

    # Convert angles back to degrees if input was in degrees
    if not is_radians:
        triangle1[angle_label] = angle_A
        triangle1[angle_B_label] = radians_to_degrees(triangle1[angle_B_label], symbolic)
        triangle1[angle_C_label] = radians_to_degrees(triangle1[angle_C_label], symbolic)

    solutions.append(triangle1)

    # Second triangle solution with obtuse angle B
    angle_B_rad2 = pi_val - angle_B_rad1
    angle_C_rad2 = pi_val - angle_A_rad - angle_B_rad2

    # Check validity of second solution using angle_A + angle_B' < 180
    valid_second = symbolic or (angle_C_rad2 > 0 and radians_to_degrees(angle_A_rad + angle_B_rad2, symbolic) < 180)

    if valid_second:
        side_c2 = side_a * sin_func(angle_C_rad2) / sin_func(angle_A_rad)

        # Create second triangle solution
        triangle2 = {'A': 0, 'B': 0, 'C': 0, 'a': 0, 'b': 0, 'c': 0}
        triangle2[angle_label] = angle_A_rad
        triangle2[angle_B_label] = angle_B_rad2
        triangle2[angle_C_label] = angle_C_rad2
        triangle2[side_a_label] = side_a
        triangle2[side_b_label] = side_b
        triangle2[side_c_label] = side_c2

        # Convert angles back to degrees if input was in degrees
        if not is_radians:
            triangle2[angle_label] = angle_A
            triangle2[angle_B_label] = radians_to_degrees(triangle2[angle_B_label], symbolic)
            triangle2[angle_C_label] = radians_to_degrees(triangle2[angle_C_label], symbolic)

        solutions.append(triangle2)

        if print_result:
            print("Two possible triangles:")
            print("Triangle 1:")
            _print_triangle_solution(triangle1, is_radians, symbolic)
            print("\nTriangle 2:")
            _print_triangle_solution(triangle2, is_radians, symbolic)
    elif print_result:
        print("One possible triangle:")
        _print_triangle_solution(triangle1, is_radians, symbolic)

    return solutions


def calculate_triangle_area_SAS(side1: float,
                                side2: float,
                                included_angle: float,
                                is_radians: bool = False,
                                print_result: bool = True,
                                symbolic: bool = False) -> Union[float, sp.Expr]:
    """
    Calculate the area of a triangle using two sides and the included angle (SAS).
    
    Args:
        side1 (float): First side of the triangle.
        side2 (float): Second side of the triangle.
        included_angle (float): The angle between the two sides.
        is_radians (bool, optional): Whether the angle is in radians. Defaults to False.
        print_result (bool, optional): Whether to print the result. Defaults to True.
        symbolic (bool, optional): Whether to use symbolic calculations. Defaults to False.
    
    Returns:
        Union[float, sp.Expr]: The calculated area of the triangle.
    """
    # Convert angle to radians if needed
    if not is_radians:
        angle_rad = degrees_to_radians(included_angle, symbolic)
    else:
        angle_rad = included_angle

    # Choose appropriate sin function based on symbolic flag
    sin_func = sp.sin if symbolic else math.sin

    # Calculate area using the formula: Area = (1/2) * side1 * side2 * sin(angle)
    area = 0.5 * side1 * side2 * sin_func(angle_rad)

    if print_result:
        if symbolic:
            print(f"Triangle area: {sp.nsimplify(area)}")
        else:
            print(f"Triangle area: {area:.4f}")

    return area


def calculate_triangle_area_ASA(angle1: float,
                                angle2: float,
                                included_side: float,
                                is_radians: bool = False,
                                print_result: bool = True,
                                symbolic: bool = False) -> Union[float, sp.Expr]:
    """
    Calculate the area of a triangle using two angles and the included side (ASA).
    
    Args:
        angle1 (float): First angle of the triangle.
        angle2 (float): Second angle of the triangle.
        included_side (float): The side between the two angles.
        is_radians (bool, optional): Whether angles are in radians. Defaults to False.
        print_result (bool, optional): Whether to print the result. Defaults to True.
        symbolic (bool, optional): Whether to use symbolic calculations. Defaults to False.
    
    Returns:
        Union[float, sp.Expr]: The calculated area of the triangle.
    """
    # Convert angles to radians if needed
    if not is_radians:
        angle1_rad = degrees_to_radians(angle1, symbolic)
        angle2_rad = degrees_to_radians(angle2, symbolic)
    else:
        angle1_rad = angle1
        angle2_rad = angle2

    # Choose appropriate math functions based on symbolic flag
    sin_func = sp.sin if symbolic else math.sin
    pi_val = sp.pi if symbolic else math.pi

    # Calculate the third angle
    angle3_rad = pi_val - angle1_rad - angle2_rad

    # Calculate the other two sides using the law of sines
    # If c is the included side between angles A and B, then:
    # a/sin(A) = b/sin(B) = c/sin(C)
    side_a = included_side * sin_func(angle1_rad) / sin_func(angle3_rad)
    side_b = included_side * sin_func(angle2_rad) / sin_func(angle3_rad)

    # Calculate area using the formula: Area = (1/2) * a * b * sin(C)
    area = 0.5 * side_a * side_b * sin_func(angle3_rad)

    if print_result:
        if symbolic:
            print(f"Triangle area: {sp.nsimplify(area)}")
        else:
            print(f"Triangle area: {area:.4f}")

    return area
