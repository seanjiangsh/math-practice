# Type definitions

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import math
from scipy.signal import find_peaks

from numpy import ndarray
from typing import TypedDict, Optional

from preCalculus import get_pi_fraction_string, find_polar_vertices


class PointDict(TypedDict):
    x: float
    y: float
    equal: Optional[bool]
    label: Optional[str]


class LineDict(TypedDict):
    x: ndarray[float]
    y: ndarray[float]
    color: Optional[str]
    linestyle: Optional[str]
    label: Optional[str]
    points: Optional[list[PointDict]]


class FillBetweenDict(TypedDict):
    x: ndarray[float]  # X range for filling
    y1: ndarray[float]  # Y values for the lower boundary
    y2: ndarray[float]  # Y values for the upper boundary
    where: ndarray[bool]  # Condition for filling
    color: Optional[str]  # Color for filling


class LimitDict(TypedDict):
    x: tuple[float, float]
    y: tuple[float, float]


class PolarCurveDict(TypedDict):
    theta: ndarray[float]
    r: ndarray[float]
    color: Optional[str]
    linestyle: Optional[str]
    label: Optional[str]
    points: Optional[list[PointDict]]


# Functions


def setup_plot(limits: LimitDict = None) -> None:
    """
    Set up the plot with grid, ticks, and limits.
    """
    plt.axhline(0, color='black', linewidth=1.0)
    plt.axvline(0, color='black', linewidth=1.0)
    plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    if limits:
        plt.xlim(limits['x'])
        plt.ylim(limits['y'])

        x_start = (limits['x'][0])
        x_end = ((limits['x'][1]) + 1)
        y_start = (limits['y'][0])
        y_end = ((limits['y'][1]) + 1)

        x_ticks = (limits['x'][1] - limits['x'][0]) / 4
        y_ticks = (limits['y'][1] - limits['y'][0]) / 4
        plt.gca().set_xticks(np.arange(x_start, x_end, x_ticks), minor=False)
        plt.gca().set_yticks(np.arange(y_start, y_end, y_ticks), minor=False)
    else:
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.gca().set_xticks(np.arange(-10, 11, 5), minor=False)
        plt.gca().set_yticks(np.arange(-10, 11, 5), minor=False)


def plot_lines(title: str,
               lines: list[LineDict],
               fills: list[FillBetweenDict] = None,
               limits: LimitDict = None,
               xlabel: str = 'x',
               ylabel: str = 'y',
               other_points: list[PointDict] = None):
    plt.figure(figsize=(6, 6))
    setup_plot(limits)

    for line in lines:
        x = line['x']
        y = line['y']
        line_color = line.get('color', 'blue')
        linestyle = line.get('linestyle', '-')
        label = line.get('label', None)
        plt.plot(x, y, color=line_color, linestyle=linestyle, label=label)

        # Plot points
        points = line.get('points', [])
        for point in points:
            x = point['x']
            y = point['y']
            equal = point['equal'] if 'equal' in point else True
            label = point.get('label', None)
            point_color = line_color if equal else 'white'
            marker_edge_color = None if equal else line_color
            plt.plot(x, y, 'o', color=point_color, markersize=5, markeredgewidth=1, markeredgecolor=marker_edge_color)
            # Annotate each point with a label
            if label:
                plt.annotate(label, xy=(x, y), xytext=(5, 5), textcoords='offset points')

    # Plot other_points (independent of lines)
    if other_points is not None:
        for point in other_points:
            x = point['x']
            y = point['y']
            equal = point['equal'] if 'equal' in point else True
            label = point.get('label', None)
            point_color = 'blue' if equal else 'white'  # Default color for other_points
            marker_edge_color = None if equal else 'red'
            plt.plot(x, y, 'o', color=point_color, markersize=5, markeredgewidth=1, markeredgecolor=marker_edge_color)
            # Annotate each point with a label
            if label:
                plt.annotate(label, xy=(x, y), xytext=(5, 5), textcoords='offset points')

    # Plot fills
    if fills is not None:
        for fill in fills:
            x = fill['x']
            y1 = fill['y1']
            y2 = fill['y2']
            where = fill['where']
            fill_color = fill.get('color', 'lightblue')
            plt.fill_between(x, y1, y2, where, color=fill_color, alpha=0.5)

    # Set custom axis labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel, rotation=0)
    plt.title(title)

    # Check if any line has a label and add legend if true
    has_label = any(map(lambda line: isinstance(line.get('label'), str), lines))
    if has_label:
        plt.legend()

    plt.show()


def get_slope_intercept_points(slope: float, intercept=0):
    x = np.linspace(-10, 10, 100)
    # Calculate y values based on the equation y = mx + b
    y = slope * x + intercept
    return {'x': x, 'y': y}


class GeometryLineDict(TypedDict):
    coordinates: ndarray[float]  # Shape (n, 2) where each row is [x, y]
    color: Optional[str]
    point_labels: Optional[list[str]]
    label: Optional[str]


class PolygonDict(TypedDict):
    coordinates: ndarray[float]  # Shape (n, 2) where each row is [x, y]
    point_labels: list[str]
    color: Optional[str]
    label: Optional[str]


POINT_TEXT_OFFSET = 0.3  # Adjust this value to shift horizontally


def plot_geometry(title: str, lines: list[GeometryLineDict] = None, polygons: list[PolygonDict] = None, limits: LimitDict = None):
    plt.figure(figsize=(6, 6))
    setup_plot(limits)

    if lines is not None:
        for line in lines:
            coords = line['coordinates']
            x = coords[:, 0]  # Extract x coordinates
            y = coords[:, 1]  # Extract y coordinates
            line_color = line.get('color', 'blue')
            label = line.get('label', None)
            plt.plot(x, y, 'o-', color=line_color, markersize=5, label=label)

            # Plot point texts
            point_texts = line.get('point_labels', [])
            for i, text in enumerate(point_texts):
                plt.text(x[i] + POINT_TEXT_OFFSET, y[i] + POINT_TEXT_OFFSET, text)

    if polygons is not None:
        for polygon in polygons:
            add_polygon(polygon)

    plt.xlabel('x')
    plt.ylabel('y', rotation=0)
    plt.title(title)

    # Check if any line has a label and add legend if true
    has_label = False  # Initialize has_label to a default value

    if lines is not None:
        has_label = any(map(lambda line: isinstance(line.get('label'), str), lines))

    if has_label:
        plt.legend()

    plt.show()


def add_polygon(polygon: PolygonDict):
    coords = polygon['coordinates']
    x = coords[:, 0]  # Extract x coordinates
    y = coords[:, 1]  # Extract y coordinates

    # Close the polygon by adding the first point at the end
    x = np.append(x, x[0])
    y = np.append(y, y[0])

    point_labels = polygon['point_labels']
    line_color = polygon.get('color', 'blue')
    fill_color = polygon.get('color', 'lightblue')
    label = polygon.get('label', None)

    # Plot the polygon edges
    plt.plot(x, y, 'o-', color=line_color, markersize=5)

    # Plot point texts (don't include the repeated first point)
    for i, txt in enumerate(point_labels):
        plt.text(x[i] + POINT_TEXT_OFFSET, y[i] + POINT_TEXT_OFFSET, txt, fontsize=12)

    # Fill the polygon
    plt.fill(x, y, color=fill_color, alpha=0.5, label=label)


def plot_radian_in_unit_circle(radian: float, title: str):
    plt.figure(figsize=(6, 6))

    # Create a unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)

    # Plot the unit circle
    plt.plot(x, y)

    # Plot the angle
    x_angle = [0, np.cos(radian)]
    y_angle = [0, np.sin(radian)]
    plt.plot(x_angle, y_angle, label=f'Angle {radian} rad', color='red')

    # Plot the angle arc
    arc = np.linspace(0, radian, 100)
    x_arc = np.cos(arc)
    y_arc = np.sin(arc)
    plt.plot(x_arc, y_arc, color='red', linestyle='dotted')

    # Set equal scaling
    plt.axis('equal')

    # Add labels and legend
    plt.xlabel('x')
    plt.ylabel('y', rotation=0)

    # Add title
    plt.title(title)

    # Show the plot
    plt.show()


def get_segment_lines_by_peaks(x: ndarray[float], y: ndarray[float], threshold=None) -> list[LineDict]:
    """
    Identifies the local minima (valleys) and maxima (peaks) in the given y data,
    and segments the x and y data at these points. Each segment is then stored
    as a dictionary with keys 'x', 'y', and 'color'.
    Args:
        x (ndarray[float]): The x-coordinates of the data points.
        y (ndarray[float]): The y-coordinates of the data points.
    Returns:
        list[LineDict]: A list of dictionaries, each representing a segment of the
                        data with keys 'x' (x-coordinates), 'y' (y-coordinates), 
                        and 'color' (set to 'red').
    """

    # Find local minima (valleys) and maxima (peaks)
    peaks, _ = find_peaks(y, height=0, threshold=threshold)  # Find peaks (top of U-shape)
    valleys, _ = find_peaks(-y, height=0, threshold=threshold)  # Find valleys (bottom of U-shape)

    # Combine peak and valley indices as segment boundaries
    segment_boundaries = np.sort(np.concatenate((peaks, valleys)))

    # Split data at these boundaries
    x_segments = np.split(x, segment_boundaries)
    y_segments = np.split(y, segment_boundaries)

    # Plot each segment separately to avoid connecting lines
    lines: list[LineDict] = []
    for x_seg, y_seg in zip(x_segments, y_segments):
        line = {"x": x_seg, "y": y_seg}
        lines.append(line)

    return lines


def plot_polar_cartesian(title: str, curves: list[PolarCurveDict], limits: Optional[tuple[float, float]] = None):
    """
    Plot polar curves by converting to cartesian coordinates.
    This completely avoids matplotlib's polar plot auto-fill behavior.
    
    Args:
        title (str): Title of the plot
        curves (list[PolarCurveDict]): List of polar curves to plot
        limits (Optional[tuple[float, float]]): Optional radial limits as (r_min, r_max)
    """
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 8))

    # Make the plot square and centered
    ax.set_aspect('equal')  # Set up grid circles and angle lines to mimic polar plot
    max_r = 0
    for curve in curves:
        # Handle NaN values properly by using nanmax
        valid_r = curve['r'][~np.isnan(curve['r'])]
        if len(valid_r) > 0:
            max_r = max(max_r, np.max(np.abs(valid_r)))

    if limits:
        max_r = max(max_r, abs(limits[1]) if limits[1] else max_r)

    # Ensure we have a reasonable minimum max_r
    if max_r == 0:
        max_r = 5

    max_r = max_r + 1  # Add some padding# Draw grid circles and add radial labels
    radial_values = np.linspace(0, max_r, 6)[1:]  # Skip 0
    for i, r in enumerate(radial_values):
        circle = plt.Circle((0, 0), r, fill=False, color='gray', alpha=0.3, linewidth=0.5)
        ax.add_patch(circle)

        # Add radial labels along the positive x-axis
        if r > 0:
            ax.text(r, 0.1, f'{r:.1f}', ha='center', va='bottom', fontsize=9, color='gray')  # Draw angle lines for the main directions only
    main_angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]  # 0°, 90°, 180°, 270°
    for angle in main_angles:
        x_line = [0, max_r * np.cos(angle)]
        y_line = [0, max_r * np.sin(angle)]
        ax.plot(x_line, y_line, color='gray', alpha=0.5, linewidth=0.8)  # Add theta tick labels at the cardinal directions
    label_radius = max_r + 0.3  # Position labels slightly outside the grid
    theta_labels = [(0, label_radius * np.cos(0), label_radius * np.sin(0), '0'),
                    (np.pi / 2, label_radius * np.cos(np.pi / 2), label_radius * np.sin(np.pi / 2), '$\\frac{\\pi}{2}$'),
                    (np.pi, label_radius * np.cos(np.pi), label_radius * np.sin(np.pi), '$\\pi$'),
                    (3 * np.pi / 2, label_radius * np.cos(3 * np.pi / 2), label_radius * np.sin(3 * np.pi / 2), '$\\frac{3\\pi}{2}$')]

    for angle, x_pos, y_pos, label_text in theta_labels:
        ax.text(x_pos, y_pos, label_text, ha='center', va='center', fontsize=11, fontweight='bold')

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    for curve in curves:
        theta = curve['theta']
        r = curve['r']
        line_color = curve.get('color', 'blue')
        linestyle = curve.get('linestyle', '-')
        label = curve.get('label', None)  # Convert polar to cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Remove NaN values to avoid broken line segments
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        if np.any(valid_mask):
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]

            # For curves with gaps (like lemniscates), we need to split at NaN boundaries
            # to avoid connecting distant points
            if not np.all(valid_mask):
                # Find groups of consecutive valid points
                diff = np.diff(np.concatenate(([False], valid_mask, [False])).astype(int))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]

                # Plot each continuous segment separately
                for start, end in zip(starts, ends):
                    x_segment = x[start:end]
                    y_segment = y[start:end]
                    ax.plot(x_segment,
                            y_segment,
                            color=line_color,
                            linestyle=linestyle,
                            label=label if start == starts[0] else None,
                            linewidth=1.5)
            else:
                # Plot the entire curve normally
                ax.plot(x_valid, y_valid, color=line_color, linestyle=linestyle, label=label, linewidth=1.5)
        else:
            # All values are NaN, skip plotting
            pass

        # Plot points if specified
        points = curve.get('points', [])
        for point in points:
            theta_val = point['x']  # theta is stored in x
            r_val = point['y']  # r is stored in y
            equal = point['equal'] if 'equal' in point else True
            point_label = point.get('label', None)
            point_color = line_color if equal else 'white'
            marker_edge_color = None if equal else line_color

            # Convert point to cartesian
            x_point = r_val * np.cos(theta_val)
            y_point = r_val * np.sin(theta_val)

            ax.plot(x_point, y_point, 'o', color=point_color, markersize=5, markeredgewidth=1, markeredgecolor=marker_edge_color)

            # Add label if specified
            if point_label:
                ax.annotate(point_label, xy=(x_point, y_point), xytext=(5, 5), textcoords='offset points')

    # Set limits
    if limits:
        ax.set_xlim(-limits[1], limits[1])
        ax.set_ylim(-limits[1], limits[1])
    else:
        ax.set_xlim(-max_r, max_r)
        ax.set_ylim(-max_r, max_r)

    # Add origin marker
    ax.plot(0, 0, 'ko', markersize=3)  # Add title and legend
    # Center the title at the top with enough space to avoid π/2 label
    plt.suptitle(title, x=0.5, y=0.95, ha='center', va='top', fontsize=14, fontweight='bold')
    has_label = any(curve.get('label') for curve in curves)
    if has_label:
        plt.legend(loc='upper right')

    plt.show()


def plot_polar_equation(title: str, equation, n_points=1000, show_vertices=True, min_vertex_distance=0.05, **kwargs):
    """
    Plot a polar equation of the form r = f(theta).
    
    Args:
        title (str): Title of the plot
        equation (callable): Function that takes theta and returns r
        n_points (int): Number of points to plot
        show_vertices (bool): If True, find and mark vertices of the curve
        min_vertex_distance (float): Minimum distance between vertices to avoid duplicates
        **kwargs: Additional arguments to pass to the plot function (color, linestyle, label)
    """# Generate theta values
    theta = np.linspace(0, 2 * np.pi, n_points)
    r = equation(theta)
    r = np.broadcast_to(r, theta.shape)

    # Create curve dictionary
    curve = {
        'theta': theta,
        'r': r,
        'color': kwargs.get('color', 'blue'),
        'linestyle': kwargs.get('linestyle', '-'),
        'label': kwargs.get('label', None)
    }  # Add vertices if requested
    if show_vertices:  # Find vertices of the equation
        vertices = find_polar_vertices(equation, (0, 2 * np.pi), n_points, min_vertex_distance)

        vertex_points = []

        # Use a more sophisticated approach for duplicate detection that considers
        # both polar coordinates and their cartesian equivalents
        def cartesian_tuple(theta, r):
            """Convert to cartesian and round for duplicate detection"""
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            return (round(x, 1), round(y, 1))  # Use coarser precision for cartesian coordinates

        marked_cartesian_set = set()

        for theta_v, r_v in vertices:
            r_v_rounded = round(r_v, 2)

            # Convert to cartesian for duplicate detection
            cart_tuple = cartesian_tuple(theta_v, r_v)

            # Only add the point if we haven't already marked this cartesian location
            if cart_tuple not in marked_cartesian_set:
                point = {'x': theta_v, 'y': r_v, 'equal': True, 'label': f"({get_pi_fraction_string(theta_v)}, {r_v_rounded})"}
                vertex_points.append(point)
                marked_cartesian_set.add(cart_tuple)

        curve['points'] = vertex_points  # Print vertices information (but filter to only show unique cartesian points)
        unique_vertices = []
        seen_cartesian = set()
        for theta_v, r_v in vertices:
            cart_tuple = cartesian_tuple(theta_v, r_v)
            if cart_tuple not in seen_cartesian:
                unique_vertices.append((theta_v, r_v))
                seen_cartesian.add(cart_tuple)

        print(f"Found {len(unique_vertices)} vertices:")
        for i, (theta_v, r_v) in enumerate(unique_vertices):
            r_v_rounded = round(r_v, 2)
            print(f"Vertex {i+1}: θ = {get_pi_fraction_string(theta_v)}, r = {r_v_rounded}"
                 )  # Plot the polar curve using cartesian coordinates
    plot_polar_cartesian(title, [curve], None)
