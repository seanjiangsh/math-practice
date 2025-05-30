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


def plot_lines(title: str, lines: list[LineDict], fills: list[FillBetweenDict] = None, limits: LimitDict = None):
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

    # Plot fills
    if fills is not None:
        for fill in fills:
            x = fill['x']
            y1 = fill['y1']
            y2 = fill['y2']
            where = fill['where']
            fill_color = fill.get('color', 'lightblue')
            plt.fill_between(x, y1, y2, where, color=fill_color, alpha=0.5)

    plt.xlabel('x')
    plt.ylabel('y', rotation=0)
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
    x: ndarray[float]
    y: ndarray[float]
    color: Optional[str]
    points: Optional[list[str]]
    labels: Optional[str]


class PolygonDict(TypedDict):
    x: ndarray[float]
    y: ndarray[float]
    points: list[str]
    color: Optional[str]
    label: Optional[str]


POINT_TEXT_OFFSET = 0.3  # Adjust this value to shift horizontally


def plot_geometry(title: str, lines: list[GeometryLineDict] = None, polygons: list[PolygonDict] = None, limits: LimitDict = None):
    plt.figure(figsize=(6, 6))
    setup_plot(limits)

    if lines is not None:
        for line in lines:
            x = line['x']
            y = line['y']
            line_color = line.get('color', 'blue')
            labels = line.get('labels', None)
            plt.plot(x, y, 'o-', color=line_color, markersize=5, label=labels)

            # Plot point texts
            point_texts = line.get('points', [])
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
    x = np.append(polygon['x'], polygon['x'][0])
    y = np.append(polygon['y'], polygon['y'][0])
    points = polygon['points']
    line_color = polygon.get('color', 'blue')
    fill_color = polygon.get('color', 'lightblue')
    label = polygon.get('label', None)

    # Plot the polygon edges
    plt.plot(x, y, 'o-', color=line_color, markersize=5)

    # Plot point texts
    for i, txt in enumerate(points):
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


def plot_polar(title: str, curves: list[PolarCurveDict], limits: Optional[tuple[float, float]] = None):
    """
    Plot curves in polar coordinates using radians.
    
    Args:
        title (str): Title of the plot
        curves (list[PolarCurveDict]): List of polar curves to plot
        limits (Optional[tuple[float, float]]): Optional radial limits as (r_min, r_max)
    """
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(projection='polar')

    # Set to counterclockwise direction with east as zero location
    # ax.set_theta_zero_location('E')  # 0 radians at the right (east)
    # ax.set_theta_direction(-1)  # -1 for counterclockwise (mathematical convention)

    # Set custom radian labels in counterclockwise order
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4]
    labels = [
        '0', '$\\frac{\\pi}{4}$', '$\\frac{\\pi}{2}$', '$\\frac{3\\pi}{4}$', '$\\pi$', '$\\frac{5\\pi}{4}$', '$\\frac{3\\pi}{2}$',
        '$\\frac{7\\pi}{4}$'
    ]

    ax.set_xticks(angles)
    ax.set_xticklabels(labels)

    for curve in curves:
        theta = curve['theta']
        r = curve['r']
        line_color = curve.get('color', 'blue')
        linestyle = curve.get('linestyle', '-')
        label = curve.get('label', None)

        ax.plot(theta, r, color=line_color, linestyle=linestyle, label=label)

        # Plot points if specified
        points = curve.get('points', [])
        for point in points:
            theta_val = point['x']  # theta is stored in x
            r_val = point['y']  # r is stored in y
            equal = point['equal'] if 'equal' in point else True
            label = point.get('label', None)
            point_color = line_color if equal else 'white'
            marker_edge_color = None if equal else line_color

            ax.plot(theta_val, r_val, 'o', color=point_color, markersize=5, markeredgewidth=1, markeredgecolor=marker_edge_color)

            # Add label if specified
            if label:
                ax.annotate(label, xy=(theta_val, r_val), xytext=(5, 5), textcoords='offset points')

    # Set radial limits if specified
    if limits:
        ax.set_ylim(limits)

    # Add grid and labels
    ax.grid(True)
    plt.title(title)

    # Add legend if any curve has a label
    has_label = any(map(lambda curve: isinstance(curve.get('label'), str), curves))
    if has_label:
        plt.legend(loc='upper right')

    plt.show()


def plot_polar_equation(title: str,
                        equation,
                        theta_range=(0, 2 * np.pi),
                        n_points=1000,
                        limits: Optional[tuple[float, float]] = None,
                        show_vertices=True,
                        min_vertex_distance=0.05,
                        **kwargs):
    """
    Plot a polar equation of the form r = f(theta).
    
    Args:
        title (str): Title of the plot
        equation (callable): Function that takes theta and returns r
        theta_range (tuple): Range of theta values to plot (start, end)
        n_points (int): Number of points to plot
        limits (Optional[tuple[float, float]]): Optional radial limits as (r_min, r_max)
        show_vertices (bool): If True, find and mark vertices of the curve
        min_vertex_distance (float): Minimum distance between vertices to avoid duplicates
        **kwargs: Additional arguments to pass to the plot function (color, linestyle, label)
    """
    # Generate theta values
    theta = np.linspace(theta_range[0], theta_range[1], n_points)

    # Calculate r values
    r = equation(theta)

    # Create curve dictionary
    curve = {
        'theta': theta,
        'r': r,
        'color': kwargs.get('color', 'blue'),
        'linestyle': kwargs.get('linestyle', '-'),
        'label': kwargs.get('label', None)
    }

    # Add vertices if requested
    if show_vertices:
        # Find vertices of the equation
        vertices = find_polar_vertices(equation, theta_range, n_points, min_vertex_distance)

        # Add points for vertices
        vertex_points = []
        for theta_v, r_v in vertices:
            # Round r value for display
            r_v_rounded = round(r_v, 2)
            point = {
                'x': theta_v,  # theta is stored in x
                'y': r_v,  # r is stored in y (original value for accurate plotting)
                'equal': True,
                'label': f"({get_pi_fraction_string(theta_v)}, {r_v_rounded})"
            }
            vertex_points.append(point)

        curve['points'] = vertex_points

        # Print vertices information
        print(f"Found {len(vertices)} vertices:")
        for i, (theta_v, r_v) in enumerate(vertices):
            # Round r value for display
            r_v_rounded = round(r_v, 2)
            print(f"Vertex {i+1}: θ = {get_pi_fraction_string(theta_v)}, r = {r_v_rounded}")

    # Plot the polar curve
    plot_polar(title, [curve], limits)
