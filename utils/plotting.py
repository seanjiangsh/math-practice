# Type definitions

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import math

from numpy import ndarray
from typing import TypedDict, Optional


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

        x_start = (limits['x'][0] // 5) * 5
        x_end = ((limits['x'][1] // 5) + 1) * 5
        y_start = (limits['y'][0] // 5) * 5
        y_end = ((limits['y'][1] // 5) + 1) * 5
        plt.gca().set_xticks(np.arange(x_start, x_end, 5), minor=False)
        plt.gca().set_yticks(np.arange(y_start, y_end, 5), minor=False)
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
