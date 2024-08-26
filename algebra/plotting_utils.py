# Type definitions

import matplotlib.pyplot as plt
import numpy as np

from numpy import ndarray
from typing import TypedDict, Optional


class PointDict(TypedDict):
    x: float
    y: float
    equal: bool


class LineDict(TypedDict):
    x: ndarray[float]
    y: ndarray[float]
    color: Optional[str]
    linestyle: Optional[str]
    label: Optional[str]
    points: Optional[list[PointDict]]


class LimitDict(TypedDict):
    x: tuple[float, float]
    y: tuple[float, float]


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
