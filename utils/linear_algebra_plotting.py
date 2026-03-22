import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from typing import Optional, TypedDict

from plotting import setup_plot


class LimitDict(TypedDict):
    x: tuple[float, float]
    y: tuple[float, float]


def _auto_limits(vectors: dict[str, tuple[float, float]], padding: float = 1.0) -> LimitDict:
    """Compute plot limits that include origin and all vector endpoints."""
    xs = [0.0] + [v[0] for v in vectors.values()]
    ys = [0.0] + [v[1] for v in vectors.values()]

    x_min = min(xs) - padding
    x_max = max(xs) + padding
    y_min = min(ys) - padding
    y_max = max(ys) + padding

    return {
        'x': (np.floor(x_min), np.ceil(x_max)),
        'y': (np.floor(y_min), np.ceil(y_max)),
    }


def _auto_limits_tip_to_tail(vectors: dict[str, tuple[float, float]], padding: float = 1.0) -> LimitDict:
    """Compute limits for a tip-to-tail sketch using cumulative endpoints."""
    x_points = [0.0]
    y_points = [0.0]

    current_x = 0.0
    current_y = 0.0
    for dx, dy in vectors.values():
        current_x += dx
        current_y += dy
        x_points.append(current_x)
        y_points.append(current_y)

    x_min = min(x_points) - padding
    x_max = max(x_points) + padding
    y_min = min(y_points) - padding
    y_max = max(y_points) + padding

    return {
        'x': (np.floor(x_min), np.ceil(x_max)),
        'y': (np.floor(y_min), np.ceil(y_max)),
    }


def plot_basic_vectors(
    vectors: dict[str, tuple[float, float]],
    title: str = 'Basic Vectors',
    limits: Optional[LimitDict] = None,
    tip_to_tail: bool = False,
) -> dict[str, tuple[float, float]]:
    """Plot named vectors from the origin or in tip-to-tail order and return endpoints."""
    if not vectors:
        raise ValueError('vectors must not be empty')

    plt.figure(figsize=(7, 7))
    if limits is not None:
        effective_limits = limits
    elif tip_to_tail:
        effective_limits = _auto_limits_tip_to_tail(vectors)
    else:
        effective_limits = _auto_limits(vectors)

    setup_plot(effective_limits)

    color_cycle = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink']
    result_endpoints: dict[str, tuple[float, float]] = {}
    start_x, start_y = 0.0, 0.0

    for idx, (name, (x_end, y_end)) in enumerate(vectors.items()):
        color = color_cycle[idx % len(color_cycle)]
        end_x = start_x + x_end if tip_to_tail else x_end
        end_y = start_y + y_end if tip_to_tail else y_end

        arrow = FancyArrowPatch(
            (start_x, start_y) if tip_to_tail else (0, 0),
            (end_x, end_y),
            arrowstyle='-|>',
            color=color,
            mutation_scale=20,
            linewidth=2,
            zorder=4,
        )
        plt.gca().add_patch(arrow)

        plt.plot(end_x, end_y, 'o', color=color, markersize=5, zorder=5)
        plt.annotate(
            f"{name} = ({x_end}, {y_end})",
            xy=(end_x, end_y),
            xytext=(6, 6),
            textcoords='offset points',
            fontsize=10,
            color=color,
        )

        result_endpoints[name] = (end_x, end_y)
        if tip_to_tail:
            start_x, start_y = end_x, end_y

    plt.xlabel('x')
    plt.ylabel('y', rotation=0)
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    return result_endpoints
