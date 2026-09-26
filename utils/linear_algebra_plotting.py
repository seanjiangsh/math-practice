import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from typing import Optional, TypedDict

from plotting import setup_plot


class LimitDict(TypedDict):
    x: tuple[float, float]
    y: tuple[float, float]


class Limit3DDict(TypedDict):
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]


Vector2D = tuple[float, float]
Vector3D = tuple[float, float, float]
Vector = Vector2D | Vector3D
PlotLimits = LimitDict | Limit3DDict


def _auto_limits(vectors: dict[str, Vector2D], padding: float = 1.0) -> LimitDict:
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


def _auto_limits_tip_to_tail(vectors: dict[str, Vector2D], padding: float = 1.0) -> LimitDict:
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
    vectors: dict[str, Vector],
    title: str = 'Basic Vectors',
    limits: Optional[PlotLimits] = None,
    tip_to_tail: bool = False,
    new_figure: bool = True,
    setup_axes: bool = True,
    show: bool = True,
    figsize: tuple[float, float] = (7, 7),
) -> dict[str, Vector]:
    """Plot named vectors from the origin or in tip-to-tail order and return endpoints."""
    if not vectors:
        raise ValueError('vectors must not be empty')

    dims = {len(v) for v in vectors.values()}
    if len(dims) != 1:
        raise ValueError('all vectors must have the same dimension (2D or 3D)')

    dim = dims.pop()
    if dim not in {2, 3}:
        raise ValueError('only 2D or 3D vectors are supported')

    if tip_to_tail and dim == 3:
        raise ValueError('tip_to_tail is currently supported for 2D vectors only')

    if dim == 3:
        if limits is not None and 'z' not in limits:
            raise ValueError('3D limits must include x, y, and z bounds')

        if limits is None:
            xs = [0.0] + [v[0] for v in vectors.values()]
            ys = [0.0] + [v[1] for v in vectors.values()]
            zs = [0.0] + [v[2] for v in vectors.values()]
            effective_limits_3d: Limit3DDict = {
                'x': (float(np.floor(min(xs) - 1.0)), float(np.ceil(max(xs) + 1.0))),
                'y': (float(np.floor(min(ys) - 1.0)), float(np.ceil(max(ys) + 1.0))),
                'z': (float(np.floor(min(zs) - 1.0)), float(np.ceil(max(zs) + 1.0))),
            }
        else:
            effective_limits_3d = limits  # type: ignore[assignment]

        if not new_figure:
            raise ValueError('new_figure=False is not supported for 3D vector plots')

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Overlay reference axes through the origin for easier 3D orientation.
        x_limits = effective_limits_3d['x']
        y_limits = effective_limits_3d['y']
        z_limits = effective_limits_3d['z']
        ax.plot([x_limits[0], x_limits[1]], [0, 0], [0, 0], color='black', linewidth=1, linestyle='--', alpha=0.7)
        ax.plot([0, 0], [y_limits[0], y_limits[1]], [0, 0], color='black', linewidth=1, linestyle='--', alpha=0.7)
        ax.plot([0, 0], [0, 0], [z_limits[0], z_limits[1]], color='black', linewidth=1, linestyle='--', alpha=0.7)

        color_cycle = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink']
        result_endpoints_3d: dict[str, Vector3D] = {}

        for idx, (name, vector) in enumerate(vectors.items()):
            x_end, y_end, z_end = vector
            color = color_cycle[idx % len(color_cycle)]

            ax.quiver(0, 0, 0, x_end, y_end, z_end, color=color, arrow_length_ratio=0.12, linewidth=2)
            ax.scatter([x_end], [y_end], [z_end], color=color, s=25)
            ax.text(x_end, y_end, z_end, f"{name} = ({x_end}, {y_end}, {z_end})", color=color, fontsize=9)
            result_endpoints_3d[name] = (x_end, y_end, z_end)

        ax.set_xlim(effective_limits_3d['x'])
        ax.set_ylim(effective_limits_3d['y'])
        ax.set_zlim(effective_limits_3d['z'])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(title)
        ax.grid(True)
        if show:
            plt.show()

        return result_endpoints_3d

    vectors_2d = vectors  # type: ignore[assignment]

    if new_figure:
        plt.figure(figsize=figsize)

    if limits is not None:
        effective_limits = limits
    elif tip_to_tail:
        effective_limits = _auto_limits_tip_to_tail(vectors_2d)
    else:
        effective_limits = _auto_limits(vectors_2d)

    if setup_axes:
        setup_plot(effective_limits)
    elif limits is not None:
        plt.xlim(limits['x'])
        plt.ylim(limits['y'])

    color_cycle = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink']
    result_endpoints: dict[str, Vector2D] = {}
    start_x, start_y = 0.0, 0.0

    for idx, (name, (x_end, y_end)) in enumerate(vectors_2d.items()):
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
    if show:
        plt.show()

    return result_endpoints
