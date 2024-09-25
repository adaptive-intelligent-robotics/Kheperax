import dataclasses
from typing import List, Tuple

from kheperax.simu.geoms import Pos, Segment


@dataclasses.dataclass
class MazeMap:
    segments: list
    target_pos: tuple
    target_radius: float


KHEPERAX_MAZES = {}


def register_target_maze(
    name: str,
    segments: List[Segment],
    target_pos: Tuple[float, float],
    target_radius: float,
) -> None:
    KHEPERAX_MAZES[name] = MazeMap(segments, target_pos, target_radius)


def get_target_maze_map(name: str) -> MazeMap:
    return KHEPERAX_MAZES[name]


# Standard Kheperax maze
register_target_maze(
    "standard",
    segments=[
        Segment(Pos(0.25, 0.25), Pos(0.25, 0.75)),  # Vertical middle
        Segment(Pos(0.14, 0.45), Pos(0.0, 0.65)),  # Obstacle top left
        Segment(Pos(0.25, 0.75), Pos(0.0, 0.8)),  # Wall to target
        Segment(Pos(0.25, 0.75), Pos(0.66, 0.875)),  # Wall top right
        Segment(Pos(0.355, 0.0), Pos(0.525, 0.185)),  # Obstacle bottom right
        Segment(Pos(0.25, 0.5), Pos(0.75, 0.215)),  # Funnel bottom
        Segment(Pos(1.0, 0.25), Pos(0.435, 0.55)),  # Funnel top
        # For quad
        Segment(Pos(0.0, 0.8), Pos(0.0, 1.0)),  # Wall top left
        Segment(Pos(0.355, 0.0), Pos(1.0, 0.0)),  # Wall bottom right
    ],
    target_pos=(0.15, 0.9),
    target_radius=0.05,
)

# Pointmaze maze
y_bottom = 0.25
y_top = 0.6
width = 0.75
register_target_maze(
    "pointmaze",
    segments=[
        Segment(Pos(0.0, y_bottom), Pos(width, y_bottom)),  # bottom wall
        Segment(Pos(1 - width, y_top), Pos(1.0, y_top)),  # top wall
        # For quad
        Segment(Pos(0.0, y_bottom), Pos(0.0, 1.0)),  # vertical left
    ],
    target_pos=(0.9, 0.9),
    target_radius=0.05,
)

# Curiosity ES mazes
left_wall = 0.2
start_left_max_x = 0.8
start_left_y = [0.8, 0.4]
start_right_min_x = 0.4
start_right_y = [0.6, 0.2]

register_target_maze(
    "snake",
    segments=[
        Segment(Pos(left_wall, 0.0), Pos(left_wall, 0.8)),  # left wall
        Segment(Pos(0.0, 0.2), Pos(0.0, 1.0)),  # Left wall for quad
        Segment(Pos(0.2, 0.0), Pos(1.0, 0.0)),  # bottom wall
        *[
            Segment(Pos(left_wall, y), Pos(start_left_max_x, y)) for y in start_left_y
        ],  # walls starting left
        *[
            Segment(Pos(start_right_min_x, y), Pos(1.0, y)) for y in start_right_y
        ],  # walls starting right
    ],
    target_pos=(0.9, 0.1),
    target_radius=0.05,
)
