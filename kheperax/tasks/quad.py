import dataclasses
from typing import List, TypeVar

import jax
import jax.numpy as jnp

from kheperax.envs.maze_maps import MazeMap
from kheperax.simu.geoms import Pos, Segment
from kheperax.simu.maze import Maze
from kheperax.simu.posture import Posture
from kheperax.tasks.config import KheperaxConfig


def flip_segment(segment: Segment, flip_x: bool, flip_y: bool) -> Segment:
    """
    Flip a segment in the x and/or y-axis.
    """
    p1 = segment.p1
    p2 = segment.p2
    if flip_x:
        p1 = Pos(p1.x, -p1.y)
        p2 = Pos(p2.x, -p2.y)
    if flip_y:
        p1 = Pos(-p1.x, p1.y)
        p2 = Pos(-p2.x, p2.y)
    return Segment(p1, p2)


def flip_map(maze_map: MazeMap, flip_x: bool, flip_y: bool) -> List[Segment]:
    base_segments = maze_map.segments
    segments = [flip_segment(s, flip_x=flip_x, flip_y=flip_y) for s in base_segments]
    return segments


T_KheperaxConfig = TypeVar("T_KheperaxConfig", bound=KheperaxConfig)


def make_quad_config(kheperax_config: T_KheperaxConfig) -> T_KheperaxConfig:
    """
    Convert a KheperaxConfig to a
    Args:
        kheperax_config:

    Returns:

    """
    inside_walls = kheperax_config.maze.inside_walls
    new_limits = ((-1.0, -1.0), (1.0, 1.0))

    # Flip the map in all possible ways
    all_segments_with_flipped = []
    for flip_x in [False, True]:
        for flip_y in [False, True]:
            all_segments_with_flipped.append(
                flip_segment(inside_walls, flip_x=flip_x, flip_y=flip_y)
            )
    all_segments_with_flipped = jax.tree_util.tree_map(
        lambda *x: jnp.concatenate(x), *all_segments_with_flipped
    )

    new_maze = Maze(
        inside_walls=all_segments_with_flipped,
        border_walls=Maze.make_border_walls(new_limits),
    )

    # Robot starts in the middle
    new_robot = kheperax_config.robot
    new_robot = new_robot.replace(posture=Posture(0.0, 0.0, new_robot.posture.angle))

    return dataclasses.replace(
        kheperax_config,
        maze=new_maze,
        robot=new_robot,
        limits=new_limits,
    )
