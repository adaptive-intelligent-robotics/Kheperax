import dataclasses
from typing import List, TypeVar

import jax
import jax.numpy as jnp

from kheperax.envs.maze_maps import get_target_maze_map, TargetMazeMap
from kheperax.simu.geoms import Pos, Segment
from kheperax.simu.maze import Maze
from kheperax.simu.posture import Posture
from kheperax.tasks.main import KheperaxConfig
from kheperax.tasks.target import TargetKheperaxConfig


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


def flip_map(maze_map: TargetMazeMap, flip_x: bool, flip_y: bool) -> List[Segment]:
    base_segments = maze_map.segments
    segments = [flip_segment(
        s,
        flip_x=flip_x,
        flip_y=flip_y
    ) for s in base_segments]
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
    new_limits = ((-1., -1.), (1., 1.))

    # Flip the map in all possible ways
    all_segments_with_flipped = []
    for flip_x in [False, True]:
        for flip_y in [False, True]:
            all_segments_with_flipped.append(flip_segment(inside_walls, flip_x=flip_x, flip_y=flip_y))
    all_segments_with_flipped = jax.tree_util.tree_map(lambda *x: jnp.concatenate(x), *all_segments_with_flipped)

    new_maze = Maze(
        inside_walls=all_segments_with_flipped,
        border_walls=Maze.make_border_walls(new_limits)
    )

    # Robot starts in the middle
    new_robot = kheperax_config.robot
    new_robot = new_robot.replace(
        posture=Posture(0., 0., new_robot.posture.angle)
    )

    return dataclasses.replace(kheperax_config,
                               maze=new_maze,
                               robot=new_robot,
                               limits=new_limits,
                               )


@dataclasses.dataclass
class QuadKheperaxConfig(TargetKheperaxConfig):
    @classmethod
    def get_default(cls):
        return cls.get_default_for_map("standard")

    @classmethod
    def get_default_for_map(cls, map_name):
        parent_config = TargetKheperaxConfig.get_default_for_map(map_name)

        maze_map = get_target_maze_map(map_name)
        new_limits = ((-1., -1.), (1., 1.))
        all_segments_with_flipped = []
        for flip_x in [False, True]:
            for flip_y in [False, True]:
                all_segments_with_flipped += flip_map(maze_map, flip_x=flip_x, flip_y=flip_y)

        robot = parent_config.robot
        # Start in the middle
        robot = robot.replace(
            posture=Posture(0., 0., robot.posture.angle)
        )

        return TargetKheperaxConfig(
            episode_length=parent_config.episode_length,
            mlp_policy_hidden_layer_sizes=parent_config.mlp_policy_hidden_layer_sizes,
            resolution=parent_config.resolution,
            action_scale=parent_config.action_scale,
            maze=Maze.create(
                segments_list=all_segments_with_flipped,
                limits=new_limits,
            ),
            robot=robot,
            std_noise_wheel_velocities=parent_config.std_noise_wheel_velocities,
            target_pos=parent_config.target_pos,
            target_radius=parent_config.target_radius,
            limits=new_limits,
            action_repeat=parent_config.action_repeat,
        )
