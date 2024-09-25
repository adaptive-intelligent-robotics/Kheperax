from __future__ import annotations

import dataclasses
from typing import Any, Dict, Tuple

from kheperax.envs.maze_maps import get_target_maze_map
from kheperax.simu.maze import Maze
from kheperax.simu.robot import Robot


@dataclasses.dataclass
class KheperaxConfig:
    episode_length: int
    mlp_policy_hidden_layer_sizes: Tuple[int, ...]
    action_scale: float
    maze: Maze
    robot: Robot
    std_noise_wheel_velocities: float
    resolution: Tuple[int, int]
    limits: Tuple[Tuple[float, float], Tuple[float, float]]
    action_repeat: int

    @classmethod
    def get_default(cls) -> KheperaxConfig:
        return cls.get_default_for_map("standard")

    @classmethod
    def get_default_for_map(cls, map_name: str) -> KheperaxConfig:
        maze_map = get_target_maze_map(map_name)
        return cls(
            episode_length=1000,
            mlp_policy_hidden_layer_sizes=(8,),
            resolution=DEFAULT_RESOLUTION,
            action_scale=0.025,
            maze=Maze.create(segments_list=maze_map.segments),
            robot=Robot.create_default_robot(),
            std_noise_wheel_velocities=0.0,
            limits=((0.0, 0.0), (1.0, 1.0)),
            action_repeat=1,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }


DEFAULT_RESOLUTION = (1024, 1024)
