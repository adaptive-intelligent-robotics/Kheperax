from typing import List

import dataclasses

from kheperax.simu_components.geoms import Pos, Segment
from kheperax.tasks.maps import KHERPERAX_MAZES
from kheperax.simu_components.maze import Maze
from kheperax.simu_components.posture import Posture
from kheperax.simu_components.robot import Robot
from kheperax.tasks.target_task import TargetKheperaxConfig, DEFAULT_RESOLUTION


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


def flip_map(map, flip_x: bool, flip_y: bool) -> List[Segment]:
    base_segments = map["segments"]
    segments = [flip_segment(
        s,
        flip_x=flip_x,
        flip_y=flip_y
    ) for s in base_segments]
    return segments


@dataclasses.dataclass
class QuadKheperaxConfig(TargetKheperaxConfig):
    @classmethod
    def get_map(cls, map_name):
        map = KHERPERAX_MAZES[map_name]
        limits = ([-1., -1.], [1., 1.])
        segments = []
        for flip_x in [False, True]:
            for flip_y in [False, True]:
                segments += flip_map(map, flip_x=flip_x, flip_y=flip_y)

        robot = Robot.create_default_robot()
        # Start in the middle
        robot = robot.replace(
            posture=Posture(0., 0., robot.posture.angle)
        )

        return TargetKheperaxConfig(
            episode_length=1000,
            mlp_policy_hidden_layer_sizes=(8,),
            resolution=DEFAULT_RESOLUTION,
            action_scale=0.025,
            maze=Maze.create_custom(
                limits=limits,
                segments_list=segments
            ),
            robot=robot,
            std_noise_wheel_velocities=0.0,
            target_pos=map["target_pos"],
            target_radius=map["target_radius"],
            limits=limits
        )
