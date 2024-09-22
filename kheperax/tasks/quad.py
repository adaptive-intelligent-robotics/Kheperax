import dataclasses
from typing import List

from kheperax.simu_components.geoms import Pos, Segment
from kheperax.simu_components.maze import Maze
from kheperax.simu_components.posture import Posture
from kheperax.simu_components.robot import Robot
from kheperax.envs.maze_maps import get_target_maze_map, TargetMazeMap
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


@dataclasses.dataclass
class QuadKheperaxConfig(TargetKheperaxConfig):
    @classmethod
    def get_default_for_map(cls, map_name):
        maze_map = get_target_maze_map(map_name)
        limits = ((-1., -1.), (1., 1.))
        segments = []
        for flip_x in [False, True]:
            for flip_y in [False, True]:
                segments += flip_map(maze_map, flip_x=flip_x, flip_y=flip_y)

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
            maze=Maze.create(
                segments_list=segments,
                limits=limits,
            ),
            robot=robot,
            std_noise_wheel_velocities=0.0,
            target_pos=maze_map.target_pos,
            target_radius=maze_map.target_radius,
            limits=limits
        )
