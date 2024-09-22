from __future__ import annotations

from typing import List, Tuple

import dataclasses
import jax.tree_util
from jax import numpy as jnp
from qdax.core.neuroevolution.networks.networks import MLP

from kheperax.simu.maze import Maze
from kheperax.simu.robot import Robot
from kheperax.envs.maze_maps import get_target_maze_map
from kheperax.envs.kheperax_state import KheperaxState
from kheperax.utils.rendering_tools import RenderingTools
from kheperax.envs.env import Env
from kheperax.envs.wrappers import EpisodeWrapper, TypeFixerWrapper
from kheperax.envs.scoring import create_kheperax_scoring_fn, get_final_state_desc

DEFAULT_RESOLUTION = (1024, 1024)


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
    def get_default(cls):
        return cls.get_default_for_map("standard")

    @classmethod
    def get_default_for_map(cls, map_name):
        maze_map = get_target_maze_map(map_name)
        return cls(
            episode_length=1000,
            mlp_policy_hidden_layer_sizes=(8,),
            resolution=DEFAULT_RESOLUTION,
            action_scale=0.025,
            maze=Maze.create(
                segments_list=maze_map.segments
            ),
            robot=Robot.create_default_robot(),
            std_noise_wheel_velocities=0.0,
            limits=((0., 0.), (1., 1.)),
            action_repeat=1,
        )

    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}


class KheperaxTask(Env):
    def __init__(self, kheperax_config: KheperaxConfig, **_):
        self.kheperax_config = kheperax_config

    @classmethod
    def create_default_task(cls,
                            kheperax_config: KheperaxConfig,
                            random_key,
                            ):
        env = cls(kheperax_config)
        env = EpisodeWrapper(env, kheperax_config.episode_length, action_repeat=kheperax_config.action_repeat)
        env = TypeFixerWrapper(env)

        # Init policy network
        policy_layer_sizes = kheperax_config.mlp_policy_hidden_layer_sizes + (env.action_size,)
        policy_network = MLP(
            layer_sizes=policy_layer_sizes,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            final_activation=jnp.tanh,
        )

        bd_extraction_fn = get_final_state_desc

        scoring_fn = create_kheperax_scoring_fn(
            env,
            policy_network,
            bd_extraction_fn,
            episode_length=kheperax_config.episode_length,
        )

        return env, policy_network, scoring_fn

    def get_xy_pos(self, robot: Robot):
        return jnp.asarray([robot.posture.x, robot.posture.y])

    def reset(self, random_key: jnp.ndarray) -> KheperaxState:
        robot = self.kheperax_config.robot

        random_key, subkey = jax.random.split(random_key)
        obs = self._get_obs(robot, self.kheperax_config.maze, random_key=subkey)
        reward = 0.
        done = False

        info = {
            "state_descriptor": self.get_xy_pos(robot),
        }

        random_key, subkey = jax.random.split(random_key)

        return KheperaxState(
            maze=self.kheperax_config.maze,
            robot=robot,
            obs=obs,
            reward=reward,
            done=done,
            info=info,
            random_key=subkey,
        )

    def _get_wheel_velocities(self, action, random_key):
        random_key, subkey = jax.random.split(random_key)
        noise_wheel_velocities = jax.random.normal(subkey, shape=action.shape) \
                                 * self.kheperax_config.std_noise_wheel_velocities

        scale_actions = self.kheperax_config.action_scale
        wheel_velocities = action * scale_actions
        wheel_velocities = wheel_velocities + noise_wheel_velocities
        wheel_velocities = jnp.clip(wheel_velocities, -scale_actions, scale_actions)

        return wheel_velocities

    def step(self, state: KheperaxState, action: jnp.ndarray) -> KheperaxState:
        random_key = state.random_key

        # actions should be between -1 and 1
        action = jnp.clip(action, -1., 1.)

        random_key, subkey = jax.random.split(random_key)
        wheel_velocities = self._get_wheel_velocities(action, subkey)

        new_robot, bumper_measures = state.robot.move(wheel_velocities[0], wheel_velocities[1], state.maze)

        random_key, subkey = jax.random.split(random_key)
        obs = self._get_obs(new_robot, state.maze, bumper_measures=bumper_measures, random_key=subkey)

        # reward penalizes high action values
        reward = -1. * jnp.power(jnp.linalg.norm(wheel_velocities), 2.)

        # only stop at the end of the episode
        done = False

        state.info["state_descriptor"] = self.get_xy_pos(new_robot)

        random_key, subkey = jax.random.split(random_key)
        new_random_key = subkey

        return state.replace(
            maze=state.maze,
            robot=new_robot,
            obs=obs,
            reward=reward,
            done=done,
            random_key=new_random_key,
        )

    def _get_obs(self, robot: Robot, maze: Maze, random_key, bumper_measures: jnp.ndarray = None) -> jnp.ndarray:
        random_key, subkey = jax.random.split(random_key)
        laser_measures = robot.laser_measures(maze, random_key=subkey)

        if bumper_measures is None:
            bumper_measures = robot.bumper_measures(maze)

        return jnp.concatenate([laser_measures, bumper_measures])

    @property
    def observation_size(self) -> int:
        number_bumpers = 2
        number_lasers = len(self.kheperax_config.robot.laser_angles)
        return number_bumpers + number_lasers

    @property
    def action_size(self) -> int:
        return 2

    @property
    def state_descriptor_length(self) -> int:
        return 2

    @property
    def behavior_descriptor_length(self) -> int:
        return 2

    @property
    def behavior_descriptor_limits(self) -> Tuple[List[float], List[float]]:
        return self.kheperax_config.limits

    @property
    def backend(self) -> str:
        return "Kheperax"

    def render(self, state: KheperaxState, ) -> jnp.ndarray:
        # WARNING: only consider the maze is in the unit square
        coeff_triangle = 3.
        image = jnp.zeros(self.kheperax_config.resolution, dtype=jnp.float32)
        image = RenderingTools.place_triangle(image,
                                              point_1=(
                                                  state.robot.posture.x + coeff_triangle * state.robot.radius * jnp.cos(
                                                      state.robot.posture.angle),
                                                  state.robot.posture.y + coeff_triangle * state.robot.radius * jnp.sin(
                                                      state.robot.posture.angle)),
                                              point_2=(state.robot.posture.x + state.robot.radius * jnp.cos(
                                                  state.robot.posture.angle - jnp.pi / 2),
                                                       state.robot.posture.y + state.robot.radius * jnp.sin(
                                                           state.robot.posture.angle - jnp.pi / 2)),
                                              point_3=(state.robot.posture.x + state.robot.radius * jnp.cos(
                                                  state.robot.posture.angle + jnp.pi / 2),
                                                       state.robot.posture.y + state.robot.radius * jnp.sin(
                                                           state.robot.posture.angle + jnp.pi / 2)),
                                              value=2.)

        image = RenderingTools.place_circle(image,
                                            center=(state.robot.posture.x, state.robot.posture.y),
                                            radius=state.robot.radius,
                                            value=1.)

        white = jnp.array([1., 1., 1.])
        blue = jnp.array([0., 0., 1.])
        red = jnp.array([1., 0., 0.])
        green = jnp.array([0., 1., 0.])
        magenta = jnp.array([0.5, 0., 1.])
        cyan = jnp.array([0., 1., 1.])
        yellow = jnp.array([1., 1., 0.])
        black = jnp.array([0., 0., 0.])

        image = RenderingTools.place_segments(image,
                                              state.maze.walls,
                                              value=5.)

        index_to_color = {
            0.: white,
            1.: blue,
            2.: magenta,
            5.: black,
        }

        def _map_colors(x):
            new_array = jnp.zeros((3,))
            for key, value in index_to_color.items():
                new_array = jax.lax.cond(jnp.isclose(x, key), lambda _: value, lambda _: new_array, operand=None)
            return new_array

        image = jax.vmap(jax.vmap(_map_colors))(image)

        return image
