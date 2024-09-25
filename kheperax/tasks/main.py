from __future__ import annotations

from functools import partial
from typing import Callable, Generic, Tuple

import jax.tree_util
import numpy as np
from jax import numpy as jnp
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.custom_types import Action, Descriptor, ExtraScores, Fitness, Genotype, RNGKey
from typing_extensions import TypeVar

from kheperax.custom_types import KheperaxImage
from kheperax.envs.env import Env
from kheperax.envs.kheperax_state import KheperaxState
from kheperax.envs.scoring import create_kheperax_scoring_fn, get_final_state_desc
from kheperax.envs.wrappers import EpisodeWrapper
from kheperax.simu.geoms import Pos, Segment
from kheperax.simu.maze import Maze
from kheperax.simu.robot import Robot
from kheperax.tasks.config import KheperaxConfig
from kheperax.utils.rendering_tools import RenderingTools

ConfigT = TypeVar("ConfigT", bound=KheperaxConfig)


class KheperaxTask(Env, Generic[ConfigT]):
    def __init__(self, kheperax_config: ConfigT):
        self.kheperax_config = kheperax_config

    @classmethod
    def create_default_task(
        cls,
        kheperax_config: ConfigT,
        random_key: RNGKey,
    ) -> Tuple[
        EpisodeWrapper,
        MLP,
        Callable[[Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]],
    ]:
        env = cls(kheperax_config)
        ep_wrapper_env = EpisodeWrapper(
            env,
            kheperax_config.episode_length,
            action_repeat=kheperax_config.action_repeat,
        )

        # Init policy network
        policy_layer_sizes = kheperax_config.mlp_policy_hidden_layer_sizes + (
            ep_wrapper_env.action_size,
        )
        policy_network = MLP(
            layer_sizes=policy_layer_sizes,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            final_activation=jnp.tanh,
        )

        bd_extraction_fn = get_final_state_desc

        scoring_fn = create_kheperax_scoring_fn(
            ep_wrapper_env,
            policy_network,
            bd_extraction_fn,
            episode_length=kheperax_config.episode_length,
        )

        return ep_wrapper_env, policy_network, scoring_fn

    def get_xy_pos(self, robot: Robot) -> jax.Array:
        return jnp.asarray([robot.posture.x, robot.posture.y])

    def reset(self, random_key: jax.typing.ArrayLike) -> KheperaxState:
        robot = self.kheperax_config.robot

        random_key, subkey = jax.random.split(random_key)
        obs = self._get_obs(robot, self.kheperax_config.maze, random_key=subkey)
        reward = 0.0
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

    def _get_wheel_velocities(self, action: Action, random_key: RNGKey) -> jax.Array:
        random_key, subkey = jax.random.split(random_key)
        noise_wheel_velocities = (
            jax.random.normal(subkey, shape=action.shape)
            * self.kheperax_config.std_noise_wheel_velocities
        )

        scale_actions = self.kheperax_config.action_scale
        wheel_velocities = action * scale_actions
        wheel_velocities = wheel_velocities + noise_wheel_velocities
        wheel_velocities = jnp.clip(wheel_velocities, -scale_actions, scale_actions)

        return wheel_velocities

    def step(self, state: KheperaxState, action: jax.typing.ArrayLike) -> KheperaxState:
        random_key = state.random_key

        # actions should be between -1 and 1
        action = jnp.clip(action, -1.0, 1.0)

        random_key, subkey = jax.random.split(random_key)
        wheel_velocities = self._get_wheel_velocities(action, subkey)

        new_robot, bumper_measures = state.robot.move(
            wheel_velocities[0], wheel_velocities[1], state.maze
        )

        random_key, subkey = jax.random.split(random_key)
        obs = self._get_obs(
            new_robot, state.maze, bumper_measures=bumper_measures, random_key=subkey
        )

        # reward penalizes high action values
        reward = -1.0 * jnp.power(jnp.linalg.norm(wheel_velocities), 2.0)

        # only stop at the end of the episode
        done = False

        state.info["state_descriptor"] = self.get_xy_pos(new_robot)

        random_key, subkey = jax.random.split(random_key)
        new_random_key = subkey

        return state.replace(  # type: ignore
            maze=state.maze,
            robot=new_robot,
            obs=obs,
            reward=reward,
            done=done,
            random_key=new_random_key,
        )

    def _get_obs(
        self,
        robot: Robot,
        maze: Maze,
        random_key: RNGKey,
        bumper_measures: jax.typing.ArrayLike = None,
    ) -> jax.Array:
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
    def behavior_descriptor_limits(
        self,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return self.kheperax_config.limits

    @property
    def backend(self) -> str:
        return "Kheperax"

    def create_image(
        self,
        state: KheperaxState,
    ) -> KheperaxImage:
        # WARNING: only consider the maze is in the unit square
        image = jnp.zeros(self.kheperax_config.resolution, dtype=jnp.float32)

        # Walls
        image = RenderingTools.place_segments(
            self.kheperax_config, image, state.maze.walls, value=5.0
        )

        return image

    def add_robot(self, image: KheperaxImage, state: KheperaxState) -> KheperaxImage:
        coeff_triangle = 3.0
        image = RenderingTools.place_triangle(
            self.kheperax_config,
            image,
            point_1=(
                state.robot.posture.x
                + coeff_triangle
                * state.robot.radius
                * jnp.cos(state.robot.posture.angle),
                state.robot.posture.y
                + coeff_triangle
                * state.robot.radius
                * jnp.sin(state.robot.posture.angle),
            ),
            point_2=(
                state.robot.posture.x
                + state.robot.radius * jnp.cos(state.robot.posture.angle - jnp.pi / 2),
                state.robot.posture.y
                + state.robot.radius * jnp.sin(state.robot.posture.angle - jnp.pi / 2),
            ),
            point_3=(
                state.robot.posture.x
                + state.robot.radius * jnp.cos(state.robot.posture.angle + jnp.pi / 2),
                state.robot.posture.y
                + state.robot.radius * jnp.sin(state.robot.posture.angle + jnp.pi / 2),
            ),
            value=2.0,
        )

        image = RenderingTools.place_circle(
            self.kheperax_config,
            image,
            center=(state.robot.posture.x, state.robot.posture.y),
            radius=state.robot.radius,
            value=1.0,
        )
        return image

    def add_lasers(self, image: KheperaxImage, state: KheperaxState) -> jax.Array:
        robot = state.robot
        maze = state.maze
        laser_measures = robot.laser_measures(maze, random_key=state.random_key)

        # Replace -1 by the max range, make yellow
        laser_colors = jnp.where(
            jnp.isclose(laser_measures, -1.0),
            6.0,
            4.0,
        )
        laser_measures = jnp.where(
            jnp.isclose(laser_measures, -1.0),
            robot.laser_ranges,
            laser_measures,
        )
        laser_relative_angles = robot.laser_angles
        robot_angle = robot.posture.angle
        laser_angles = laser_relative_angles + robot_angle

        robot_pos = Pos.from_posture(robot.posture)
        # segments = []
        for laser_measure, laser_angle, laser_color in zip(
            laser_measures, laser_angles, laser_colors
        ):
            laser_x = robot_pos.x + laser_measure * jnp.cos(laser_angle)
            laser_y = robot_pos.y + laser_measure * jnp.sin(laser_angle)
            laser_pos = Pos(x=laser_x, y=laser_y)
            laser_segment = Segment(robot_pos, laser_pos)
            # segments.append(laser_segment)

            segments = jax.tree_util.tree_map(
                lambda *x: jnp.asarray(x, dtype=jnp.float32), *[laser_segment]
            )

            image = RenderingTools.place_segments(
                self.kheperax_config, image, segments, value=laser_color
            )

        return image

    def render(
        self,
        state: KheperaxState,
    ) -> np.ndarray:
        image = self.create_image(state)
        image = self.add_robot(image, state)
        image = self.render_rgb_image(image)
        return image

    def render_rgb_image(self, image: KheperaxImage, flip: bool = False) -> np.ndarray:
        # Add 2 empty channels
        empty = -jnp.inf + jnp.ones(image.shape[:2])
        rgb_image = jnp.stack([image, empty, empty], axis=-1)

        white = jnp.array([1.0, 1.0, 1.0])
        blue = jnp.array([0.0, 0.0, 1.0])
        red = jnp.array([1.0, 0.0, 0.0])
        green = jnp.array([0.0, 1.0, 0.0])
        magenta = jnp.array([0.5, 0.0, 1.0])
        cyan = jnp.array([0.0, 1.0, 1.0])
        yellow = jnp.array([1.0, 1.0, 0.0])
        black = jnp.array([0.0, 0.0, 0.0])

        index_to_color = {
            0: white,
            1: blue,
            2: magenta,
            3: green,
            4: red,
            5: black,
            6: yellow,
            7: cyan,
        }

        def colorize(x: jax.typing.ArrayLike, _color_id: int, _rgb: float) -> jax.Array:
            return jnp.where(jnp.isclose(x[0], _color_id), _rgb * 255, x)

        for color_id, rgb in index_to_color.items():
            colorize_color_fn = partial(colorize, _color_id=color_id, _rgb=rgb)
            rgb_image = jax.vmap(jax.vmap(colorize_color_fn))(rgb_image)

        rgb_image = jnp.array(rgb_image).astype("uint8")

        if flip:
            rgb_image = rgb_image[::-1, :, :]

        rgb_image = np.array(rgb_image)
        return rgb_image
