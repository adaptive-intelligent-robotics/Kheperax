from __future__ import annotations

from functools import partial
from typing import Callable, Generic, Tuple

import jax.tree_util
from jax import numpy as jnp
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.custom_types import Action, Descriptor, ExtraScores, Fitness, Genotype, RNGKey
from typing_extensions import TypeVar

from kheperax.custom_types import KheperaxImage
from kheperax.envs.env import Env
from kheperax.envs.kheperax_state import KheperaxState
from kheperax.envs.scoring import create_kheperax_scoring_fn, get_final_state_desc
from kheperax.envs.wrappers import EpisodeWrapper
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

    def render(
        self,
        state: KheperaxState,
    ) -> KheperaxImage:
        # WARNING: only consider the maze is in the unit square
        coeff_triangle = 3.0
        image = jnp.zeros(self.kheperax_config.resolution, dtype=jnp.float32)
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

        white = jnp.array([1.0, 1.0, 1.0])
        blue = jnp.array([0.0, 0.0, 1.0])
        # red = jnp.array([1.0, 0.0, 0.0])
        # green = jnp.array([0.0, 1.0, 0.0])
        magenta = jnp.array([0.5, 0.0, 1.0])
        # cyan = jnp.array([0.0, 1.0, 1.0])
        # yellow = jnp.array([1.0, 1.0, 0.0])
        black = jnp.array([0.0, 0.0, 0.0])

        image = RenderingTools.place_segments(
            self.kheperax_config, image, state.maze.walls, value=5.0
        )

        index_to_color = {
            0.0: white,
            1.0: blue,
            2.0: magenta,
            5.0: black,
        }

        def _map_colors(x: jax.typing.ArrayLike) -> jax.Array:
            new_array = jnp.zeros((3,))

            def identity_fn(_: jax.typing.ArrayLike, x: float) -> float:
                return x

            for index_color, value_color in index_to_color.items():
                new_array = jax.lax.cond(
                    jnp.isclose(x, index_color),
                    partial(identity_fn, value_color),
                    partial(identity_fn, new_array),
                    operand=None,
                )
            return new_array

        image = jax.vmap(jax.vmap(_map_colors))(image)

        return image
