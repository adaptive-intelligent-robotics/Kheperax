from __future__ import annotations

import dataclasses
from typing import Callable, Tuple

import jax
from jax import numpy as jnp
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from kheperax.custom_types import KheperaxImage
from kheperax.envs.kheperax_state import KheperaxState
from kheperax.envs.maze_maps import get_target_maze_map
from kheperax.envs.scoring import create_kheperax_scoring_fn, get_final_state_desc
from kheperax.envs.wrappers import EpisodeWrapper
from kheperax.tasks.config import KheperaxConfig
from kheperax.tasks.main import KheperaxTask
from kheperax.utils.rendering_tools import RenderingTools


@dataclasses.dataclass
class TargetKheperaxConfig(KheperaxConfig):
    target_pos: tuple
    target_radius: float

    @classmethod
    def get_default(cls) -> TargetKheperaxConfig:
        return cls.get_default_for_map("standard")

    @classmethod
    def get_default_for_map(cls, map_name: str) -> TargetKheperaxConfig:
        maze_map = get_target_maze_map(map_name)
        parent_config = KheperaxConfig.get_default_for_map(map_name)

        return cls(
            **parent_config.to_dict(),
            target_pos=maze_map.target_pos,
            target_radius=maze_map.target_radius,
        )


class TargetKheperaxTask(KheperaxTask[TargetKheperaxConfig]):
    def __init__(self, kheperax_config: TargetKheperaxConfig):
        super().__init__(kheperax_config)

    @classmethod
    def create_default_task(
        cls, kheperax_config: TargetKheperaxConfig, random_key: RNGKey
    ) -> Tuple[
        EpisodeWrapper,
        MLP,
        Callable[[Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]],
    ]:
        env = cls(kheperax_config)
        env_wrapper = EpisodeWrapper(
            env,
            kheperax_config.episode_length,
            action_repeat=kheperax_config.action_repeat,
        )

        # Init policy network
        policy_layer_sizes = kheperax_config.mlp_policy_hidden_layer_sizes + (
            env_wrapper.action_size,
        )
        policy_network = MLP(
            layer_sizes=policy_layer_sizes,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            final_activation=jnp.tanh,
        )

        bd_extraction_fn = get_final_state_desc

        scoring_fn = create_kheperax_scoring_fn(
            env_wrapper,
            policy_network,
            bd_extraction_fn,
            episode_length=kheperax_config.episode_length,
        )

        return env_wrapper, policy_network, scoring_fn

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

        # Reward is the distance to the target
        target_dist = jnp.linalg.norm(
            jnp.array(self.kheperax_config.target_pos)
            - jnp.array(self.get_xy_pos(new_robot))
        )
        reward = -1.0 * target_dist

        # done if the robot is in the target
        done = target_dist < self.kheperax_config.target_radius

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

    def create_image(
        self,
        state: KheperaxState,
    ) -> KheperaxImage:
        # WARNING: only consider the maze is in the unit square
        image = jnp.zeros(self.kheperax_config.resolution, dtype=jnp.float32)

        # Target
        image = RenderingTools.place_circle(
            self.kheperax_config,
            image,
            center=(
                self.kheperax_config.target_pos[0],
                self.kheperax_config.target_pos[1],
            ),
            radius=self.kheperax_config.target_radius,
            value=3.0,
        )

        # Walls
        image = RenderingTools.place_segments(
            self.kheperax_config, image, state.maze.walls, value=5.0
        )

        return image
