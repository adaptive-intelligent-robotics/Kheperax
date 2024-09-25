from functools import partial
from typing import Callable, Tuple

import flax.linen as nn
import jax
from jax import numpy as jnp
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey

from kheperax.envs.env import Env
from kheperax.envs.kheperax_state import KheperaxState
from kheperax.envs.scoring import create_kheperax_scoring_fn, get_final_state_desc
from kheperax.envs.wrappers import EpisodeWrapper
from kheperax.tasks.target import TargetKheperaxConfig, TargetKheperaxTask


class FinalDistKheperaxTask(TargetKheperaxTask):
    """Kheperax task that only rewards the final distance to the target"""

    @classmethod
    def final_play_step_fn(
        cls,
        policy_network: nn.Module,
        env: Env,
        policy_params: Params,
        env_state: KheperaxState,
        key: RNGKey,
    ) -> Tuple[KheperaxState, QDTransition]:
        """
        Play an environment step and return the updated EnvState and the transition.

        Args: env_state: The state of the environment (containing for instance the
        actor joint positions and velocities, the reward...). policy_params: The
        parameters of policies/controllers. random_key: JAX random key.

        Returns:
            next_state: The updated environment state.
            transition: containing some information about the transition: observation,
                reward, next observation, policy action...
        """

        actions = policy_network.apply(policy_params, env_state.obs)

        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        # Use last reward only at last step, -1 everywhere else
        distance_reward = jnp.where(
            jnp.logical_and(
                next_state.done,
                jnp.logical_not(env_state.done),
            ),
            next_state.reward - 1,
            -1 * jnp.ones_like(next_state.reward),
        )

        final_reward = distance_reward

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=final_reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
        )

        return next_state, transition

    @classmethod
    def create_default_task(
        cls,
        kheperax_config: TargetKheperaxConfig,
        random_key: RNGKey,
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

        play_step_fn = partial(
            cls.final_play_step_fn,
            policy_network,
            env_wrapper,
        )

        scoring_fn = create_kheperax_scoring_fn(
            env_wrapper,
            policy_network,
            bd_extraction_fn,
            play_step_fn=play_step_fn,
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

        # Reward is the distance to the target * 100
        target_dist = jnp.linalg.norm(
            jnp.array(self.kheperax_config.target_pos)
            - jnp.array(self.get_xy_pos(new_robot))
        )
        reward = -100 * target_dist

        # done if the robot is in the target of if already done
        done = target_dist < self.kheperax_config.target_radius

        # Reward 0 if target is reached
        reward = jnp.where(
            done,
            0.0,
            reward,
        )

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
