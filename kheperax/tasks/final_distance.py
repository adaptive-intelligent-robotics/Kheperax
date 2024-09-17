import jax
from jax import numpy as jnp
import brax.v1.envs
import flax.linen as nn
from typing import Callable, Tuple

from qdax.tasks.brax_envs import create_brax_scoring_fn
from qdax.environments.bd_extractors import (
    get_final_xy_position,
)

from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.custom_types import (
    EnvState,
    Params,
    RNGKey,
)


from kheperax.tasks.main_task import KheperaxConfig, KheperaxState
from kheperax.utils.type_fixer_wrapper import TypeFixerWrapper
from kheperax.tasks.target import TargetKheperaxTask


def make_final_policy_network_play_step_fn_brax(  # TODO: ?
    env: brax.v1.envs.Env,
    policy_network: nn.Module,
) -> Callable[
    [EnvState, Params, RNGKey], Tuple[EnvState, Params, RNGKey, QDTransition]
]:
    """
    Creates a function that when called, plays a step of the environment.

    Args:
        env: The BRAX environment.
        policy_network:  The policy network structure used for creating and evaluating
            policy controllers.

    Returns:
        default_play_step_fn: A function that plays a step of the environment.
    """
    def final_play_step_fn(
        env_state: EnvState,
        policy_params: Params,
        random_key: RNGKey,
    ) -> Tuple[EnvState, Params, RNGKey, QDTransition]:
        """
        Play an environment step and return the updated EnvState and the transition.

        Args: env_state: The state of the environment (containing for instance the
        actor joint positions and velocities, the reward...). policy_params: The
        parameters of policies/controllers. random_key: JAX random key.

        Returns:
            next_state: The updated environment state.
            policy_params: The parameters of policies/controllers (unchanged).
            random_key: The updated random key.
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
        # Use 1.0 reward at each step after done
        # early_stop_reward = jnp.where(
        #     jnp.logical_and(
        #         next_state.done,
        #         env_state.done,
        #     ),
        #     jnp.ones_like(next_state.reward),
        #     jnp.zeros_like(next_state.reward),
        # )
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

        return next_state, policy_params, random_key, transition

    return final_play_step_fn


class FinalDistKheperaxTask(TargetKheperaxTask):
    """Kheperax task that only rewards the final distance to the target"""
    @classmethod
    def create_default_task(cls,
                            kheperax_config: KheperaxConfig,
                            random_key,
                            ):
        env = cls(kheperax_config)
        print(type(env))
        env = brax.v1.envs.wrappers.EpisodeWrapper(env, kheperax_config.episode_length, action_repeat=1)
        env = TypeFixerWrapper(env)

        # Init policy network
        policy_layer_sizes = kheperax_config.mlp_policy_hidden_layer_sizes + (env.action_size,)
        policy_network = MLP(
            layer_sizes=policy_layer_sizes,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            final_activation=jnp.tanh,
        )

        bd_extraction_fn = get_final_xy_position

        play_step_fn = make_final_policy_network_play_step_fn_brax(
            env,
            policy_network,
        )

        scoring_fn, random_key = create_brax_scoring_fn(
            env,
            policy_network,
            bd_extraction_fn,
            random_key,
            play_step_fn=play_step_fn,
            episode_length=kheperax_config.episode_length,
        )

        return env, policy_network, scoring_fn

    def step(self, state: KheperaxState, action: jnp.ndarray) -> KheperaxState:
        random_key = state.random_key

        # actions should be between -1 and 1
        action = jnp.clip(action, -1., 1.)

        random_key, subkey = jax.random.split(random_key)
        wheel_velocities = self._get_wheel_velocities(action, subkey)

        new_robot, bumper_measures = state.robot.move(wheel_velocities[0], wheel_velocities[1], state.maze)

        random_key, subkey = jax.random.split(random_key)
        obs = self._get_obs(new_robot, state.maze, bumper_measures=bumper_measures, random_key=subkey)

        # Standard: reward penalizes high action values
        # reward = -1. * jnp.power(jnp.linalg.norm(wheel_velocities), 2.)

        # Reward is the distance to the target * 100
        target_dist = jnp.linalg.norm(jnp.array(self.kheperax_config.target_pos) - jnp.array(self.get_xy_pos(new_robot)))
        reward = -100 * target_dist
        # reward = -1.

        # Standard: only stop at the end of the episode
        # done = False

        # done if the robot is in the target of if already done
        done = target_dist < self.kheperax_config.target_radius

        # Reward 0 if target is reached
        reward = jnp.where(
            done,
            0.,
            reward,
        )

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