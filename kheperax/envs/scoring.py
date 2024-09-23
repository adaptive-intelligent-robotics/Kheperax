import functools
from functools import partial
from typing import Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from qdax.core.neuroevolution.buffers.buffer import QDTransition, Transition
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, Params, RNGKey

from kheperax.envs.env import Env
from kheperax.envs.kheperax_state import KheperaxState


def get_final_state_desc(data: QDTransition, mask: jax.typing.ArrayLike) -> Descriptor:
    """Compute final xy position.

    This function suppose that state descriptor is the xy position, as it
    just select the final one of the state descriptors given.
    """
    # reshape mask for bd extraction
    mask = jnp.expand_dims(mask, axis=-1)

    # Get behavior descriptor
    last_index = jnp.int32(jnp.sum(1.0 - mask, axis=1)) - 1
    descriptors = jax.vmap(lambda x, y: x[y])(data.state_desc, last_index)

    # remove the dim coming from the trajectory
    return descriptors.squeeze(axis=1)


def make_policy_network_play_step_fn(
    step_fn: Callable[[KheperaxState, jax.typing.ArrayLike], KheperaxState],
    policy_network: nn.Module,
) -> Callable[[Params, KheperaxState, RNGKey], Tuple[KheperaxState, QDTransition]]:
    """
    Creates a function that when called, plays a step of the environment.

    Args:
        step_fn: The function to play a step of the environment.
        policy_network: The policy network structure used for creating and evaluating
            policy controllers.

    Returns:
        default_play_step_fn: A function that plays a step of the environment.
    """

    # Define the function to play a step with the policy in the environment
    def default_play_step_fn(
        policy_params: Params,
        env_state: KheperaxState,
        key: RNGKey,
    ) -> Tuple[KheperaxState, QDTransition]:
        """
        Play an environment step and return the updated EnvState and the transition.

        Args:
            env_state: The state of the environment (containing for instance the
                actor joint positions and velocities, the reward...).
            policy_params: The parameters of policies/controllers.
            key: JAX random key.

        Returns:
            next_env_state: The updated environment state.
            policy_params: The parameters of policies/controllers (unchanged).
            key: The updated random key.
            transition: containing some information about the transition: observation,
                reward, next observation, policy action...
        """

        actions = policy_network.apply(policy_params, env_state.obs)

        state_desc = env_state.info["state_descriptor"]
        next_env_state = step_fn(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_env_state.obs,
            rewards=next_env_state.reward,
            dones=next_env_state.done,
            actions=actions,
            truncations=next_env_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_env_state.info["state_descriptor"],
        )

        return next_env_state, transition

    return default_play_step_fn


def get_mask_from_transitions(
    data: Transition,
) -> jax.Array:
    is_done = jnp.clip(jnp.cumsum(data.dones, axis=1), 0, 1)
    mask = jnp.roll(is_done, 1, axis=1)
    mask = mask.at[:, 0].set(0)
    return mask


def generate_unroll(
    init_state: KheperaxState,
    policy_params: Params,
    key: RNGKey,
    episode_length: int,
    play_step_fn: Callable[
        [Params, KheperaxState, RNGKey],
        Tuple[
            KheperaxState,
            Transition,
        ],
    ],
) -> Tuple[KheperaxState, Transition]:
    """Generates an episode according to the agent's policy, returns the final state of
    the episode and the transitions of the episode.

    Args:
        init_state: first state of the rollout.
        policy_params: params of the individual.
        key: random key for stochasiticity handling.
        episode_length: length of the rollout.
        play_step_fn: function describing how a step need to be taken.

    Returns:
        A new state, the experienced transition.
    """
    play_step_params_fn = partial(play_step_fn, policy_params)
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, episode_length)

    def _scan_play_step_fn(
        carry: Tuple[KheperaxState,],
        key: RNGKey,
    ) -> Tuple[Tuple[KheperaxState], Transition]:
        _env_state = carry[0]
        _env_state, _transitions = play_step_params_fn(_env_state, key)
        return (_env_state,), _transitions

    (state,), transitions = jax.lax.scan(
        _scan_play_step_fn,
        (init_state,),
        xs=keys,
        length=episode_length,
    )
    return state, transitions


def scoring_function_kheperax_envs(
    policies_params: Genotype,
    key: RNGKey,
    episode_length: int,
    reset_fn: Callable[[RNGKey], KheperaxState],
    play_step_fn: Callable[
        [Params, KheperaxState, RNGKey],
        Tuple[KheperaxState, QDTransition],
    ],
    descriptor_extractor: Callable[[QDTransition, jax.typing.ArrayLike], Descriptor],
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """Evaluates policies contained in policies_params in parallel.
    The play_reset_fn function allows for a more general scoring_function that can be
    called with different batch-size and not only with a batch-size of the same
    dimension as init_states.

    To define purely stochastic environments, using the reset function from the
    environment, use "play_reset_fn = env.reset".

    To define purely deterministic environments, as in "scoring_function", generate
    a single init_state using "init_state = env.reset(key)", then use
    "play_reset_fn = lambda key: init_state".

    Args:
        policies_params: The parameters of closed-loop controllers/policies to evaluate.
        key: A jax random key
        episode_length: The maximal rollout length.
        reset_fn: The function to reset the environment and obtain initial states.
        play_step_fn: The function to play a step of the environment.
        descriptor_extractor: The function to extract the descriptor.

    Returns:
        fitness: Array of fitnesses of all evaluated policies
        descriptor: Behavioural descriptors of all evaluated policies
        extra_scores: Additional information resulting from the evaluation
        key: The updated random key
    """
    batch_size = jax.tree.leaves(policies_params)[0].shape[0]

    # Reset environments
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, batch_size)
    init_states = jax.vmap(reset_fn)(keys)

    # Step environments
    unroll_fn = partial(
        generate_unroll,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
    )
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, batch_size)
    _, data = jax.vmap(unroll_fn)(init_states, policies_params, keys)

    # Create a mask to extract data properly
    mask = get_mask_from_transitions(data)

    # Evaluate
    fitnesses = jnp.sum(data.rewards * (1.0 - mask), axis=1)
    descriptors = descriptor_extractor(data, mask)

    return fitnesses, descriptors, {"transitions": data}, key


def create_kheperax_scoring_fn(
    env: Env,
    policy_network: nn.Module,
    descriptor_extraction_fn: Callable[
        [
            QDTransition,
            jax.typing.ArrayLike,
        ],
        Descriptor,
    ],
    episode_length: int,
    play_step_fn: Optional[
        Callable[[Params, KheperaxState, RNGKey], Tuple[KheperaxState, QDTransition]]
    ] = None,
    reset_fn: Optional[Callable[[RNGKey], KheperaxState]] = None,
) -> Callable[[Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]]:
    """
    Creates a scoring function to evaluate a policy in a BRAX task.

    Args:
        env: The Kheperax task.
        policy_network: The policy network controller.
        descriptor_extraction_fn: The behaviour descriptor extraction function.
        key: a random key used for stochastic operations.
        play_step_fn: the function used to perform environment rollouts and collect
            evaluation episodes. If None, we use make_policy_network_play_step_fn_brax
            to generate it.
        episode_length: The maximal episode length.
        reset_fn: the function used to reset the environment to an initial state.
            Only used if deterministic is False. If None, we take env.reset as
            default reset function.

    Returns:
        The scoring function: a function that takes a batch of genotypes and compute
            their fitnesses and descriptors
    """
    if play_step_fn is None:
        play_step_fn = make_policy_network_play_step_fn(env.step, policy_network)

    # Stochastic case
    if reset_fn is None:
        reset_fn = env.reset

    scoring_fn = functools.partial(
        scoring_function_kheperax_envs,
        episode_length=episode_length,
        reset_fn=reset_fn,
        play_step_fn=play_step_fn,
        descriptor_extractor=descriptor_extraction_fn,
    )

    return scoring_fn
