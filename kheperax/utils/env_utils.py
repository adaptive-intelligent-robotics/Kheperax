from __future__ import annotations

import abc

import jax
import jax.numpy as jnp

from kheperax.envs.kheperax_state import KheperaxState


class Env(abc.ABC):
    """API for driving an agent."""

    @abc.abstractmethod
    def reset(self, rng: jnp.ndarray) -> KheperaxState:
        """Resets the environment to an initial state."""

    @abc.abstractmethod
    def step(self, state: KheperaxState, action: jnp.ndarray) -> KheperaxState:
        """Run one timestep of the environment's dynamics."""

    @property
    def observation_size(self) -> int:
        """The size of the observation vector returned in step and reset."""
        rng = jax.random.PRNGKey(0)
        reset_state = self.unwrapped.reset(rng)
        return reset_state.obs.shape[-1]

    @property
    @abc.abstractmethod
    def action_size(self) -> int:
        """The size of the action vector expected by step."""

    @property
    def unwrapped(self) -> Env:
        return self


class Wrapper(Env):
    """Wraps the environment to allow modular transformations."""

    def __init__(self, env: Env):
        self.env = env

    def reset(self, rng: jnp.ndarray) -> KheperaxState:
        return self.env.reset(rng)

    def step(self, state: KheperaxState, action: jnp.ndarray) -> KheperaxState:
        return self.env.step(state, action)

    @property
    def observation_size(self) -> int:
        return self.env.observation_size

    @property
    def action_size(self) -> int:
        return self.env.action_size

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name):
        if name == '__setstate__':
            raise AttributeError(name)
        return getattr(self.env, name)


class EpisodeWrapper(Wrapper):
    """Maintains episode step count and sets done at episode end."""

    def __init__(self, env: Env, episode_length: int,
                 action_repeat: int):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jnp.ndarray) -> KheperaxState:
        state = self.env.reset(rng)
        state.info['steps'] = jnp.zeros(rng.shape[:-1])
        state.info['truncation'] = jnp.zeros(rng.shape[:-1])
        return state

    def step(self, state: KheperaxState, action: jnp.ndarray) -> KheperaxState:
        def f(state, _):
            nstate = self.env.step(state, action)
            return nstate, nstate.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jnp.sum(rewards, axis=0))
        steps = state.info['steps'] + self.action_repeat
        one = jnp.ones_like(state.done)
        zero = jnp.zeros_like(state.done)
        episode_length = jnp.array(self.episode_length, dtype=jnp.int32)
        done = jnp.where(steps >= episode_length, one, state.done)
        state.info['truncation'] = jnp.where(steps >= episode_length,
                                             1 - state.done, zero)
        state.info['steps'] = steps
        return state.replace(done=done)


class TypeFixerWrapper(Wrapper):
    """
    Wrapper that fixes the type of the truncation field in the info dictionary
    """

    def reset(self, rng: jnp.ndarray):
        reset_state = self.env.reset(rng)
        reset_state.info["truncation"] = jnp.asarray(reset_state.info["truncation"], dtype=jnp.int32)

        return reset_state

    def step(self, state, action: jnp.ndarray):
        state = self.env.step(state, action)
        state.info["truncation"] = jnp.asarray(state.info["truncation"], dtype=jnp.int32)
        return state
