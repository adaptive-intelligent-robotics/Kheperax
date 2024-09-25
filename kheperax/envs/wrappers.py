from __future__ import annotations

from typing import Any, Tuple

import jax
from jax import numpy as jnp

from kheperax.envs.env import Env
from kheperax.envs.kheperax_state import KheperaxState


class Wrapper(Env):
    """Wraps the environment to allow modular transformations."""

    def __init__(self, env: Env):
        self.env = env

    def reset(self, rng: jax.typing.ArrayLike) -> KheperaxState:
        return self.env.reset(rng)

    def step(self, state: KheperaxState, action: jax.typing.ArrayLike) -> KheperaxState:
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

    def __getattr__(self, name: str) -> Any:
        if name == "__setstate__":
            raise AttributeError(name)
        return getattr(self.env, name)


class EpisodeWrapper(Wrapper):
    """Maintains episode step count and sets done at episode end."""

    def __init__(self, env: Env, episode_length: int, action_repeat: int):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jax.typing.ArrayLike) -> KheperaxState:
        state = self.env.reset(rng)
        state.info["steps"] = jnp.zeros(rng.shape[:-1])
        state.info["truncation"] = jnp.zeros(rng.shape[:-1])
        state.info["truncation"] = jnp.asarray(
            state.info["truncation"], dtype=jnp.int32
        )
        return state

    def step(self, state: KheperaxState, action: jax.typing.ArrayLike) -> KheperaxState:
        def f(state: KheperaxState, _: Any) -> Tuple[KheperaxState, jax.Array]:
            nstate = self.env.step(state, action)
            return nstate, nstate.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jnp.sum(rewards, axis=0))
        steps = state.info["steps"] + self.action_repeat
        one = jnp.ones_like(state.done)
        zero = jnp.zeros_like(state.done)
        episode_length = jnp.array(self.episode_length, dtype=jnp.int32)
        done = jnp.where(steps >= episode_length, one, state.done)
        state.info["truncation"] = jnp.where(
            steps >= episode_length, 1 - state.done, zero
        )
        state.info["truncation"] = jnp.asarray(
            state.info["truncation"], dtype=jnp.int32
        )
        state.info["steps"] = steps
        return state.replace(done=done)  # type: ignore
