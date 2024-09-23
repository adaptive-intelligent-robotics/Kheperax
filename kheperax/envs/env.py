from __future__ import annotations

import abc

import jax

from kheperax.envs.kheperax_state import KheperaxState


class Env(abc.ABC):
    """API for driving an agent."""

    @abc.abstractmethod
    def reset(self, rng: jax.typing.ArrayLike) -> KheperaxState:
        """Resets the environment to an initial state."""

    @abc.abstractmethod
    def step(self, state: KheperaxState, action: jax.typing.ArrayLike) -> KheperaxState:
        """Run one timestep of the environment's dynamics."""

    @property
    def observation_size(self) -> int:
        """The size of the observation vector returned in step and reset."""
        rng = jax.random.PRNGKey(0)
        reset_state = self.unwrapped.reset(rng)
        return int(reset_state.obs.shape[-1])

    @property
    @abc.abstractmethod
    def action_size(self) -> int:
        """The size of the action vector expected by step."""

    @property
    def unwrapped(self) -> Env:
        return self
