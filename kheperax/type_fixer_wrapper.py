import jax.numpy as jnp
from brax.v1.envs import Wrapper


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
