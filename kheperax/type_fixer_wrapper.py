import jax.numpy as jnp
from brax.v1.envs import Wrapper
# from brax.v1.envs import env as brax_env


class TypeFixerWrapper(Wrapper):
    def reset(self, rng: jnp.ndarray):
        reset_state = self.env.reset(rng)
        reset_state.info["truncation"] = jnp.asarray(reset_state.info["truncation"], dtype=jnp.int32)

        return reset_state

    def step(self, state, action: jnp.ndarray):
        state = self.env.step(state, action)
        state.info["truncation"] = jnp.asarray(state.info["truncation"], dtype=jnp.int32)
        return state
