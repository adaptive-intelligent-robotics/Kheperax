from __future__ import annotations

import flax.struct
import jax
from jax import numpy as jnp
from qdax.custom_types import RNGKey

from kheperax.simu.geoms import Pos, Segment


class Laser(flax.struct.PyTreeNode):
    pos: Pos
    angle: float
    max_range: float

    @jax.jit
    def get_segment(self) -> Segment:
        return Segment(
            p1=self.pos,
            p2=Pos(
                x=self.pos.x + self.max_range * jnp.cos(self.angle),
                y=self.pos.y + self.max_range * jnp.sin(self.angle),
            ),
        )

    @jax.jit
    def get_intersection_with_segment(self, segment: Segment) -> Pos:
        return self.get_segment().get_intersection_with(segment)

    def get_measure(
        self,
        tree_segments: Segment,
        return_minus_one_if_out_of_range: bool,
        std_noise: float,
        random_key: RNGKey,
    ) -> jax.Array:

        all_measures = jax.vmap(self.get_measure_for_segment)(tree_segments)
        measure = jnp.min(all_measures)

        random_key, subkey = jax.random.split(random_key)
        noise = jax.random.normal(subkey) * std_noise
        measure = measure + noise
        measure = jnp.maximum(measure, 0.0)

        if return_minus_one_if_out_of_range:
            out_of_range_value = -1.0
        else:
            out_of_range_value = self.max_range

        measure = jnp.where(jnp.isinf(measure), out_of_range_value, measure)
        measure = jnp.where(measure > self.max_range, out_of_range_value, measure)

        return measure

    def get_measure_for_segment(self, segment: Segment) -> jax.Array:
        return self.pos.dist_to(self.get_intersection_with_segment(segment))
