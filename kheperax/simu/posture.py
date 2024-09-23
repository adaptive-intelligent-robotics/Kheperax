from __future__ import annotations

from typing import Any

import flax.struct
import jax
from jax import numpy as jnp


class Posture(flax.struct.PyTreeNode):
    x: jax.typing.ArrayLike
    y: jax.typing.ArrayLike
    angle: jax.typing.ArrayLike

    def dist_to(self, other: Posture) -> jax.Array:
        return self.dist_to_xy(other.x, other.y)

    def dist_to_xy(self, x: jax.typing.ArrayLike, y: jax.typing.ArrayLike) -> jax.Array:
        return jnp.linalg.norm(jnp.asarray([self.x, self.y]) - jnp.asarray([x, y]))

    def rotate(self, angle: jax.typing.ArrayLike) -> Posture:
        x_ = self.x * jnp.cos(angle) - self.y * jnp.sin(angle)
        y_ = self.x * jnp.sin(angle) + self.y * jnp.cos(angle)
        theta_ = self.normalize_angle(self.angle + angle)
        return Posture(x_, y_, theta_)

    def add_pos(self, p: Posture) -> Posture:
        return self.add_pos_xytheta(p.x, p.y, p.angle)

    def add_pos_xytheta(
        self,
        x: jax.typing.ArrayLike,
        y: jax.typing.ArrayLike,
        theta: jax.typing.ArrayLike,
    ) -> Posture:
        return Posture(
            self.x + x,
            self.y + y,
            self.normalize_angle(self.angle + theta),
        )

    @staticmethod
    def normalize_angle(angle: jax.typing.ArrayLike) -> jax.Array:
        return jnp.mod(angle + jnp.pi, 2 * jnp.pi) - jnp.pi

    def move(
        self,
        d_l: jax.typing.ArrayLike,
        d_r: jax.typing.ArrayLike,
        wheels_dist: jax.typing.ArrayLike,
    ) -> Posture:
        old_pos = self
        alpha = (d_r - d_l) / wheels_dist

        def _if_alpha_high(_: Any) -> Posture:
            r = (d_l / alpha) + (wheels_dist / 2)
            d_x = (jnp.cos(alpha) - 1) * r
            d_y = jnp.sin(alpha) * r
            delta_p = Posture(d_x, d_y, alpha)
            delta_p = delta_p.rotate(old_pos.angle - jnp.pi / 2)
            delta_p = delta_p.replace(angle=self.normalize_angle(alpha))
            return delta_p

        def _if_alpha_low(_: Any) -> Posture:
            delta_p = Posture(
                x=d_l * jnp.cos(old_pos.angle),
                y=d_l * jnp.sin(old_pos.angle),
                angle=0.0,
            )
            return delta_p

        delta_p = jax.lax.cond(
            jnp.abs(alpha) > 1e-10, _if_alpha_high, _if_alpha_low, None
        )

        return self.add_pos(delta_p)
