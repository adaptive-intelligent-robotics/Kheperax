from typing import Tuple

import jax
import jax.numpy as jnp

from kheperax.custom_types import KheperaxImage
from kheperax.simu.geoms import Pos, Segment
from kheperax.tasks.config import KheperaxConfig


def make_meshgrid(
    cfg: KheperaxConfig, image: KheperaxImage
) -> Tuple[jax.Array, jax.Array]:
    (min_x, min_y), (max_x, max_y) = cfg.limits
    x, y = jnp.meshgrid(
        jnp.linspace(min_x, max_x, image.shape[0]),
        jnp.linspace(min_y, max_y, image.shape[1]),
    )
    return x, y


class RenderingTools:
    @classmethod
    def place_circle(
        cls,
        cfg: KheperaxConfig,
        image: KheperaxImage,
        center: jax.typing.ArrayLike,
        radius: float,
        value: float,
    ) -> KheperaxImage:
        x, y = make_meshgrid(cfg, image)

        return jnp.where(
            (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2, value, image
        )

    @classmethod
    def place_triangle(
        cls,
        cfg: KheperaxConfig,
        image: KheperaxImage,
        point_1: jax.typing.ArrayLike,
        point_2: jax.typing.ArrayLike,
        point_3: jax.typing.ArrayLike,
        value: float,
    ) -> KheperaxImage:
        x, y = make_meshgrid(cfg, image)

        return jnp.where(
            jnp.logical_and(
                (x - point_1[0]) * (y - point_2[1])
                - (x - point_2[0]) * (y - point_1[1])
                <= 0,
                jnp.logical_and(
                    (x - point_2[0]) * (y - point_3[1])
                    - (x - point_3[0]) * (y - point_2[1])
                    <= 0,
                    (x - point_3[0]) * (y - point_1[1])
                    - (x - point_1[0]) * (y - point_3[1])
                    <= 0,
                ),
            ),
            value,
            image,
        )

    @classmethod
    def place_rectangle(
        cls,
        cfg: KheperaxConfig,
        image: KheperaxImage,
        start: jax.typing.ArrayLike,
        width: float,
        height: float,
        value: float,
    ) -> KheperaxImage:
        x, y = make_meshgrid(cfg, image)

        return jnp.where(
            (x >= start[0] - width / 2)
            & (x <= start[0] + width / 2)
            & (y >= start[1])
            & (y <= start[1] + height),
            value,
            image,
        )

    @classmethod
    def get_distance_point_to_segment(cls, point: Pos, segment: Segment) -> jax.Array:
        return Pos.calculate_projection_on_segment(point, segment).dist_to(point)

    @classmethod
    def place_segments(
        cls, cfg: KheperaxConfig, image: KheperaxImage, segments: Segment, value: float
    ) -> KheperaxImage:
        x, y = make_meshgrid(cfg, image)

        get_distance_point_to_segment_v = jax.vmap(
            cls.get_distance_point_to_segment, in_axes=(0, None)
        )
        get_distance_point_to_segment_vv = jax.vmap(
            get_distance_point_to_segment_v, in_axes=(0, None)
        )
        get_distance_point_to_segment_vvv = jax.vmap(
            get_distance_point_to_segment_vv, in_axes=(None, 0)
        )

        points = Pos(x=x, y=y)
        distances = get_distance_point_to_segment_vvv(points, segments)
        distances = jnp.min(distances, axis=(0,))

        return jnp.where(distances <= 0.005, value, image)
