from __future__ import annotations

from typing import Any, Tuple, Union

import flax.struct
import jax
import jax.lax
import jax.tree_util
from jax import numpy as jnp
from qdax.custom_types import RNGKey

from kheperax.simu.geoms import Disk, Pos, Segment
from kheperax.simu.laser import Laser
from kheperax.simu.maze import Maze
from kheperax.simu.posture import Posture
from kheperax.utils.tree_utils import get_batch_size


class Robot(flax.struct.PyTreeNode):
    posture: Posture
    radius: float
    laser_ranges: Union[float, jax.Array]
    laser_angles: jax.typing.ArrayLike
    std_noise_sensor_measures: float = flax.struct.field(pytree_node=False, default=0.0)

    # Makes the controllers more difficult to learn
    lasers_return_minus_one_if_out_of_range: bool = flax.struct.field(
        pytree_node=False, default=False
    )

    @classmethod
    def create_default_robot(cls) -> Robot:
        return cls(
            posture=Posture(0.15, 0.15, jnp.pi / 2),
            radius=0.015,
            laser_ranges=0.2,
            laser_angles=jnp.array([-jnp.pi / 4, 0, jnp.pi / 4]),
            std_noise_sensor_measures=0.0,
            lasers_return_minus_one_if_out_of_range=False,
        )

    def get_lasers(self) -> Laser:
        list_lasers = []

        for laser_angle in self.laser_angles:
            _laser = Laser(
                Pos(self.posture.x, self.posture.y),
                self.posture.angle + laser_angle,
                self.laser_ranges,
            )
            list_lasers.append(_laser)

        tree_lasers: Laser = jax.tree_util.tree_map(
            lambda *x: jnp.asarray(x, dtype=jnp.float32), *list_lasers
        )
        return tree_lasers

    def get_disk(self) -> Disk:
        return Disk(Pos(self.posture.x, self.posture.y), self.radius)

    def collides(self, maze: Maze) -> jax.Array:
        return self.get_disk().collides(maze.walls)

    def move(
        self, v1: jax.typing.ArrayLike, v2: jax.typing.ArrayLike, maze: Maze
    ) -> Tuple[Robot, jax.Array]:
        previous_robot = self
        new_posture = self.posture.move(v1, v2, 2 * self.radius)

        new_robot: Robot = self.replace(posture=new_posture)

        def if_collides(_: Any) -> Robot:
            return previous_robot

        def if_not_collides(_: Any) -> Robot:
            return new_robot

        next_robot = jax.lax.cond(
            self.collides(maze), if_collides, if_not_collides, None
        )

        return next_robot, self.bumper_measures(maze)

    def laser_measures(self, maze: Maze, random_key: RNGKey) -> jax.Array:

        lasers = self.get_lasers()
        number_lasers = get_batch_size(lasers)

        random_key, *subkeys = jax.random.split(random_key, num=number_lasers + 1)
        array_subkeys = jnp.asarray(subkeys)

        get_measure_v = jax.vmap(
            lambda laser, _subkey: laser.get_measure(
                maze.walls,
                self.lasers_return_minus_one_if_out_of_range,
                std_noise=self.std_noise_sensor_measures,
                random_key=_subkey,
            )
        )
        return get_measure_v(self.get_lasers(), array_subkeys)

    def bumper_measure_with_segment(self, segment: Segment) -> jax.Array:
        pos = Pos.from_posture(self.posture)
        projection = pos.calculate_projection_on_segment(segment)
        vector = projection - pos
        angle = jnp.arctan2(vector.y, vector.x)
        angle_diff = jnp.mod(angle - self.posture.angle + jnp.pi, 2 * jnp.pi) - jnp.pi

        distance = projection.dist_to(pos)

        left_bumper_collides = jnp.logical_and(
            jnp.logical_and(angle_diff <= 0.0, angle_diff > -jnp.pi / 2),
            distance < self.radius,
        )
        right_bumper_collides = jnp.logical_and(
            jnp.logical_and(angle_diff >= 0.0, angle_diff < jnp.pi / 2),
            distance < self.radius,
        )

        left_bumper_measure = jnp.where(left_bumper_collides, 1.0, -1.0)
        right_bumper_measure = jnp.where(right_bumper_collides, 1.0, -1.0)

        return jnp.array([left_bumper_measure, right_bumper_measure])

    def bumper_measures(self, maze: Maze) -> jax.Array:
        get_measure_v = jax.vmap(
            lambda segment: self.bumper_measure_with_segment(segment)
        )
        return jnp.max(get_measure_v(maze.walls), axis=0)
