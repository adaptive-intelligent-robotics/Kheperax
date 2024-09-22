from __future__ import annotations

from typing import List, Tuple

import flax.struct
import jax.tree_util
from jax import numpy as jnp

from kheperax.simu.geoms import Segment, Pos


class Maze(flax.struct.PyTreeNode):
    walls: Segment

    @classmethod
    def create(
        cls,
        segments_list: List[Segment] = None,
        limits: Tuple[Tuple[float, float], Tuple[float, float]] = None,
    ):
        """
        Create a maze from a list of segments and border limits.

        Args:
            segments_list: List of segments/walls to create the maze. By default, it is an empty list.
            limits: Limits of the maze. By default, it is a square from (0, 0) to (1, 1).

        Returns:
            Maze: A maze object.
        """
        if segments_list is None:
            segments_list = []
        if limits is None:
            limits = ((0., 0.), (1., 1.))
        (min_x, min_y), (max_x, max_y) = limits

        # bottom border
        segments_list.append(Segment(Pos(min_x, min_y), Pos(max_x, min_y)))
        # left border
        segments_list.append(Segment(Pos(min_x, min_y), Pos(min_x, max_y)))
        # top border
        segments_list.append(Segment(Pos(min_x, max_y), Pos(max_x, max_y)))
        # right border
        segments_list.append(Segment(Pos(max_x, max_y), Pos(max_x, min_y)))

        walls = jax.tree_util.tree_map(
            lambda *x: jnp.asarray(x, dtype=jnp.float32), *segments_list
        )

        return Maze(walls)

    @classmethod
    def create_default_maze(cls):
        """
        Create a default hard-maze, from the original Novelty Search paper
        https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/lehman_ecj11.pdf
        """
        return cls.create(segments_list=[
            Segment(Pos(0.25, 0.25), Pos(0.25, 0.75)),
            Segment(Pos(0.14, 0.45), Pos(0., 0.65)),
            Segment(Pos(0.25, 0.75), Pos(0., 0.8)),
            Segment(Pos(0.25, 0.75), Pos(0.66, 0.875)),
            Segment(Pos(0.355, 0.), Pos(0.525, 0.185)),
            Segment(Pos(0.25, 0.5), Pos(0.75, 0.215)),
            Segment(Pos(1., 0.25), Pos(0.435, 0.55)),
        ])
