from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import flax.struct
import jax.tree_util
from jax import numpy as jnp

from kheperax.simu.geoms import Pos, Segment


class Maze(flax.struct.PyTreeNode):
    inside_walls: Segment
    border_walls: Segment

    @property
    def walls(self) -> Segment:
        return jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y]), self.inside_walls, self.border_walls
        )

    @classmethod
    def make_border_walls(
        cls,
        limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    ) -> Segment:
        if limits is None:
            limits = ((0.0, 0.0), (1.0, 1.0))

        (min_x, min_y), (max_x, max_y) = limits

        border_walls = []

        # bottom border
        border_walls.append(Segment(Pos(min_x, min_y), Pos(max_x, min_y)))
        # left border
        border_walls.append(Segment(Pos(min_x, min_y), Pos(min_x, max_y)))
        # top border
        border_walls.append(Segment(Pos(min_x, max_y), Pos(max_x, max_y)))
        # right border
        border_walls.append(Segment(Pos(max_x, max_y), Pos(max_x, min_y)))

        border_walls_tree = jax.tree_util.tree_map(
            lambda *x: jnp.asarray(x, dtype=jnp.float32), *border_walls
        )

        return border_walls_tree

    @classmethod
    def create(
        cls,
        segments_list: Optional[List[Segment]] = None,
        limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    ) -> Maze:
        """
        Create a maze from a list of segments and border limits.

        Args:
            segments_list: List of segments/walls to create the maze.
                By default, it is an empty list.
            limits: Limits of the maze.
                By default, it is a square from (0, 0) to (1, 1).

        Returns:
            Maze: A maze object.
        """
        if segments_list is None:
            segments_list = []
        else:
            segments_list = copy.deepcopy(segments_list)

        if limits is None:
            limits = ((0.0, 0.0), (1.0, 1.0))

        border_walls = cls.make_border_walls(limits=limits)

        inside_walls = jax.tree_util.tree_map(
            lambda *x: jnp.asarray(x, dtype=jnp.float32), *segments_list
        )

        return Maze(inside_walls=inside_walls, border_walls=border_walls)
