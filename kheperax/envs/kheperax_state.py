from __future__ import annotations

from typing import Any, Dict

import flax.struct
import jax
from qdax.custom_types import RNGKey

from kheperax.simu.maze import Maze
from kheperax.simu.robot import Robot


class KheperaxState(flax.struct.PyTreeNode):
    """
    Environment state for training and inference.
    """

    robot: Robot
    maze: Maze
    obs: jax.typing.ArrayLike
    reward: jax.typing.ArrayLike
    done: jax.typing.ArrayLike
    random_key: RNGKey
    metrics: Dict[str, jax.typing.ArrayLike] = flax.struct.field(default_factory=dict)
    info: Dict[str, Any] = flax.struct.field(default_factory=dict)
