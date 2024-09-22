from __future__ import annotations

from typing import Dict, Any

import flax.struct
from jax import numpy as jnp
from qdax.custom_types import RNGKey

from kheperax.simu.maze import Maze
from kheperax.simu.robot import Robot


class KheperaxState(flax.struct.PyTreeNode):
    """
    Environment state for training and inference.
    """
    robot: Robot
    maze: Maze
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    random_key: RNGKey
    metrics: Dict[str, jnp.ndarray] = flax.struct.field(default_factory=dict)
    info: Dict[str, Any] = flax.struct.field(default_factory=dict)
