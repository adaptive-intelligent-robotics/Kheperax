import jax
from chex import ArrayTree


def get_batch_size(tree: ArrayTree) -> int:
    batch_size = jax.tree_leaves(tree)[0].shape[0]
    return batch_size


def get_index_pytree(tree: ArrayTree, index: int) -> ArrayTree:
    return jax.tree_map(lambda x: x[index], tree)
