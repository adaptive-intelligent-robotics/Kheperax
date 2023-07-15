"""
Example directly inspired from:
https://github.com/adaptive-intelligent-robotics/QDax/blob/b44969f94aaa70dc6e53aaed95193f65f20400c2/examples/scripts/me_example.py
"""


import functools

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from kheperax.maze import Maze

from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.utils.metrics import default_qd_metrics
from qdax.utils.plotting import plot_2d_map_elites_repertoire

from kheperax.task import KheperaxTask, KheperaxConfig


def run_me() -> None:
    seed = 42
    batch_size = 2048
    num_evaluations = int(1e6)
    num_iterations = num_evaluations // batch_size
    grid_shape = (50, 50)
    episode_length = 250
    mlp_policy_hidden_layer_sizes = (8,)

    iso_sigma = 0.2
    line_sigma = 0.0

    # Init a random key
    random_key = jax.random.PRNGKey(seed)
    random_key, subkey = jax.random.split(random_key)

    # Define Task configuration
    config_kheperax = KheperaxConfig.get_default()

    # Example of modification of the robots attributes (same thing could be done with the maze)
    config_kheperax.robot = config_kheperax.robot.replace(lasers_return_minus_one_if_out_of_range=True)

    # Create Kheperax Task.
    (
        env,
        policy_network,
        scoring_fn,
    ) = KheperaxTask.create_default_task(
        config_kheperax,
        random_key=subkey,
    )

    # initialising first variables for Map-Elites init
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=batch_size)
    fake_batch = jnp.zeros(shape=(batch_size, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)

    # Define emitter
    variation_fn = functools.partial(
        isoline_variation,
        iso_sigma=iso_sigma,
        line_sigma=line_sigma,
    )
    mixing_emitter = MixingEmitter(
        mutation_fn=lambda x, y: (x, y),
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=batch_size,
    )

    # Define a metrics function
    metrics_fn = functools.partial(
        default_qd_metrics,
        qd_offset=0.5,
    )

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_fn,
    )

    # Compute the centroids
    min_bd, max_bd = env.behavior_descriptor_limits
    centroids = compute_euclidean_centroids(
        grid_shape=grid_shape,
        minval=min_bd,
        maxval=max_bd,
    )

    # Initializes repertoire and emitter state
    repertoire, emitter_state, random_key = map_elites.init(
        init_variables, centroids, random_key
    )

    # Run MAP-Elites loop
    for iteration in range(num_iterations):
        (repertoire, emitter_state, metrics, random_key,) = map_elites.update(
            repertoire,
            emitter_state,
            random_key,
        )
        print(f"{iteration}/{num_iterations} - { {k:v.item() for (k,v) in metrics.items()} }")

    # plot archive
    fig, axes = plot_2d_map_elites_repertoire(
        centroids=repertoire.centroids,
        repertoire_fitnesses=repertoire.fitnesses,
        minval=min_bd,
        maxval=max_bd,
        repertoire_descriptors=repertoire.descriptors,
        # vmin=-0.2,
        # vmax=0.0,
    )

    plt.show()


if __name__ == "__main__":
    run_me()