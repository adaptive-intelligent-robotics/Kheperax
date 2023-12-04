"""
Example directly inspired from:
https://github.com/adaptive-intelligent-robotics/QDax/blob/b44969f94aaa70dc6e53aaed95193f65f20400c2/examples/scripts/me_example.py
"""
# Remove FutureWarning 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import functools
from tqdm import tqdm 

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from kheperax.maze import Maze

from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.map_elites import MAPElites
from qdax.utils.metrics import default_qd_metrics
from qdax.utils.plotting import plot_2d_map_elites_repertoire, plot_map_elites_results

# from kheperax.task import KheperaxTask, KheperaxConfig
from kheperax.target import TargetKheperaxConfig, TargetKheperaxTask

def run_me() -> None:
    seed = 42
    batch_size = 2048
    num_evaluations = int(1e6)
    num_iterations = num_evaluations // batch_size
    grid_shape = (50, 50)
    episode_length = 1000
    mlp_policy_hidden_layer_sizes = (8,)

    iso_sigma = 0.2
    line_sigma = 0.0

    # Init a random key
    random_key = jax.random.PRNGKey(seed)
    random_key, subkey = jax.random.split(random_key)

    # Define Task configuration
    config_kheperax = TargetKheperaxConfig.get_default()
    config_kheperax.episode_length = episode_length
    config_kheperax.mlp_policy_hidden_layer_sizes = mlp_policy_hidden_layer_sizes

    # Example of modification of the robots attributes (same thing could be done with the maze)
    config_kheperax.robot = config_kheperax.robot.replace(lasers_return_minus_one_if_out_of_range=True)

    # Create Kheperax Task.
    (
        env,
        policy_network,
        scoring_fn,
    ) = TargetKheperaxTask.create_default_task(
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
    qd_offset = episode_length * jnp.sqrt(2)
    metrics_fn = functools.partial(
        default_qd_metrics,
        qd_offset=qd_offset,
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

    all_metrics = []
    for iteration in range(num_iterations):
        (repertoire, emitter_state, metrics, random_key,) = map_elites.update(
            repertoire,
            emitter_state,
            random_key,
        )
        all_metrics.append(metrics)
        print(f"{iteration}/{num_iterations} - { {k:v.item() for (k,v) in metrics.items()} }")

    metrics = {
        k: jnp.stack([m[k] for m in all_metrics]) for k in all_metrics[0].keys()
    }

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
    plt.savefig("results/target_repertoire.png")
    # plt.show()

    env_steps = jnp.arange(1, num_iterations + 1) * batch_size * episode_length
    fig, axes = plot_map_elites_results(
        env_steps=env_steps,
        metrics=metrics,
        repertoire=repertoire,
        min_bd=min_bd,
        max_bd=max_bd,
    )
    # Make big title
    plt.suptitle("Map-Elites in Target-based Kheperax", y=0.9, fontsize=20)

    plt.savefig("results/ME_stats.png")

    # Record gif
    elite_index = jnp.argmax(repertoire.fitnesses)
    elite = jax.tree_util.tree_map(
        lambda x: x[elite_index], 
        repertoire.genotypes
        )

    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(policy_network.apply)
    state = env.reset(subkey)

    rollout = []

    for _ in tqdm(range(episode_length)):
        # Render
        image = env.render(state)
        image = jnp.array(image*255).astype('uint8')
        # Flip vertically
        image = image[::-1, :, :]
        rollout.append(image)
        
        # Update state
        if state.done:
            break
        action = jit_inference_fn(elite, state.obs)
        state = jit_env_step(state, action)

    # Make GIF
    import imageio
    fps = 30
    duration = 1000/fps
    imageio.mimsave("results/kheperax.gif", rollout, duration=duration)
        

if __name__ == "__main__":
    # matplotlib backend agg for headless mode
    plt.switch_backend("agg")
    run_me()