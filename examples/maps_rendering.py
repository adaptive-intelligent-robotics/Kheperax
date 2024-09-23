from __future__ import annotations

# Remove FutureWarning
import warnings

import jax.random
from matplotlib import pyplot as plt

from kheperax.envs.maze_maps import TARGET_KHEPERAX_MAZES
from kheperax.tasks.quad import make_quad_config
from kheperax.tasks.target import TargetKheperaxConfig, TargetKheperaxTask

warnings.simplefilter(action='ignore', category=FutureWarning)


def example_usage_render(map_name='standard'):
    random_key = jax.random.PRNGKey(1)

    random_key, subkey = jax.random.split(random_key)

    task_config = TargetKheperaxConfig.get_default_for_map(map_name)
    task_config.resolution = (1024, 1024)

    env, policy_network, scoring_fn = TargetKheperaxTask.create_default_task(
        kheperax_config=task_config,
        random_key=subkey,
    )

    random_key, subkey = jax.random.split(subkey)
    init_state = env.reset(subkey)

    img = env.render(init_state)

    plt.imshow(img, origin='lower')
    plt.xticks([])
    plt.yticks([])

    file_name = f'maps/{map_name}.png'
    plt.savefig(file_name)
    print("Saved file:", file_name)


def quad_maps_render(map_name='standard'):
    print(f"Rendering Quad {map_name}")
    random_key = jax.random.PRNGKey(1)

    random_key, subkey = jax.random.split(random_key)

    task_config = make_quad_config(TargetKheperaxConfig.get_default_for_map(map_name))
    task_config.resolution = (1024, 1024)

    env, policy_network, scoring_fn = TargetKheperaxTask.create_default_task(
        kheperax_config=task_config,
        random_key=subkey,
    )

    random_key, subkey = jax.random.split(subkey)
    init_state = env.reset(subkey)

    img = env.render(init_state)

    plt.imshow(img, origin='lower')
    plt.xticks([])
    plt.yticks([])

    file_name = f'maps/quad_{map_name}.png'
    plt.savefig(file_name)
    print("Saved file:", file_name)


def run_example():
    # matplotlib backend agg for headless mode
    plt.switch_backend("agg")
    for map_name in TARGET_KHEPERAX_MAZES:
        print(map_name)
        quad_maps_render(map_name)
        example_usage_render(map_name)


if __name__ == '__main__':
    run_example()
