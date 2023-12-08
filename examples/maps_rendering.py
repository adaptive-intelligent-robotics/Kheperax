from __future__ import annotations

# Remove FutureWarning 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import jax.random
from matplotlib import pyplot as plt

from kheperax.target import TargetKheperaxConfig, TargetKheperaxTask
from kheperax.maps import KHERPERAX_MAZES
from kheperax.quad_task import QuadKheperaxConfig

def example_usage_render(map_name='standard'):

    random_key = jax.random.PRNGKey(1)

    random_key, subkey = jax.random.split(random_key)

    task_config = TargetKheperaxConfig.get_map(map_name)
    task_config.resolution = (1024, 1024)

    env, policy_network, scoring_fn = TargetKheperaxTask.create_default_task(
        kheperax_config=task_config,
        random_key=subkey,
    )

    random_key, subkey = jax.random.split(subkey)
    init_state = env.reset(subkey)

    # print(init_state.obs)
    print(init_state.robot.posture)

    img = env.render(init_state)
    # print(img.shape)
    # print(img)

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

    task_config = QuadKheperaxConfig.get_map(map_name)
    task_config.resolution = (1024, 1024)

    env, policy_network, scoring_fn = TargetKheperaxTask.create_default_task(
        kheperax_config=task_config,
        random_key=subkey,
    )

    random_key, subkey = jax.random.split(subkey)
    init_state = env.reset(subkey)

    # print(init_state.obs)
    print(init_state.robot.posture)

    img = env.render(init_state)
    # print(img.shape)
    # print(img)

    plt.imshow(img, origin='lower')
    plt.xticks([])
    plt.yticks([])

    file_name = f'maps/quad_{map_name}.png'
    plt.savefig(file_name)
    print("Saved file:", file_name)


if __name__ == '__main__':
    # matplotlib backend agg for headless mode
    plt.switch_backend("agg")
    for map_name in KHERPERAX_MAZES:
        print(map_name)
        quad_maps_render(map_name)
        example_usage_render(map_name)