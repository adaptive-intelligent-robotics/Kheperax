from __future__ import annotations

# Remove FutureWarning 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import jax.random
from matplotlib import pyplot as plt

from kheperax.tasks.target import TargetKheperaxConfig, TargetKheperaxTask


def example_usage_render():

    random_key = jax.random.PRNGKey(1)

    random_key, subkey = jax.random.split(random_key)

    task_config = TargetKheperaxConfig.get_default()
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

    file_name = 'results/target_maze.png'
    plt.savefig(file_name)
    print("Saved file:", file_name)


if __name__ == '__main__':
    # matplotlib backend agg for headless mode
    plt.switch_backend("agg")
    example_usage_render()
