from __future__ import annotations

from pathlib import Path

import jax.random
from matplotlib import pyplot as plt

from kheperax.tasks.main import KheperaxConfig, KheperaxTask


def example_usage_render():

    random_key = jax.random.PRNGKey(1)

    random_key, subkey = jax.random.split(random_key)

    task_config = KheperaxConfig.get_default()
    task_config.resolution = (1024, 1024)

    env, policy_network, scoring_fn = KheperaxTask.create_default_task(
        kheperax_config=task_config,
        random_key=subkey,
    )

    random_key, subkey = jax.random.split(subkey)
    init_state = env.reset(subkey)

    img = env.render(init_state)

    plt.imshow(img, origin='lower')
    plt.xticks([])
    plt.yticks([])

    folder_save = Path('output')
    folder_save.mkdir(exist_ok=True, parents=True)
    file_name = folder_save / "maze.png"
    plt.savefig(file_name)
    print("Saved file:", file_name)


if __name__ == '__main__':
    # matplotlib backend agg for headless mode
    plt.switch_backend("agg")
    example_usage_render()
