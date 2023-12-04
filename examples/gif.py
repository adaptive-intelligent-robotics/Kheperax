from __future__ import annotations

# Remove FutureWarning 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import jax.random
import jax.numpy as jnp
from matplotlib import pyplot as plt
from PIL import Image

from kheperax.task import KheperaxConfig, KheperaxTask


def example_usage_gif():

    random_key = jax.random.PRNGKey(1)

    random_key, subkey = jax.random.split(random_key)

    task_config = KheperaxConfig.get_default()
    task_config.resolution = (1024, 1024)

    env, policy_network, scoring_fn = KheperaxTask.create_default_task(
        kheperax_config=task_config,
        random_key=subkey,
    )

    random_key, subkey = jax.random.split(subkey)
    state = env.reset(subkey)

    episode_length = task_config.episode_length
    # episode_length = 10 # debug
    rollout = []
    from tqdm import tqdm 
    for _ in tqdm(range(episode_length)):
        # Render
        image = env.render(state)
        image = jnp.array(image*255).astype('uint8')
        # Flip vertically
        image = image[::-1, :, :]
        rollout.append(image)
        # Random action
        action = jax.random.uniform(random_key, shape=(env.action_size,), minval=-1.0, maxval=1.0)
        state = env.step(state, action)


    # Make GIF
    import imageio
    fps = 30
    duration = 1000/fps
    imageio.mimsave(
        "results/kheperax.gif", 
        rollout, 
        duration=duration,
        loop=0,
        )


if __name__ == '__main__':
    # matplotlib backend agg for headless mode
    plt.switch_backend("agg")
    example_usage_gif()
