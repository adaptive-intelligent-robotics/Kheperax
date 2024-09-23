from __future__ import annotations

# Remove FutureWarning
import warnings
from pathlib import Path

import imageio
import jax.random
from matplotlib import pyplot as plt
from tqdm import tqdm

from kheperax.tasks.quad import make_quad_config
from kheperax.tasks.target import TargetKheperaxConfig, TargetKheperaxTask

warnings.simplefilter(action="ignore", category=FutureWarning)


def example_usage_gif(map_name: str = "standard") -> None:
    print(f"Rendering GIF {map_name}")

    random_key = jax.random.PRNGKey(1)

    random_key, subkey = jax.random.split(random_key)

    # Define Task configuration
    if "quad_" in map_name:
        base_map_name = map_name.replace("quad_", "")
        config_kheperax = make_quad_config(
            TargetKheperaxConfig.get_default_for_map(base_map_name)
        )
    else:
        config_kheperax = TargetKheperaxConfig.get_default_for_map(map_name)

    env, policy_network, scoring_fn = TargetKheperaxTask.create_default_task(
        kheperax_config=config_kheperax,
        random_key=subkey,
    )

    random_key, subkey = jax.random.split(subkey)
    jit_env_step = jax.jit(env.step)
    state = env.reset(subkey)

    rollout = []
    base_image = env.create_image(state)

    episode_length = config_kheperax.episode_length
    # episode_length = 10  # debug
    for _ in tqdm(range(episode_length)):
        # Render
        image = env.add_robot(base_image, state)
        image = env.add_lasers(image, state)
        image = env.render_rgb_image(image, flip=True)
        rollout.append(image)
        # Random action
        action = jax.random.uniform(
            random_key, shape=(env.action_size,), minval=-1.0, maxval=1.0
        )
        state = jit_env_step(state, action)

    # Make GIF
    fps = 30
    duration = 1000 / fps

    folder = Path("output/")
    folder.mkdir(exist_ok=True, parents=True)
    imageio.mimsave(
        folder / "kheperax.gif",
        rollout,
        duration=duration,
        loop=0,
    )


def run_example() -> None:
    map_name = "standard"
    # map_name='pointmaze'
    # map_name='snake'

    # quad=True
    quad = False

    # matplotlib backend agg for headless mode
    map_name = ("quad_" if quad else "") + map_name
    plt.switch_backend("agg")
    example_usage_gif(map_name)


if __name__ == "__main__":
    run_example()
