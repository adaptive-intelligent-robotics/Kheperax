from __future__ import annotations

# Remove FutureWarning 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import jax.random
from matplotlib import pyplot as plt

from kheperax.tasks.target import TargetKheperaxConfig, TargetKheperaxTask
from kheperax.tasks.quad_task import QuadKheperaxConfig

def example_usage_gif(map_name='standard'):
    print(f"Rendering GIF {map_name}")

    random_key = jax.random.PRNGKey(1)

    random_key, subkey = jax.random.split(random_key)

    # Define Task configuration
    if "quad_" in map_name:
        base_map_name = map_name.replace("quad_", "")
        config_kheperax = QuadKheperaxConfig.get_map(base_map_name)
    else:
        config_kheperax = TargetKheperaxConfig.get_map(map_name)

    env, policy_network, scoring_fn = TargetKheperaxTask.create_default_task(
        kheperax_config=config_kheperax,
        random_key=subkey,
    )

    random_key, subkey = jax.random.split(subkey)
    state = env.reset(subkey)

    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(policy_network.apply)
    state = env.reset(subkey)


    rollout = []
    base_image = env.create_image(state)

    episode_length = config_kheperax.episode_length
    episode_length = 10 # debug
    from tqdm import tqdm 
    for _ in tqdm(range(episode_length)):
        # Render
        image = env.add_robot(base_image, state)
        image = env.add_lasers(image, state)
        image = env.render_rgb_image(image, flip=True)
        rollout.append(image)
        # Random action
        action = jax.random.uniform(random_key, shape=(env.action_size,), minval=-1.0, maxval=1.0)
        state = jit_env_step(state, action)


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

map_name='standard'
# map_name='pointmaze'
# map_name='snake'

# quad=True
quad=False

if __name__ == '__main__':
    # matplotlib backend agg for headless mode
    map_name = ("quad_" if quad else "") + map_name
    plt.switch_backend("agg")
    example_usage_gif(map_name)
