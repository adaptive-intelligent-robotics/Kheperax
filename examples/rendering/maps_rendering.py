from __future__ import annotations

# Remove FutureWarning
import warnings
from pathlib import Path
from typing import Union

import jax.random
from matplotlib import pyplot as plt

from kheperax.envs.maze_maps import KHEPERAX_MAZES
from kheperax.tasks.config import KheperaxConfig
from kheperax.tasks.main import KheperaxTask
from kheperax.tasks.quad import make_quad_config
from kheperax.tasks.target import TargetKheperaxConfig, TargetKheperaxTask

warnings.simplefilter(action="ignore", category=FutureWarning)


def quad_maps_render(
    map_name: str = "standard", consider_target: bool = True, make_quad: bool = False
) -> None:
    print(
        f"Rendering {map_name}, "
        f"consider_target={consider_target}, "
        f"make_quad={make_quad}"
    )
    random_key = jax.random.PRNGKey(1)

    random_key, subkey = jax.random.split(random_key)

    task_config: Union[KheperaxConfig, TargetKheperaxConfig]

    if consider_target:
        create_fn = TargetKheperaxTask.create_default_task
        task_config = TargetKheperaxConfig.get_default_for_map(map_name)
    else:
        create_fn = KheperaxTask.create_default_task
        task_config = KheperaxConfig.get_default_for_map(map_name)

    if make_quad:
        task_config = make_quad_config(task_config)
    task_config.resolution = (1024, 1024)

    env, policy_network, scoring_fn = create_fn(
        kheperax_config=task_config,  # type: ignore
        random_key=subkey,
    )

    random_key, subkey = jax.random.split(subkey)
    init_state = env.reset(subkey)

    img = env.render(init_state)

    plt.imshow(img, origin="lower")
    plt.xticks([])
    plt.yticks([])

    folder = Path("output/")
    if consider_target:
        folder = folder / "target"
    else:
        folder = folder / "no_target"

    if make_quad:
        folder = folder / "quad"
    else:
        folder = folder / "no_quad"

    folder.mkdir(exist_ok=True, parents=True)

    file_name = folder / f"{map_name}.png"
    plt.savefig(file_name)
    print("Saved file:", file_name)


def run_example() -> None:
    # matplotlib backend agg for headless mode
    plt.switch_backend("agg")
    for map_name in KHEPERAX_MAZES:
        for consider_target in [True, False]:
            for make_quad in [True, False]:
                quad_maps_render(map_name, consider_target, make_quad)


if __name__ == "__main__":
    run_example()
