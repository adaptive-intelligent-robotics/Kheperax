import jax
from jax import numpy as jnp
import dataclasses
import brax.envs

from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks.brax_envs import create_brax_scoring_fn
from qdax.environments.bd_extractors import (
    get_final_xy_position,
)

from kheperax.task import KheperaxConfig, KheperaxTask, KheperaxState

from kheperax.maze import Maze
from kheperax.rendering_tools import RenderingTools
from kheperax.robot import Robot
from kheperax.type_fixer_wrapper import TypeFixerWrapper

from kheperax.maps import KHERPERAX_MAZES

@dataclasses.dataclass
class TargetKheperaxConfig(KheperaxConfig):
    target_pos: tuple 
    target_radius: float 

    @classmethod
    def get_default(cls):
        return cls.get_map("standard")
    
    @classmethod
    def get_map(cls, map_name):
        map = KHERPERAX_MAZES[map_name]
        return cls(
            episode_length=1000,
            mlp_policy_hidden_layer_sizes=(8,),
            resolution=(128, 128),
            action_scale=0.025,
            maze=Maze.create(
                segments_list=map["segments"]
            ),
            robot=Robot.create_default_robot(),
            std_noise_wheel_velocities=0.0,
            target_pos=map["target_pos"],
            target_radius=map["target_radius"],
        )

class TargetKheperaxTask(KheperaxTask):
    @classmethod
    def create_default_task(cls,
                            kheperax_config: KheperaxConfig,
                            random_key,
                            ):

        env = cls(kheperax_config)
        print(type(env))
        env = brax.envs.wrappers.EpisodeWrapper(env, kheperax_config.episode_length, action_repeat=1)
        env = TypeFixerWrapper(env)

        # Init policy network
        policy_layer_sizes = kheperax_config.mlp_policy_hidden_layer_sizes + (env.action_size,)
        policy_network = MLP(
            layer_sizes=policy_layer_sizes,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            final_activation=jnp.tanh,
        )

        bd_extraction_fn = get_final_xy_position

        scoring_fn, random_key = create_brax_scoring_fn(
            env,
            policy_network,
            bd_extraction_fn,
            random_key,
            episode_length=kheperax_config.episode_length,
        )

        return env, policy_network, scoring_fn
    

    def step(self, state: KheperaxState, action: jnp.ndarray) -> KheperaxState:
        random_key = state.random_key

        # actions should be between -1 and 1
        action = jnp.clip(action, -1., 1.)

        random_key, subkey = jax.random.split(random_key)
        wheel_velocities = self._get_wheel_velocities(action, subkey)

        new_robot, bumper_measures = state.robot.move(wheel_velocities[0], wheel_velocities[1], state.maze)

        random_key, subkey = jax.random.split(random_key)
        obs = self._get_obs(new_robot, state.maze, bumper_measures=bumper_measures, random_key=subkey)

        # Standard: reward penalizes high action values
        # reward = -1. * jnp.power(jnp.linalg.norm(wheel_velocities), 2.)

        # Reward is the distance to the target
        target_dist = jnp.linalg.norm(jnp.array(self.kheperax_config.target_pos) - jnp.array(self.get_xy_pos(new_robot)))
        reward = -1. * target_dist
        # reward = -1.

        # Standard: only stop at the end of the episode
        # done = False

        # done if the robot is in the target
        done = target_dist < self.kheperax_config.target_radius

        state.info["state_descriptor"] = self.get_xy_pos(new_robot)

        random_key, subkey = jax.random.split(random_key)
        new_random_key = subkey

        return state.replace(
            maze=state.maze,
            robot=new_robot,
            obs=obs,
            reward=reward,
            done=done,
            random_key=new_random_key,
        )
    
    def render(self, state: KheperaxState, ) -> jnp.ndarray:
        # WARNING: only consider the maze is in the unit square
        coeff_triangle = 3.
        image = jnp.zeros(self.kheperax_config.resolution, dtype=jnp.float32)
        image = RenderingTools.place_triangle(image,
                                              point_1=(
                                                  state.robot.posture.x + coeff_triangle * state.robot.radius * jnp.cos(
                                                      state.robot.posture.angle),
                                                  state.robot.posture.y + coeff_triangle * state.robot.radius * jnp.sin(
                                                      state.robot.posture.angle)),
                                              point_2=(state.robot.posture.x + state.robot.radius * jnp.cos(
                                                  state.robot.posture.angle - jnp.pi / 2),
                                                       state.robot.posture.y + state.robot.radius * jnp.sin(
                                                           state.robot.posture.angle - jnp.pi / 2)),
                                              point_3=(state.robot.posture.x + state.robot.radius * jnp.cos(
                                                  state.robot.posture.angle + jnp.pi / 2),
                                                       state.robot.posture.y + state.robot.radius * jnp.sin(
                                                           state.robot.posture.angle + jnp.pi / 2)),
                                              value=2.)

        image = RenderingTools.place_circle(image,
                                            center=(state.robot.posture.x, state.robot.posture.y),
                                            radius=state.robot.radius,
                                            value=1.)
        
        image = RenderingTools.place_circle(image,
                                            center=(self.kheperax_config.target_pos[0], self.kheperax_config.target_pos[1]),
                                            radius=self.kheperax_config.target_radius,
                                            value=3.)

        white = jnp.array([1., 1., 1.])
        blue = jnp.array([0., 0., 1.])
        red = jnp.array([1., 0., 0.])
        green = jnp.array([0., 1., 0.])
        magenta = jnp.array([0.5, 0., 1.])
        cyan = jnp.array([0., 1., 1.])
        yellow = jnp.array([1., 1., 0.])
        black = jnp.array([0., 0., 0.])

        image = RenderingTools.place_segments(image,
                                              state.maze.walls,
                                              value=5.)

        index_to_color = {
            0.: white,
            1.: blue,
            2.: magenta,
            3.: green,
            5.: black,
        }

        def _map_colors(x):
            new_array = jnp.zeros((3,))
            for key, value in index_to_color.items():
                new_array = jax.lax.cond(jnp.isclose(x, key), lambda _: value, lambda _: new_array, operand=None)
            return new_array

        image = jax.vmap(jax.vmap(_map_colors))(image)

        return image

