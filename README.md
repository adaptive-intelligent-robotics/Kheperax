# Kheperax

The *Kheperax* task is a re-implementation of the `fastsim` simulator from _Mouret and Doncieux (2012)_.
Kheperax is fully written using [JAX](https://github.com/google/jax), to leverage hardware accelerators and massive parallelization.

## Task Properties

Each episode is run for a fixed amount of timesteps (by default equal to `250`).
The agent corresponds to a Khepera-like robot that moves in a planar 2-dimensional maze.
This robot has (by default):
- 3 lasers to estimate its distance to some walls in specific directions (by default -45, 0 and 45 degrees).
- 2 bumpers to detect contact with walls.
At each time-step, the agent receives an observation, which corresponds to all laser and bumper measures: 
```
# by default:
[laser 1, laser 2, laser 3, bumper left, bumper right]
```
The bumpers return `1` if there's a contact with a wall and `-1` otherwise.

The actions to pass to the environment should be between `-1` and `1`. 
They are then scaled depending on a scale defined in the environment configuration.

**Quality-Diversity Properties**:
- Fitness: sum of negated norm of actions (-1 * sum of norm a_t, ~negated energy)
- Descriptor: final 2-dimensional location of the robot.


## Run examples

### Creating Virtual Environment

```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Launch MAP-Elites Example

```shell
python -m examples.me_training
```

### Test Rendering a Maze Image

```shell
python -m examples.rendering
```

## Documentation

## Task Configuration

The `KheperaxTask` takes as input a `KheperaxConfig` object, organised as follows:

`KheperaxConfig`
- `Robot`:
  - `posture`: initial `Posture` (containing position and orientation)
  - `radius`: robot radius
  - `laser_ranges`: laser max ranges.
  - `laser_angles`: angles of placement of the lasers on the robot
  - `std_noise_sensor_measures`: standard deviation of the gaussian noise applied to the sensor measures.
  - `lasers_return_minus_one_if_out_of_range`: 
  If `True`, then the lasers return `-1` if their measure is out of range (like in the original implementation).
  If `False`, then returns the max laser range.
- `Maze`:
  - `walls`: tree of `Segments` representing the placement of the walls in the environment.
- `action_scale`: all the commanded wheel velocities will be between -1 * `action_scale` and `action_scale`
- `std_noise_wheel_velocities`: standard deviation of the gaussian noise applied to the wheel velocities.
- `resolution`: resolution of the maze when calling `env.render(...)` 

To get an initial configuration, you can run:
```python
config_kheperax = KheperaxConfig.get_default()
```

You can then modify the properties of the config by directly changing its attributes.


