# Examples
## Human Play
A human controller is implemented
in the [`examples.humanplay`](humanplay) package
that makes use of the Gym interface.
Give it a quick spin using
```bash
python -m examples.humanplay.play_vgdl vgdl/games/aliens_lvl0.txt
```
It creates a runnable game,
registers a new gym environment derived from it on the fly,
then runs it with some custom code to handle human input.

Any of the games in [`vgdl/games`](/vgdl/games) will work,
so feel free to play around.

### Recording
The human play module is set up
to use the `gym_recording` package
to save all traces to a `traces` directory.
These can be used for replay
and analysis.

### Playing your own games
The human controller can also play non-standard games
(games not included with this package).
You'll need to provide a Python module
which contains the relevant game classes
(previously installed or somehow discoverable by Python).
For example,
the small Gap World game included in
[`examples/pybrain`](pybrain)
is not included by default,
but can be run using
```bash
python -m examples.humanplay.play_vgdl examples/pybrain/gapworld_lvl0.txt \
  -d examples/pybrain/gapworld.txt -m examples.pybrain.gapworld
```

Be sure to try
```bash
python -m examples.humanplay.play_vgdl --help
```
to figure out what those command-line parameters do.


## OpenAI Gym interface
It's on my TODO list to make a dedicated Gym example,
but really the interface is so simple
that it shouldn't take you long to get started.
The [official docs](https://gym.openai.com/docs/)
have a simple enough introduction.
A few environments are registered by default,
and can be instantiated the usual way,
e.g. `gym.make('vgdl_aliens-v0')`.
List environments by running
[`humanplay/list_envs.py`]().
To use your own games,
you will have to register them with Gym
before getting started.
Check out
[`register_vgdl_env()`](humanplay/play_vgdl.py)
to see for yourself exactly how straightforward this is.


## Planning / PyBrain interface
PyBrain is a bit aged,
but it does some things well still.
I wouldn't use it for function approximation,
but to just exhaustively characterise an MDP
it is absolutely adequate.
You might want to do this to verify your intended
value function, for example.
[`pybrain/gapworld.py`](pybrain/gapworld.py)
implements just such an example.
A game is loaded,
converted to an MDP
(ie a full model),
then the optimal policy is computed by policy iteration.
```bash
python examples/pybrain/gapworld.py
```
To just play Gap World yourself,
see the example under "Playing your own games".

