## Human Play
A human controller is implemented
in the [`examples.humanplay`](humanplay) package
that makes use of the Gym interface.
Give it a quick spin using
```bash
python -m examples.humanplay.play_vgdl vgdl/games/aliens_lvl0.txt
```
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
python -m examples.humanplay.play_vgdl vgdl/games/aliens_lvl0.txt \
  -m examples.pybrain.gapworld -s examples/pybrain/gapworld.txt
```

Be sure to try
```bash
python -m examples.humanplay.play_vgdl --help
```
to figure out what those command-line parameters do.


