## Notes on project structure

Currently games are kept in `vgdl/games` folder
so they are reachable through `pkg_resources`.
That is, games are bundled with the package,
unlike the `examples` folder,
which is excluded.
Even though games are in a sense examples.
This is so `vgdl.interfaces.gym`
can reach them and provide some default environments.
