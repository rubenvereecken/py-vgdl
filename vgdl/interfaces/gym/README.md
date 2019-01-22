## Some notes on OpenAI Gym control flow

Order of calls on an `Env` object:

```
render
reset

step
render
step
render
..
```

As you can see,
the very first `render` call happens before `reset`.
