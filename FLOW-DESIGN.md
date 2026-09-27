# Experiment flows

Waterberry Farms flows execute the existing experiment notebooks in a defined
order. They are deliberately not a general workflow engine: there is no DAG,
database, component registry, or automatic implementation selection.

## Notebook entry points

Every resolved exp/run has an `input-to-notebook` list. Each item is a POSIX
path relative to the repository root. The field is metadata; loading the
exp/run never executes a notebook.

For an ordinary algorithm run, list order is significant: the simulation
notebook comes first and its visualization notebook second. Environment runs
have their precalculation notebook as their only entry. Collection runs have
their comparison notebook as their only entry. Supporting configurations have
an empty list.

The notebooks selected by a flow always come from this field. The checked-in
metadata test verifies that all declared paths exist.

Queue construction resolves exp/runs with `create_data_dir=False`. Inspecting
the flow therefore does not create result directories or interfere with the
producer's selected `creation_style`.

## Workspace

`setup_flow()` creates a named workspace below the machine-specific
`flows_path` setting:

```text
<flows_path>/<flow-name>/
  expruns/
  results/
  executed-notebooks/
```

It copies the requested experiment families from the currently active
experiment path, so a caller may start from an external collection of exp/runs.
It then points `Config` at the copied configurations and external results.
Generated data and executed notebook copies never belong in the source tree.

Stage notebooks receive the same five Papermill parameters:

- `experiment`
- `run`
- `creation_style`
- `expruns_path`
- `results_path`

Producer and comparison steps receive the flow's selected `creation_style`.
Visualization always receives `exist-ok`, because it must reuse rather than
replace the simulation directory it reads. `exist-ok` means reuse the
directory; it does not mean skip the computation.

Papermill exceptions are not suppressed. A failed stage stops the flow, and
the executed notebook under `executed-notebooks` is the diagnostic artifact.

## Initial flows

`notebooks/Flow-1Robot1Day.ipynb` uses
`1robot1day/all-supported`. It deduplicates the environment references, runs
all selected algorithms, visualizes every run, and executes the collection
comparison.

`notebooks/Flow-nRobot1Day.ipynb` uses `mrmr/mrmr_all`. It precalculates the
clustered and unclustered environments, runs and visualizes the six selected
MRMR benchmark configurations, and produces overall and per-environment
comparisons.

Changing flow membership is an explicit edit of the collection's `tocompare`
list. Adding a new implementation remains the responsibility of experiment
code: YAML contains its parameters and notebook entry points, not a Python
class, factory, registry key, or dynamic import instruction.
