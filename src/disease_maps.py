"""
disease_maps.py

Generated disease maps: pairs of clustered and unclustered maps with exactly the same number of
diseased plants, generated with the epidemic spreading model and determined by a seed.
See src/papers/y2027_mrmr/DESIGN-ClusteredDiseaseMap.md
"""

import numpy as np
from scipy import ndimage

from environment import EpidemicSpreadEnvironment
from water_berry_farm import default_infection_seeds, default_spread_dimension

VERSIONS = ["clustered", "unclustered"]

# clustered: a few initial infections, and spread to the neighborhood only
CLUSTERED = {"p-transmission": 0.25, "infection-duration": 5,
             "infection-seeds": "default", "spread-dimension": "default", "jumps-per-day": 0}
# unclustered: weak spread to the immediate neighbors, and "long jumps" to random plants every day.
# The fractional values are relative to the target number of diseased plants.
UNCLUSTERED = {"p-transmission": 0.05, "infection-duration": 5,
               "infection-seeds": 0.02, "spread-dimension": 3, "jumps-per-day": 0.04}
PARAMETERS = {"clustered": CLUSTERED, "unclustered": UNCLUSTERED}


def sub_seed(seed, *keys):
    """An independent seed for one map, derived from the provided seed"""
    return int(np.random.SeedSequence([seed, *keys]).generate_state(1)[0])


def resolve(value, default, target):
    """A parameter value: "default" gives the default, a fraction below 1 is relative to the target"""
    if value == "default":
        return default
    if isinstance(value, float) and value < 1:
        return max(1, int(value * target))
    return int(value)


def generate_disease_map(width, height, version, seed, fraction, params=None, immunity_mask=None, max_days=1000):
    """Generate a width x height disease map with exactly int(fraction * plantable) diseased plants, where
    plantable is the number of cells not immune in the immunity mask (-2 marks an immune cell).
    Returns the epidemic model, whose value field is the map (1.0 healthy, 0.5 infected, 0.0 destroyed),
    and the number of days it ran. The map depends only on the seed, the size and the version."""
    if params is None:
        params = PARAMETERS[version]
    plantable = width * height if immunity_mask is None else int(np.sum(immunity_mask != -2))
    target = int(fraction * plantable)
    duration = params["infection-duration"]
    env = EpidemicSpreadEnvironment(
        "disease", width, height, sub_seed(seed, width, height, VERSIONS.index(version)),
        p_transmission=params["p-transmission"], infection_duration=duration,
        spread_dimension=resolve(params["spread-dimension"], default_spread_dimension(width), target),
        infection_seeds=resolve(params["infection-seeds"], default_infection_seeds(width), target),
        immunity_mask=immunity_mask)
    env.create_value()
    jumps = resolve(params["jumps-per-day"], 0, target) if params["jumps-per-day"] else 0
    days = 0
    while diseased_count(env.value) < target:
        if days == max_days:
            raise Exception(f"The disease did not reach {target} plants in {max_days} days")
        env.proceed(1)
        days += 1
        if jumps:
            healthy = np.flatnonzero(env.status == 0)
            new = env.random.choice(healthy, size=min(jumps, len(healthy)), replace=False)
            env.status.flat[new] = duration
            env.create_value()
    # the last day usually overshoots: heal the surplus among the newest infections
    excess = diseased_count(env.value) - target
    if excess > 0:
        newest = np.flatnonzero(env.status == env.status.max())
        if len(newest) < excess:
            newest = np.flatnonzero(env.status > 0)
        env.status.flat[env.random.choice(newest, size=excess, replace=False)] = 0
        env.create_value()
    return env, days


def diseased_count(value):
    """The number of diseased plants (infected or destroyed) of a map"""
    return int(np.sum(value < 1.0))


def cluster_stats(value):
    """Statistics of the diseased plants of a map: their number, and the connected spots they form
    (diagonal neighbors included)"""
    diseased = value < 1.0
    labels, count = ndimage.label(diseased, structure=np.ones((3, 3)))
    spots = np.bincount(labels.ravel())[1:]
    return {"diseased": int(diseased.sum()),
            "fraction": round(float(diseased.mean()), 4),
            "spots": count,
            "largest spot": int(spots.max()) if count else 0,
            "mean spot": round(float(spots.mean()), 1) if count else 0,
            "in spots >= 10": round(float(spots[spots >= 10].sum() / diseased.sum()), 2) if count else 0}
