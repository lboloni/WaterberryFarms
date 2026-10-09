# Generated clustered and unclustered disease maps

## Purpose

The MRMR experiments compare the approaches on two TYLCV maps. One map is clustered, with the disease in a few compact outbreaks, and the other unclustered, with the disease scattered over the field. Until now, both maps were hand-drawn pictures, `data/expruns/environment/mrmr-clustered-100.png` and `mrmr-notclustered-100.png`. This has three drawbacks:
- **Different amounts of disease:** the clustered map has 491 diseased plants (4.9%), the unclustered map 142 (1.4%). A comparison between the two maps therefore mixes the effect of clustering with the effect of the amount of disease.
- **One size only:** the maps are 100x100. Scenarios with more robots need larger maps.
- **One instance only:** there is a single pair, so results cannot be averaged over several maps.

The maps are now generated instead, from the disease spreading model of the simulator. The maps come in pairs: a clustered and an unclustered version with **exactly the same number of diseased plants**. The pairs are generated in the sizes 30x30, 100x100, and the new 200x200. Every map is determined by a provided random seed.

**Status:** implemented. The generator is `generate_disease_map` in `src/disease_maps.py`. Environments use it through the `tylcv-generated` (and `ccr-generated`) keys of their exp/run (Section 7). The MRMR 2027 experiments run on the generated 100x100 pair. `MRMR-GeneratedMaps.ipynb` visualizes the pairs in all sizes, compares them with the hand-drawn maps, and shows the dependence on the seed. The geometry `Miniberry-200` makes the 200x200 maps usable in simulations. It is tested in `src/test/test_disease_maps.py`.

## 1. The disease spreading model

`EpidemicSpreadEnvironment` (`src/environment.py`) is an SIR-like model on the grid. Every plant has a status:
- `0` is healthy;
- `k >= 1` is infected, with `k` days left;
- `-1` is destroyed (the end of the infection);
- `-2` is immune.

Every day:
- The infections age, and those that end become destroyed.
- Each healthy plant becomes infected with probability `1 - (1 - p_transmission)^c`. The count `c` is the number of infected plants in a `spread_dimension x spread_dimension` neighborhood around it, weighted by `1/distance²`.
- The model starts from `infection_seeds` infections at random plants.

Its value field is the map:
- `1.0` healthy;
- `0.5` infected;
- `0.0` destroyed.

A plant is **diseased** if its value is below 1.0, that is, infected or destroyed. This is also what the VoI score counts as a positive (`v_pos`).

## 2. Clustered maps: local spread only

The clustered version uses the model as the simulator uses it for TYLCV:

| Parameter | Value |
|---|---|
| `p_transmission` | 0.25 |
| `infection_duration` | 5 days |
| `infection_seeds` | `default_infection_seeds(size)` = 3 per 30 cells of width: 3, 9 and 18 for the three sizes |
| `spread_dimension` | `default_spread_dimension(size)` = 3, 5 and 7 for the three sizes |

With a high transmission probability and only a few initial infections, the disease grows as compact, roughly round outbreaks around the initial infections. Their cores are destroyed and their rims are currently infected. The target is reached in 7 to 9 days.

## 3. Unclustered maps: weak local spread and long jumps

With the local model alone, an unclustered map cannot be made. With a low transmission probability, the small outbreaks die out before the disease reaches the target. With a high probability, they grow into clusters.

The unclustered version therefore adds **long jumps**: every day, after the local spread, new infections appear at randomly chosen healthy plants, for instance carried there by flying insects. This is the mechanism the paper uses to describe the unclustered scenario: "the long jumps predominate".

| Parameter | Value |
|---|---|
| `p_transmission` | 0.05 |
| `infection_duration` | 5 days |
| `infection_seeds` | 2% of the target number of diseased plants |
| `spread_dimension` | 3 (only the immediate neighbors) |
| jumps per day | 4% of the target number of diseased plants |

The weak local spread still produces some small groups of two to a few plants, so the map is not merely uniform noise. The target is reached in 12 to 22 days.

The long jumps are implemented in `generate_disease_map`, outside the model. After `env.proceed(1)`, it sets the status of the chosen healthy plants to `infection_duration`, and recomputes the value field.

## 4. The same amount of disease in both maps of a pair

The target is `disease_fraction * size * size` diseased plants, with a default `disease_fraction` of 0.05. This is close to the hand-drawn clustered map.

1. **Growing:** both versions run day by day until the number of diseased plants reaches the target.
2. **Trimming:** the last day usually overshoots. The surplus is removed by healing randomly chosen plants among the newest infections, those with the most days left. Removing the newest infections changes the maps least: in the clustered version they are on the rims of the outbreaks, so the outbreaks keep their shape. Only if there are fewer newest infections than the surplus are the plants chosen among all infected plants.

As a result, both maps of a pair have exactly the target number of diseased plants.

## 5. Seeds

Every map has its own random generator. Its seed is derived from the provided seed, the width, the height and the version (0 for clustered, 1 for unclustered):

```python
int(np.random.SeedSequence([seed, width, height, version]).generate_state(1)[0])
```

The model's own generator (`env.random`) drives the initial infections, the daily spread, the long jumps and the trimming. A map therefore depends only on the seed, its size and its version. It does not depend on which other maps are generated, or in which order. The notebook checks that generating a map twice with the same seed gives the same map.

## 6. Results with seed 1

| Size | Version | Days | Diseased | Spots | Largest spot | Mean spot | In spots of size >= 10 |
|---|---|---|---|---|---|---|---|
| 30x30 | clustered | 7 | 45 | 3 | 21 | 15.0 | 84% |
| 30x30 | unclustered | 22 | 45 | 19 | 6 | 2.4 | 0% |
| 100x100 | clustered | 9 | 500 | 8 | 147 | 62.5 | 100% |
| 100x100 | unclustered | 12 | 500 | 200 | 16 | 2.5 | 11% |
| 200x200 | clustered | 9 | 2000 | 22 | 264 | 90.9 | 100% |
| 200x200 | unclustered | 12 | 2000 | 830 | 14 | 2.4 | 10% |

A spot is a connected group of diseased plants, with diagonal neighbors included. For comparison, the hand-drawn maps:
- **Clustered:** 491 plants, in 19 spots of mean size 25.8. It has six large outbreaks of about 100 plants each, plus isolated dots.
- **Unclustered:** 142 plants, in 85 spots of mean size 1.7.

**Differences from the hand-drawn maps:**
- **Isolated dots:** the generated clustered maps have no isolated dots, because they have no long jumps. If the dots are wanted, a small number of jumps per day can be added to the clustered version.
- **Shades:** the generated maps have two shades, infected (0.5) and destroyed (0.0), whereas the hand-drawn maps are black and white. Both shades count as diseased, so this does not change the VoI.
- **The 30x30 maps:** these are small. With 45 diseased plants, the clustered map has only three outbreaks. The distinction is clearest at 100x100 and 200x200.

## 7. Use in the environments

**Generator:** `generate_disease_map(width, height, version, seed, fraction, params=None, immunity_mask=None)` in `src/disease_maps.py`.
- It returns the epidemic model, whose value field is the map, and the number of days it ran.
- `params` defaults to `CLUSTERED` or `UNCLUSTERED`, the parameters of Sections 2 and 3.
- **Immunity mask:** cells marked `-2` are never infected, and the target is `fraction` of the remaining, plantable cells. The long jumps, too, only choose healthy plantable cells.
- `cluster_stats` and `diseased_count` are the statistics used above.

**Environment keys:** the epidemic fields of an environment exp/run take three new keys (`_defaults_environment.yaml`):

| Key | Default | Meaning |
|---|---|---|
| `tylcv-generated` | `null` | `clustered` or `unclustered`: the field is a generated map |
| `tylcv-generated-seed` | 1 | the seed of the map |
| `tylcv-generated-fraction` | 0.05 | the fraction of the tomato cells that are diseased |

The same keys exist for `ccr`.

**`apply_generated_maps(wbfe, exp_env)`** in `src/wbf_helper.py`, called by `create_wbfe`:
- It generates the map with the crop of the epidemic as the immunity mask: tomato for TYLCV, strawberry for CCR.
- It replaces the field with a `ScalarFieldEnvironment` holding the map, exactly like a static picture. The map is therefore the field on every day, whatever the precompute time and the `time-start-environment` of the runs.
- A field cannot have both a picture and a generated map; that raises an exception.
- **Caching:** the keys start with `tylcv-`, so they are part of the environment configuration that decides whether a cached precomputation is current. Changing the seed or the fraction recomputes the environment.

**The environment exp/runs** `environment/mrmr-generated-clustered-100` and `environment/mrmr-generated-unclustered-100`:
- size Miniberry-100, `planting: tomato-only`;
- seed 1, fraction 0.05, so 500 diseased plants each.

All six `mrmr2027-run` runs use them instead of the hand-drawn `mrmr-custom-clustered` and `mrmr-custom-notclustered`, which are kept but no longer used.

**Miniberry-200:** a 200x200 `MiniberryFarm` (scale 20), registered in `create_wbf` and `get_geometry` of `src/wbf_helper.py`, and in the legacy `create_wbfe` of `src/water_berry_farm.py`. Its nominal timesteps per day are 0.4 x 40000.

## 8. Next steps (not implemented)

1. **Larger scenarios:** MRMR environment exp/runs and runs on Miniberry-200, with more robots, for example 3 pioneers and 10 contractors.
2. **Several seeds:** run the MRMR comparisons over several map seeds (`tylcv-generated-seed`), with confidence intervals, as listed in the paper's criticism (`CRITICISM-Lotzi.md`, "Run multiple experiments over them with different randomness").
3. **Isolated dots:** optionally, a few long jumps per day in the clustered version, if the isolated dots of the hand-drawn map are wanted.
