import pathlib
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from environment import Environment, EpidemicSpreadEnvironment, PrecalculatedEnvironment, ScalarFieldEnvironment, SoilMoistureEnvironment


class CountingEnvironment(Environment):
    def __init__(self, time_expansion):
        super().__init__(3, 3, seed=0, time_expansion=time_expansion)
        self.updates = []

    def inner_proceed(self, delta_t=1.0):
        self.updates.append((self.time, delta_t))


class IncrementEnvironment(ScalarFieldEnvironment):
    def inner_proceed(self, delta_t=1.0):
        self.value += delta_t


class TestEnvironment(unittest.TestCase):
    def test_time_expansion_updates_at_crossed_boundaries(self):
        env = CountingEnvironment(time_expansion=2)
        env.proceed(1)
        self.assertEqual(env.updates, [])
        env.proceed(4.5)
        self.assertEqual(env.updates, [(2, 2), (4, 2)])
        self.assertEqual(env.time, 5.5)

    def test_scalar_field_copies_initial_value(self):
        value = np.arange(9).reshape(3, 3)
        env = ScalarFieldEnvironment("field", 3, 3, seed=0, value=value)
        value[1, 1] = -1
        self.assertEqual(env.get(1.8, 1.2), 4)

    def test_epidemic_spread_respects_immunity(self):
        immunity = np.zeros((5, 5))
        immunity[2, 3] = -2
        env = EpidemicSpreadEnvironment(
            "disease", 5, 5, seed=0, p_transmission=1.0,
            infection_duration=3, spread_dimension=3,
            infection_seeds=0, immunity_mask=immunity)
        env.status[2, 2] = 2
        env.proceed(1)
        self.assertEqual(env.status[2, 2], 1)
        self.assertEqual(env.status[1, 2], 3)
        self.assertEqual(env.status[2, 3], -2)
        self.assertEqual(env.value[2, 3], 1.0)

    def test_soil_evaporation_without_rain(self):
        env = SoilMoistureEnvironment(
            "soil", 4, 4, seed=0, evaporation=0.2,
            rainfall=0.0, rain_likelihood=0.0, warmup_time=0)
        env.value.fill(1.0)
        env.proceed(1)
        np.testing.assert_allclose(env.value, 0.8)

    def test_precalculation_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            savedir = pathlib.Path(directory)
            source = IncrementEnvironment("source", 3, 3, seed=0)
            writer = PrecalculatedEnvironment(3, 3, source, savedir)
            writer.proceed(1)

            reader = PrecalculatedEnvironment(3, 3, None, savedir)
            reader.proceed(1)
            np.testing.assert_array_equal(reader.value, source.value)


if __name__ == "__main__":
    unittest.main()
