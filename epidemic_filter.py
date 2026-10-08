"""
epidemic_filter.py

A model-based estimator of a disease field: a localized particle filter over the epidemic spread model.
"""

import numpy as np
from scipy.spatial.distance import cdist

from environment import EpidemicSpreadEnvironment
from information_model import (AbstractScalarFieldIM, DISEASED, grid_cells, latest_per_cell,
                               probability_uncertainty)
from water_berry_farm import default_infection_seeds, default_spread_dimension


class EpidemicParticleFilterIM(AbstractScalarFieldIM):
    """A particle filter over the epidemic model of a disease field.

    The prior is an ensemble of particles, each the state after `days` days of an epidemic with random
    seeds, under the parameters assumed by the estimator (not the environment's: the estimator never sees
    the environment's seed, and its parameters may be mismatched on purpose). The estimator does not know
    which cells are planted with the crop, so the particles may also infect cells of the other crop.

    The observations weight the particles with the likelihood of the observed diseased / healthy state
    (observation_noise is the probability of a wrong state). The weights are localized: the estimate of a
    cell uses only the observations within localization_radius, which avoids the collapse of a global
    weighting to a single particle. Cells without nearby observations keep the prior. Within a day the
    environment is static, so the particles are weighted, not propagated."""
    UNCERTAINTY = "probability"

    def __init__(self, width, height, default_value = 1.0, particles = 200, p_transmission = 0.25,
                 infection_duration = 5, infection_seeds = None, spread_dimension = None, days = 6,
                 localization_radius = 5.0, observation_noise = 0.05, seed = 0):
        super().__init__(width, height, default_value)
        self.particles, self.days, self.seed = particles, days, seed
        self.p_transmission, self.infection_duration = p_transmission, infection_duration
        self.infection_seeds = default_infection_seeds(width) if infection_seeds is None else infection_seeds
        self.spread_dimension = default_spread_dimension(width) if spread_dimension is None else spread_dimension
        self.localization_radius, self.observation_noise = localization_radius, observation_noise
        self.ensemble = None # (particles, width * height) values of the prior particles, created when first needed

    def prior_ensemble(self):
        """The values of the particles after `days` days of epidemics with seeds seed, seed + 1, ..."""
        values = []
        for i in range(self.particles):
            epidemic = EpidemicSpreadEnvironment("particle", self.width, self.height, self.seed + i,
                p_transmission=self.p_transmission, infection_duration=self.infection_duration,
                spread_dimension=self.spread_dimension, infection_seeds=self.infection_seeds)
            epidemic.proceed(self.days)
            values.append(epidemic.value.ravel())
        return np.array(values)

    def estimate(self, observations, prior_value, prior_uncertainty):
        if self.ensemble is None:
            self.ensemble = self.prior_ensemble()
        diseased = self.ensemble < DISEASED
        cells, values = latest_per_cell(observations)
        if len(values) == 0:
            weights = np.full(self.ensemble.shape, 1.0 / self.particles)
        else:
            # the log-likelihood of every observation under every particle: (particles, observations)
            index = cells[:, 0] * self.height + cells[:, 1]
            match = diseased[:, index] == (values < DISEASED)
            loglikelihood = np.where(match, np.log(1 - self.observation_noise), np.log(self.observation_noise))
            # localization: every cell uses the observations within the radius: (cells, observations)
            nearby = (cdist(grid_cells(self.width, self.height), cells) <= self.localization_radius) * 1.0
            logweights = loglikelihood @ nearby.T
            weights = np.exp(logweights - logweights.max(axis=0))
            weights /= weights.sum(axis=0)
        probability = (weights * diseased).sum(axis=0).reshape(self.width, self.height)
        value = (weights * self.ensemble).sum(axis=0).reshape(self.width, self.height)
        # the observed cells are known
        probability[cells[:, 0], cells[:, 1]] = values < DISEASED
        value[cells[:, 0], cells[:, 1]] = values
        self.probability = probability
        return value, probability_uncertainty(probability)
