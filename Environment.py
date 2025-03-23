"""
environment.py

Classes that implement the general environment model and environment types with specific rules (epidemic spread etc. ). 

"""
import math
from functools import partial
import numpy as np
from scipy import signal
import scipy

import matplotlib.pyplot as plt
from matplotlib import animation, rc
import timeit
import pickle
import random
import pathlib

import logging
logging.basicConfig(level=logging.WARNING)


compress_ext = "gz"
# compress_ext = "gz"
if compress_ext == "bz2":
    import bz2 as compress
elif compress_ext == "gz":
    import gzip as compress
else:
    print(f"Undefined compression {compress_ext}")
    exit(1)


class Environment:
    """The ancestor of all environment models: by default it is just an area [0,0, width, height]"""
    def __init__(self, width, height, seed, time_expansion = 1):
        self.width, self.height = width, height
        self.time_expansion = time_expansion
        self.time = 0
        self.random = np.random.default_rng(seed)
        
    def proceed(self, delta_t = 1.0):
        nexttime = int(self.time / self.time_expansion) * self.time_expansion
        if nexttime - self.time <= delta_t:
            self.time = self.time + delta_t
            logging.info("Environment.proceed - calling the inner_proceed")
            self.inner_proceed(delta_t)
        else:
            self.time = self.time + delta_t
            logging.info("Environment.proceed - skipping the inner_proceed")

    def inner_proceed(self, delta_t = 1.0):
        pass

class ScalarFieldEnvironment(Environment):
    """A static environment which for each point it has a scalar field value. 
    This environment does not change: the dynamic aspect happens in the way it 
    is discovered."""
    
    def __init__(self, name, width, height, seed, value = None):
        super().__init__(width, height, seed)
        self.name = name # the name of the value
        if value != None:
            self.value = value.copy()
        else:
            self.value = np.zeros((self.width, self.height))
        
    def get(self, x, y):
        """Accessing by x and y is rounded to the integer value"""
        i = int(x)
        j = int(y)
        return self.value[i, j]

    

class DissipationModelEnvironment(ScalarFieldEnvironment):
    """An environment which models values that are gradually spreading and dissipating. This might be something such as water, fertilizer, or also polution"""
    
    def __init__(self, name, width, height, seed, dissipation_rate = 0.95, 
                 p_pollution = 0.1, pollution_max_strength = 1000, evolve_speed = 1):
        super().__init__(name, width, height, seed) 
        self.evolve_speed = evolve_speed # measured in convolutions / second
        self.p_pollution = p_pollution
        self.pollution_max_strength = pollution_max_strength
        self.dissipation_rate = dissipation_rate
        self.evolve_kernel = np.array([[0.3, 0.5, 0.3],[0.5, 1, 0.5],[0.3, 0.5, 0.3]])
        self.evolve_kernel = self.evolve_kernel * (self.dissipation_rate / self.evolve_kernel.sum())
        
    def inner_proceed(self, delta_t = 1.0):
        """Apply the polution events. Applyies the evolution kernel a number of times 
        determined by the evolve speed"""
        self.pollution_init(delta_t)
        iterations = int(self.evolve_speed * delta_t)
        # initial implementation: create an space 9 times larger, this is very inefficient
        #convspace = np.zeros((self.width * 3, self.height * 3))
        #convspace[self.width: self.width *2, self.height: self.height*2] = #self.value
        # create a padding model FIXME: intelligent padding
        padding = 1 + iterations * self.evolve_kernel.shape[0] // 2
        convspace = np.zeros((self.width + 2 * padding, self.height + 2 * padding))
        convspace[padding: self.width + padding, padding: self.height + padding] = self.value
        for i in range(iterations):
            convspace = signal.convolve2d(convspace, self.evolve_kernel, mode="same")
        self.value = convspace[padding: self.width + padding, padding: self.height + padding]
        # logging.info(f"sum {self.value.sum()}")        
        
    def pollution_init(self, delta_t):
        """One or more new pollution sources appear"""
        val = self.random.random()
        pollution_event_count = int(delta_t * self.p_pollution / val)
        for i in range(0, pollution_event_count):            
            rands = self.random.random(3)
            x = int(rands[0] * self.width-7) # the -7 is due to the way the convolution works
            y = int(rands[1] * self.height-7)            
            #x = self.random.randint(0, self.width-7)
            #y = self.random.randint(0, self.height-7)
            strength = rands[2]
            self.value[x,y]=strength * self.pollution_max_strength
    

class EpidemicSpreadEnvironment(ScalarFieldEnvironment):
    """An environment modeling the spreading of an epidemic such as a plant disease in the field. 

    The model adjusts the "status" field which is defined as follows:
       0 - Susceptible (aka healthy)
       >=1 - Infected - currently spreading infection, the exact number is the number of days remaining until recovery or destruction. 
       -1 - Recovered and thus immune (or destroyed and thus not spreading any more)
       -2 - Natural immunity - the epidemic cannot touch this location (eg. different plants). It can be used to set a filter.

    The "value" field is created from the status field and it will have the values:
        0.0 - Healthy
        0.5 - Infected

    The immunity mask must be a map of the same size as the environment, but the values that are immune need to be set to -2 

    """
    def __init__(self, name, width, height, seed, p_transmission = 0.2, infection_duration = 5, spread_dimension = 11, infection_seeds = -1, infection_seeds_initial = 5, immunity_mask = None):
        super().__init__(name, width, height, seed)
        if immunity_mask is None:
            self.status = np.zeros((width, height))        
        else:
            self.status = np.copy(immunity_mask)
        # the size of the spread matrix
        self.spread_dimension = spread_dimension
        self.change_p_transmission(p_transmission)
        self.infection_duration = infection_duration
        self.infection_seeds = infection_seeds
        # if set to -1, make the number of seeds 1 per 100 squares
        if self.infection_seeds == -1:
            self.infection_seeds = max(1, int(height * width / 100))
        self.infection_seeds_initial = infection_seeds_initial
        self.initial_infection()

    def initial_infection(self):
        """Creates a random initial infection but only in the areas that are not immune"""
        for i in range(self.infection_seeds):
            x = self.random.integers(0, self.width-1)
            y = self.random.integers(0, self.height-1)
            if self.status[x,y] != -2.0:
                self.status[x, y] = self.infection_seeds_initial

    def change_p_transmission(self, p_transmission):
        self.p_transmission = p_transmission 
        self.p_infection = self.calculate_infection_matrix(self.p_transmission, self.spread_dimension)

    @staticmethod
    def calculate_infection_matrix(p_transmission, dimension): 
        """Calculates a matrix for the probability of the transmission function of the number of infected neighbors"""
        length = dimension * dimension
        pr = np.ones(length) * (1 - p_transmission)
        pows = np.linspace(0, length-1, length)
        p_infection = np.ones(length) - np.power(pr, pows)
        logging.info(f"Infection matrix: {p_infection}")
        return p_infection
    
    @staticmethod
    def countinfected_neighbor(statusmap, dimension):
        """Given a matrix with the status values, returns a matrix with the counts of the infected neighbors. Uses a convolution kernel for speedup """
        # in the statusmap, everything that is larger than 0 is infected
        infected = np.zeros(statusmap.shape)
        infected[statusmap > 0] = 1
        center = dimension // 2 
        neighborcounting_kernel = np.zeros((dimension, dimension))
        for i in range(dimension):
            for j in range(dimension):
                dist2 = (i - center)*(i - center) + (j - center)*(j - center)
                if dist2 != 0.0:
                    neighborcounting_kernel[i,j] = 1 / dist2
        logging.info("convolve2d starting")
        infectedcount = signal.convolve2d(infected, neighborcounting_kernel, mode="same")
        logging.info("convolve2d done")
        return infectedcount

    def inner_proceed(self, delta_t):
        """Updates the status field with the evolution"""
        # mark those that finished their sickness "recovered" aka destroyed...
        self.status[self.status==1] = -1
        # decrease the time remaining for infection
        self.status[self.status > 0] -= 1
        # propagate the infection
        infectedcount = self.countinfected_neighbor(self.status, self.spread_dimension)
        infectionlikelihood = self.p_infection[infectedcount.astype(int)]
        p = self.random.random(self.status.shape)
        il = infectionlikelihood 
        il[il > p] = 1
        il[il <= p] = 0
        self.status[np.logical_and(self.status== 0, il == 1)] = self.infection_duration
        self.create_value()

    def create_value(self):
        """Make the value map. This model assumes that an R state means destroyed. Value 1 means that there is a full value in the field."""
        self.value = np.ones(self.status.shape)
        # set the infected 
        self.value[self.status > 0] = 0.5
        # R means destroyed
        self.value[self.status == -1] = 0
        # the masked ones are never infected
        self.value[self.status == -2.0] = 1.0
        return self.value

class PrecalculatedEnvironment(ScalarFieldEnvironment):
    """An environment for which the values are pre-calculated and saved on a location. This is useful for simulations where the values do not depend on the rest of the simulation."""
    
    def __init__(self, width, height, environment, savedir):
        """If the environment is None, we are replaying a pre-calculated one from ../__Temporary/current-dir/savedir
        If the environment not None, we run that environment and save it into the save dir."""
        super().__init__("precalculated", width, height, seed=0)
        self.environment = environment
        # p = pathlib.Path.cwd()
        self.savedir = savedir
        self.savedir.mkdir(parents=True, exist_ok = True)
        self.previous_loaded = -1 # timestamp of previously loaded env if any
        
    def get_filenames(self, timestamp):
        """"""
        file_value = pathlib.Path(self.savedir, f"env_value_{timestamp:05d}.{compress_ext}")
        jpg_value = pathlib.Path(self.savedir, f"env_value_{timestamp:05d}.jpg")
        return file_value, jpg_value

    def inner_proceed(self, delta_t):
        timestamp = int(self.time)
        logging.info(f"PrecalculatedEnvironment at timestamp {timestamp}")
        file_value, jpg_value = self.get_filenames(timestamp)
        if self.environment == None: # loading
            # search for the previous existing
            while not file_value.exists() and timestamp > self.previous_loaded:
                timestamp = timestamp - 1
                file_value, _ = self.get_filenames(timestamp)                
            if not file_value.exists():
                # raise Exception(f"Saved value {file_value} does not exist")
                logging.info(f"Saved value {file_value} does not exist - assuming no change.")
                return
            # with open(file_value, "rb") as f:
            logging.info(f"Loading from {compress_ext} {file_value}")
            #with bz2.open(file_value, "rb") as f:
            with compress.open(file_value, "rb") as f:
                 self.value = pickle.load(f)
            self.previous_loaded = timestamp
            logging.info(f"Loading from {compress_ext} {file_value} done")                 
            return
        else:
            if file_value.exists():
                #raise Exception(f"PrecalculatedEnvironment: file {file_value} already exists, skipping!")
                logging.info(f"PrecalculatedEnvironment: file {file_value} already exists, skipping!")
                return
        # assume that the environment is passed:
        self.environment.proceed(delta_t)

        #with open(file_value, "wb") as f:
        logging.info(f"Saving to {compress_ext}")
        with compress.open(file_value, "wb") as f:
        #with bz2.open(file_value, "wb") as f:
            pickle.dump(self.environment.value, f)
        logging.info(f"Saving to {compress_ext} done")
        # create a jpg file that shows stuff
        plt.ioff()
        if isinstance(self.environment, EpidemicSpreadEnvironment):
            fig, (axvalue, axstatus) = plt.subplots(1,2)
            axesimage  = axvalue.imshow(self.environment.value, vmin=0, vmax=1, origin="lower", cmap="gray")
            axvalue.set_title("value")
            axesimage  = axstatus.imshow(self.environment.value, vmin=-2, vmax=10, origin="lower", cmap="gray")
            axstatus.set_title("status")
            fig.suptitle(f"{self.environment.name} t={timestamp}")
        else:
            fig, ax = plt.subplots()
            fig.suptitle(f"{self.environment.name} t={timestamp}")
            axesimage  = ax.imshow(self.environment.value, vmin=0, vmax=1, origin="lower", cmap="gray")
        
        plt.savefig(jpg_value, bbox_inches='tight')
        plt.close(fig)

class SoilMoistureEnvironment(ScalarFieldEnvironment):
    """
    A simple model for the modeling of the soil moisture. It assumes a specific probabilities and quantities of rainfall, a specific multiplicative evaporation ratio, and the assumption that the rainfall is distributed in the form of a 2D gaussian. 

    For a more sophisticated model of soil humidity, a starting point might be
    https://www.cpc.ncep.noaa.gov/soilmst/paper.html
    """
    def __init__(self, name, width, height, seed, evaporation = 0.02, rainfall = 0.3, rain_likelihood = 0.2, warmup_time = 10):
        super().__init__(name, width, height, seed)
        #self.value = self.get_fuzzy_distribution()
        self.value = self.get_mixture_of_gaussians(count = 30, humpwidth = min(self.width, self.height))
        # the percentage of water that evaporates / day
        self.evaporation = evaporation
        # average rainfall (per area)
        self.rainfall = rainfall
        # rain likelihood
        self.rain_likelihood = rain_likelihood
        # warmup: iterate a number of times, without advancing the time
        self.warmup_time = warmup_time
        for i in range(self.warmup_time):
            self.inner_proceed(1)

    def inner_proceed(self, delta_t=1):
        # evaporation 
        self.value = (1 - self.evaporation) * self.value
        # rain
        if self.random.random() < self.rain_likelihood:
            rainquantity = self.random.uniform(0, self.rainfall * 2)
            # rain = self.get_fuzzy_distribution() * rainquantity
            rain_pattern = self.get_mixture_of_gaussians(count = 10, humpwidth = min(self.width, self.height))
            rain_pattern = rain_pattern / np.sum(rain_pattern)
            rain_quantity = (self.width * self.height * rainquantity) * rain_pattern
            self.value = self.value + rain_quantity
            self.value = np.minimum(self.value, 1.0)

    def shifted_add(self, a, b, x, y):
        """Add matrix b to matrix a shifted by positions x and y. 
        Ignore overflow at the margins.
        """
        xA, yA = a.shape
        xB, yB = b.shape

        fromAX = max(0, x)
        toAX = min(xB + x, xA)
        fromAY = max(0, y)
        toAY = min(yB + y, yA)

        fromBX = 0 + max(0, -x)
        toBX = xB + min(0, xA - x - xB)  
        fromBY = 0 + max(0, -y)
        toBY = yB + min(0, yA - y - yB)

        if toBX <= 0 or toBY <= 0:
            return
        if fromBX >= xB or fromBY >= yB:
            return

        a[fromAX:toAX, fromAY:toAY] = a[fromAX:toAX, fromAY:toAY] + b[fromBX:toBX, fromBY:toBY]
        return

    def create_gaussian_hump(self, width, height, mean = [0,0], cov = [[1, 0], [0, 1]]):
        """Fill an array of size width times height with an Gaussian 
        function of the specified mean and covariance matrix."""
        xv = np.linspace(0, width, width)
        yv = np.linspace(0, height, height)
        x, y = np.meshgrid(xv, yv)
        positions = np.column_stack((x.ravel(),y.ravel()))
        value = scipy.stats.multivariate_normal.pdf(positions, mean, cov)
        vals = np.reshape(value, [height, width])
        return vals


    def get_fuzzy_distribution(self):
        """Creates a fuzzy distribution for modeling the rain quantity. It starts by overlying a number of rectangular patches of random size and location with the value. Then it performs a series of convolutions. 
        """
        kernel = np.array([[0.1, 0.1, 0.1],[0.1, 0.2, 0.1],[0.1, 0.1, 0.1]])
        initvalue = np.full([self.width, self.height], 0.5)
        patches = int(2 * math.sqrt(min(self.height, self.width)))
        patchsize = int(0.2 * min(self.height, self.width))
        convolutions = int(0.8 * min(self.height, self.width))
        for patch in range(patches):
            size = self.random.integers(1,patchsize)
            patchvalue = self.random.uniform(0, 1)
            locx = self.random.integers(0, self.width - size)
            locy = self.random.integers(0, self.height - size)
            initvalue[locx:locx+size, locy:locy+size] = patchvalue
        value = np.copy(initvalue)
        for i in range(convolutions):
            print(f"fuzzy distr {i}/{convolutions}")
            value = signal.convolve2d(value, kernel, mode="same", boundary="symm")
        return value

    def get_mixture_of_gaussians(self, count, humpwidth):
        """Returns an array filled in with a matrix."""
        value = np.zeros([self.width, self.height])
        humph = humpw = humpwidth
        hump = self.create_gaussian_hump(humpw, humph, mean = [humpw/2, humph/2], cov = [[(humpw / 8)**2,0], [0, (humph/8)**2]])
        #plt.imshow(hump, cmap = "gray_r")
        for i in range(count):
            #locx = int(self.random.random() * self.width - humpw/2) 
            #locy = int(self.random.random() * self.height - humph/2) 
            locx = self.random.integers(-humpw/2, self.width + humpw/2)
            locy = self.random.integers(-humph/2, self.height + humph/2)
            #print(f"Adding at {locx} and {locy}")    
            self.shifted_add(value, hump, locx, locy)
        return value

#
# FIXME: remove them from here, this should just go into the Environment-experiments
#

def animate(env, axesimage, i):
    """Animates an environment by letting the environment proceed a timestep, then setting the values into image."""
    env.proceed(1.0)
    v = env.value.copy()
    axesimage.set_array(v)
    return [axesimage]

def animate_environment(env, frames=99, interval=5):
    """Creates an animated figure of the environment. Call plt.show after this. Interval specifies the speed of the animation."""
    fig, ax = plt.subplots()
    # im  = ax.imshow(env.value, cmap="gray", vmin=0, vmax=1.0)
    axesimage  = ax.imshow(env.value, vmin=0, vmax=1, cmap="gray")
    # Good try, but this doesn't work like this, it should be set from 
    # animate
    # ax.set_title(f"{env.name} time = {env.time}")
    anim = animation.FuncAnimation(fig, partial(animate, env, axesimage), 
                                       frames=frames, interval=interval, 
                                       blit=True)
    return anim

