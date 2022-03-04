import math
from functools import partial
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import timeit
import pickle
import bz2
import pathlib

import logging
logging.basicConfig(level=logging.WARNING)

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
            logging.info("Environment.proceed - calling the inner_proceed")
            self.inner_proceed(delta_t)
        else:
            logging.info("Environment.proceed - skipping the inner_proceed")
        self.time = self.time + delta_t

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

    The "value" field is created from the status field. 

    """
    def __init__(self, name, width, height, seed, p_transmission = 0.2, infection_duration = 5, spread_dimension = 11):
        super().__init__(name, width, height, seed)
        self.status = np.zeros((width, height))        
        self.spread_dimension = spread_dimension
        self.change_p_transmission(p_transmission)
        self.infection_duration = infection_duration
        ## the size of the spread matrix

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
        """Given a matrix with the status values, returns a matrix with the counts of the infected neighbors. Uses a convolution kernel for speedup
        FIXME: this kernel creates a square spread
        FIXME: maybe instead of this use scipy.ndimage.filters.gaussian_filter"""
        infected = np.zeros(statusmap.shape)
        infected[statusmap > 0] = 1
        ## This was too square
        # neighborcounting_kernel = np.array([[1,1,1], [1, 0, 1], [1, 1, 1]])
        # neighborcounting_kernel = np.ones((dimension, dimension))
        # neighborcounting_kernel[dimension//2 + 1, dimension//2 + 1] = 0

        center = dimension // 2 + 1
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
        # mark those that finished their sickness recovered...
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
        # the filtered one is not valuable either
        self.value[self.status == -2] = 0
        return self.value

class PrecalculatedEnvironment(ScalarFieldEnvironment):
    """An environment for which the values are pre-calculated and saved on a location. This is useful for simulations where the values do not depend on the rest of the simulation."""
    
    def __init__(self, width, height, environment, savedir):
        """If the environment is None, we are replaying a pre-calculated one from ../__Temporary/current-dir/savedir
        If the environment not None, we run that environment and save it into the save dir."""
        super().__init__("precalculated", width, height, seed=0)
        self.environment = environment
        p = pathlib.Path.cwd()
        self.savedir = pathlib.Path(p.parent, "__Temporary", p.name, savedir)
        self.savedir.mkdir(parents=True, exist_ok = True)
        
    def inner_proceed(self, delta_t):
        timestamp = int(self.time)
        logging.info(f"PrecalculatedEnvironment at timestamp {timestamp}")
        file_value = pathlib.Path(self.savedir, f"env_value_{timestamp:05d}.bz2")
        jpg_value = pathlib.Path(self.savedir, f"env_visual_{timestamp:05d}.jpg")
        if self.environment == None:
            if not file_value.exists():
                # raise Exception(f"Saved value {file_value} does not exist")
                logging.info(f"Saved value {file_value} does not exist - assuming no change.")
                return
            # with open(file_value, "rb") as f:
            logging.info("Loading from bz2")
            with bz2.open(file_value, "rb") as f:
                 self.value = pickle.load(f)
            logging.info("Loading from bz2 done")                 
            return
        # assume that the environment is passed:
        self.environment.proceed(delta_t)

        #with open(file_value, "wb") as f:
        logging.info("Saving to bz2")
        with bz2.open(file_value, "wb") as f:
            pickle.dump(self.environment.value, f)
        logging.info("Saving to bz2 done")        
        fig, ax = plt.subplots()
        axesimage  = ax.imshow(self.environment.value, vmin=0, vmax=1)
        plt.savefig(jpg_value, bbox_inches='tight')
        plt.close(fig)



def animate(env, axesimage, i):
    """Animates an environment by letting the environment proceed a timestep, then setting the values into image."""
    env.proceed(1.0)
    v = env.value.copy()
    axesimage.set_array(v)
    return [axesimage]

def animate_environment(env):
    fig, ax = plt.subplots()
    # im  = ax.imshow(env.value, cmap="gray", vmin=0, vmax=1.0)
    axesimage  = ax.imshow(env.value, vmin=0, vmax=1)
    anim = animation.FuncAnimation(fig, partial(animate, env, axesimage), 
                                       frames=99, interval=5, 
                                       blit=True)
    return anim

if __name__ == "__main__":
    if False:
        # visualizing the environment
        env = EpidemicSpreadEnvironment("crop", 100, 100, seed=40, infection_duration = 5, p_transmission = 0.1)
        env.status[6,10] = 2
        env.status[60,80] = 5
        # make a filtered area which won't be susceptible 
        env.status[10:50, 10:50] = -2
        anim = animate_environment(env)
        plt.show()
    if False:
        # trying out different sizes of EpidemicSpreadEnvironment
        print("EpidemicSpreadEnvironment")
        for i in [10, 50, 100, 200, 400, 1000, 2000, 4000]:
            env = EpidemicSpreadEnvironment("crop", i, i, seed=40, infection_duration = 5, p_transmission = 0.1)
            env.status[i // 2, i // 2] = 2
            env.status[(3*i) // 4, (3*i) // 4] = 5
            time = timeit.timeit("env.proceed(1.0)", number=1, globals=globals())
            print(f"map of size {i}x{i} a proceed took {time:0.2} seconds")
        # trying out different sizes of DissipationModelEnvironment
        print("DissipationModelEnvironment")
        for i in [10, 50, 100, 200, 400, 1000, 2000, 4000]:
            env = DissipationModelEnvironment("pollution", i, i, seed=40)
            # env.status[i // 2, i // 2] = 2
            # env.status[(3*i) // 4, (3*i) // 4] = 5
            time = timeit.timeit("env.proceed(1.0)", number=1, globals=globals())
            print(f"map of size {i}x{i} a proceed took {time:0.2} seconds")
    if False:
        ## saving
        height = 2000
        width = 2000
        env = EpidemicSpreadEnvironment("crop", width, height, seed=40, infection_duration = 5, p_transmission = 0.1)
        env.status[width // 2, height // 2] = 2
        env.status[(3*width) // 4, (3*height) // 4] = 5
        precenv = PrecalculatedEnvironment(env, "precalc")
        for t in range(0, 100):
            precenv.proceed(1.0)
    if True:
        ## reloading
        pe = PrecalculatedEnvironment(2000, 2000, None,"precalc")
        anim = animate_environment(pe)
        plt.show()

    if False:
        # creating a spread area
        dimension = 7
        center = dimension // 2 + 1
        neighborcounting_kernel = np.zeros((dimension, dimension))
        for i in range(dimension):
            for j in range(dimension):
                dist2 = (i - center)*(i - center) + (j - center)*(j - center)
                if dist2 != 0.0:
                    neighborcounting_kernel[i,j] = 1 / dist2
        print(neighborcounting_kernel)
