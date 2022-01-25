import math
from functools import partial
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import timeit

import logging
logging.basicConfig(level=logging.WARNING)

class Environment:
    """The ancestor of all environment models: by default it is just an area [0,0, width, height]"""
    def __init__(self, width, height, seed):
        self.width, self.height = width, height
        self.random = np.random.default_rng(seed)
        
    def proceed(self, delta_t = 1.0):
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
        return self.value(i, j)

    def proceed(self, delta_t = 1.0):
        pass
    

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
        
    def proceed(self, delta_t = 1.0):
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
        logging.info(f"sum {self.value.sum()}")        
        
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
    def __init__(self, name, width, height, seed, p_transmission = 0.2, infection_duration = 5):
        super().__init__(name, width, height, seed)
        self.status = np.zeros((width, height))        
        self.p_transmission = p_transmission 
        self.p_infection = self.calculate_infection_matrix(self.p_transmission)
        self.infection_duration = infection_duration
    
    @staticmethod
    def calculate_infection_matrix(p_transmission): 
        """Calculates a matrix for the probability of the transmission function of the number of infected neighbors"""
        pr = np.ones(9) * (1 - p_transmission)
        pows = np.linspace(0, 8, 9)
        p_infection = np.ones(9) - np.power(pr, pows)
        return p_infection
    
    @staticmethod
    def countinfected_neighbor(statusmap):
        """Given a matrix with the status values, returns a matrix with the counts of the infected neighbors. Uses a convolution kernel for speedup"""
        infected = np.zeros(statusmap.shape)
        infected[statusmap > 0] = 1
        neighborcounting_kernel = np.array([[1,1,1], [1, 0, 1], [1, 1, 1]])
        infectedcount = signal.convolve2d(infected, neighborcounting_kernel, mode="same")
        return infectedcount

    def proceed(self, delta_t):
        """Updates the status field with the evolution"""
        # mark those that finished their sickness recovered...
        self.status[self.status==1] = -1
        # decrease the time remaining for infection
        self.status[self.status > 0] -= 1
        # propagate the infection
        infectedcount = self.countinfected_neighbor(self.status)
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
                                       frames=100, interval=5, 
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
    if True:
        # trying out different sizes of EpidemicSpreadEnvironment
        for i in [10, 50, 100, 200, 400, 1000, 2000, 4000]:
            env = EpidemicSpreadEnvironment("crop", i, i, seed=40, infection_duration = 5, p_transmission = 0.1)
            env.status[i // 2, i // 2] = 2
            env.status[(3*i) // 4, (3*i) // 4] = 5
            time = timeit.timeit("env.proceed(1.0)", number=1, globals=globals())
            print(f"map of size {i}x{i} a proceed took {time:0.2} seconds")
        # trying out different sizes of DissipationModelEnvironment
        for i in [10, 50, 100, 200, 400, 1000, 2000, 4000]:
            env = DissipationModelEnvironment("pollution", i, i, seed=40)
            # env.status[i // 2, i // 2] = 2
            # env.status[(3*i) // 4, (3*i) // 4] = 5
            time = timeit.timeit("env.proceed(1.0)", number=1, globals=globals())
            print(f"map of size {i}x{i} a proceed took {time:0.2} seconds")
