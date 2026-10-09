"""
exploration_package.py

Classes of the MultiResolutionMultiRobot paper that implement exploration packages, decisions to explore certain areas.

"""

import numpy as np
import itertools
from path_generators import get_path_length

class ExplorationPackage:
    """Implements an area that needs to be explored with a certain resolution"""
    def __init__(self, x_min, x_max, y_min, y_max, step):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.step = step
        self.path = None
        
    def __repr__(self):
        retval = f"ExplorationPackage x=[{self.x_min},{self.x_max}] " +         f"y=[{self.y_min}, {self.y_max}] step={self.step}"
        return retval

    def overlap(self, other):
        """Checks if this package overlaps with another"""
        return not (self.x_max <= other.x_min or
                    self.x_min >= other.x_max or
                    self.y_max <= other.y_min or
                    self.y_min >= other.y_max)


    def clip(self, path):
        """The path inside the package: the last row of a lawnmower can overshoot the package by up to 
        two steps, which near the border of the field would lead outside the field"""
        return np.clip(np.array(path), [self.x_min, self.y_min], [self.x_max, self.y_max])

    def lawnmower_horizontal_bottom_left(self, shift=[0,0]):
        """Generates a horizontal lawnmower path, that starts at the bottom left, which is at x_min, y_min, and proceeds in the direction of higher y"""
        current = [self.x_min, self.y_min]
        path = []
        path.append(current)
        while True:
            path.append([self.x_max, current[1]])
            path.append([self.x_max, current[1]+self.step])
            path.append([self.x_min, current[1]+self.step])
            current = [self.x_min, current[1]+2 * self.step]
            path.append(current)
            if current[1] + self.step > self.y_max:
                break
        path.append([self.x_max, current[1]])
        return self.clip(path) + shift

    def lawnmower_horizontal_bottom_right(self, shift=[0,0]):
        """Generates a horizontal lawnmower path, that starts at the bottom right, which is at x_max, y_min, and proceeds in the direction of higher y"""
        current = [self.x_max, self.y_min]
        path = []
        path.append(current)
        while True:
            path.append([self.x_min, current[1]])
            path.append([self.x_min, current[1]+self.step])
            path.append([self.x_max, current[1]+self.step])
            current = [self.x_max, current[1]+2 * self.step]
            path.append(current)
            if current[1] + self.step > self.y_max:
                break
        path.append([self.x_min, current[1]])
        return self.clip(path) + shift

    def lawnmower_horizontal_top_left(self, shift=[0,0]):
        """Generates a horizontal lawnmower path, that starts at the top left, which is at x_min, y_max, and proceeds in the direction of lower y"""
        current = [self.x_min, self.y_max]
        path = []
        path.append(current)
        while True:
            path.append([self.x_max, current[1]])
            path.append([self.x_max, current[1]-self.step])
            path.append([self.x_min, current[1]-self.step])
            current = [self.x_min, current[1]-2 * self.step]
            path.append(current)
            if current[1] - self.step < self.y_min:
                break
        path.append([self.x_max, current[1]])
        return self.clip(path) + shift

    def lawnmower_horizontal_top_right(self, shift=[0,0]):
        """Generates a horizontal lawnmower path, that starts at the top left, which is at x_max, y_max, and proceeds in the direction of lower y"""
        current = [self.x_max, self.y_max]
        path = []
        path.append(current)
        while True:
            path.append([self.x_min, current[1]])
            path.append([self.x_min, current[1]-self.step])
            path.append([self.x_max, current[1]-self.step])
            current = [self.x_max, current[1]-2 * self.step]
            path.append(current)
            if current[1] - self.step < self.y_min:
                break
        path.append([self.x_min, current[1]])
        return self.clip(path) + shift

class ExplorationPackageSet: 
    """A class having a set of exploration packages. Code for creating optimal traversals"""    
    
    def __init__(self):        
        self.ep_to_explore = []
        self.ep_explored = []

    def add_ep(self, ep: ExplorationPackage):
        """Adds an exploration package to explore. Returns true if the addition was successful. Returns false if it is unsuccessful. Unsuccessful either means that it had already been explored, or it is already in the list.
        FIXME: for the time being it always succeeds
        """
        self.ep_to_explore.append(ep)

    def find_shortest_path_ep(self, start, end=None, max_evaluations=None):
        """Tries every combination of traversal directions to find the optimal one. This is a very expensive function, with a computational complexity of n!*4^n. Realistically, it can only be run up to n=5, where it takes about 10 seconds. 
        max_evaluations bounds the number of (order, directions) combinations evaluated: the search stops 
        after the EP order during which it is reached, and returns the best path so far. None searches 
        exhaustively. A count rather than a time limit makes the result independent of the speed and the 
        load of the machine, so a run is reproducible from its seeds (DESIGN-MultiSeedEvaluation.md, Section 8). 
        On the development machine, about 15000-30000 combinations are evaluated per second.
        
        Returns the path in the form of a list of dicts labeled with the EPs that are part of it

        """
        choices = [ExplorationPackage.lawnmower_horizontal_bottom_left, ExplorationPackage.lawnmower_horizontal_bottom_right, ExplorationPackage.lawnmower_horizontal_top_left, ExplorationPackage.lawnmower_horizontal_top_right]
        min_len = float('inf')
        best_path = None
        best_ep_path = None

        count = 0
        for perm in itertools.permutations(self.ep_to_explore):
            generator_choices = itertools.product(
                choices, repeat=len(self.ep_to_explore))
            for gens in generator_choices:
                path = np.array([start])
                # FIXME: this fixes the fact that the path does not start 
                # with the start but then it breaks something else
                ep_path = []
                # append the first segment, not an ep
                ep_path.append({"path": [start], "ep": None})
                intrinsic = 0
                for generator, ep in zip(gens, perm):
                    # print(generator.__name__)
                    newpath = generator(ep)
                    # print(f"path length: {generator} {get_path_length(newpath)}")
                    intrinsic += get_path_length(newpath)
                    path = np.concatenate((path, newpath), axis=0)
                    ep_path.append({"path": newpath, "ep": ep})
                # FIXME: the ep_path should also have this
                if end is not None:
                    path = np.concatenate((path, np.array([end])), axis=0)
                    # append the last segment, not an ep
                    ep_path.append({"path": [end], "ep": None})
                
                length = get_path_length(path)
                count += 1
                # print(f"{count} Lenght of current path: {length} intrinsic {intrinsic}")
                if length < min_len:
                    min_len = length
                    # print(length)
                    best_path = path
                    best_ep_path = ep_path
            if max_evaluations is not None and count >= max_evaluations:
                return best_path, best_ep_path
             
        return best_path, best_ep_path        
