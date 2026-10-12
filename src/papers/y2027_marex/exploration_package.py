"""
exploration_package.py

Classes of the MultiResolutionMultiRobot paper that implement exploration packages, decisions to explore certain areas.

"""

import math
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

    LAWNMOWERS = [ExplorationPackage.lawnmower_horizontal_bottom_left, ExplorationPackage.lawnmower_horizontal_bottom_right,
                  ExplorationPackage.lawnmower_horizontal_top_left, ExplorationPackage.lawnmower_horizontal_top_right]

    def find_shortest_path_ep(self, start, end=None, max_evaluations=None):
        """The shortest path that starts at start, covers every EP with one of its four lawnmower patterns, 
        and ends at end (if given). Returns the path as an array of points, and as a list of segments, dicts 
        labeled with the EP they cover (None for the start and the end).

        If the search space, n! EP orders times 4^n lawnmower directions, fits into max_evaluations (or 
        max_evaluations is None), the search is exhaustive and the path optimal. Otherwise, the path is 
        built greedily (greedy_path_ep). The bound is a count of evaluated combinations rather than a time, 
        so the result does not depend on the speed or the load of the machine, and a run is reproducible 
        from its seeds. Unlike an exhaustive search cut off after some combinations, the bound also holds 
        for any number of EPs: the space grows as n! 4^n, about 10^6 combinations per EP order for n = 10
        (DESIGN-MultiSeedEvaluation.md, Section 8)."""
        n = len(self.ep_to_explore)
        if max_evaluations is not None and math.factorial(n) * 4 ** n > max_evaluations:
            return self.greedy_path_ep(start, end)
        min_len = float('inf')
        best_path = None
        best_ep_path = None
        for perm in itertools.permutations(self.ep_to_explore):
            for gens in itertools.product(self.LAWNMOWERS, repeat=n):
                path = np.array([start])
                # FIXME: this fixes the fact that the path does not start 
                # with the start but then it breaks something else
                ep_path = [{"path": [start], "ep": None}]
                for generator, ep in zip(gens, perm):
                    newpath = generator(ep)
                    path = np.concatenate((path, newpath), axis=0)
                    ep_path.append({"path": newpath, "ep": ep})
                if end is not None:
                    path = np.concatenate((path, np.array([end])), axis=0)
                    ep_path.append({"path": [end], "ep": None})
                length = get_path_length(path)
                if length < min_len:
                    min_len = length
                    best_path = path
                    best_ep_path = ep_path
        return best_path, best_ep_path

    def greedy_path_ep(self, start, end=None):
        """A path built greedily: from the current position, take the remaining EP and lawnmower direction 
        with the smallest cost, the distance to the start of the lawnmower plus its length; then continue 
        from its end. Deterministic (ties go to the earlier EP and direction), and about 4 n^2 evaluations 
        for n EPs. The result has the format of find_shortest_path_ep."""
        current = np.array(start, dtype=float)
        remaining = list(self.ep_to_explore)
        path = np.array([start])
        ep_path = [{"path": [start], "ep": None}]
        while remaining:
            best = None
            for ep in remaining:
                for generator in self.LAWNMOWERS:
                    newpath = generator(ep)
                    cost = float(np.linalg.norm(np.asarray(newpath[0], dtype=float) - current)) + get_path_length(newpath)
                    if best is None or cost < best[0]:
                        best = (cost, ep, newpath)
            _, ep, newpath = best
            remaining.remove(ep)
            path = np.concatenate((path, newpath), axis=0)
            ep_path.append({"path": newpath, "ep": ep})
            current = np.asarray(newpath[-1], dtype=float)
        if end is not None:
            path = np.concatenate((path, np.array([end])), axis=0)
            ep_path.append({"path": [end], "ep": None})
        return path, ep_path
