"""
grid_limited_randomness.py

The algorithms for the paper:

Samuel Matloob, Ayan Dutta, O. Patrick Kreidl, Damla Turgut and Ladislau Bölöni. Exploring the tradeoffs between systematic and random exploration in mobile sensors. In Proc. of 26th. Int. Conf. on Modeling, Analysis and Simulation of Wireless and Mobile Systems (MSWIM-2023), October 2023.

"""
import copy
import numpy as np
from exp_run_config import Experiment
from wbf_simulate import get_geometry
from policy import FollowPathPolicy
from path_generators import euclidean_distance, get_path_length

from . import christofidesV2 

def GetXYRangeRelatedToCellNumber(cellNumber, numberOfHorizontalCells, numberOfVerticalCells, x_min, x_max, y_min, y_max):
    """Return the coordinates of a specific cell. The cells are enumerated in a back-and-forth lawnmower order. The valid cell numbers start from 1 and extend to h x v inclusive. """
    xDistanceInEachCell= (x_max-x_min)/numberOfHorizontalCells
    yDistanceInEachCell= (y_max-y_min)/numberOfVerticalCells
    cellRowLocation= (int)((int)(cellNumber-1 + numberOfHorizontalCells)/ numberOfHorizontalCells)
    direction=1
    if cellRowLocation %2==0:
        direction=-1 
    else:
        direction=1

    if direction==1:
        x1= x_min+xDistanceInEachCell*((cellNumber-1)%numberOfHorizontalCells)
        x2= x_min+xDistanceInEachCell+xDistanceInEachCell*((cellNumber-1)%numberOfHorizontalCells)
    else:
        x1= x_min+xDistanceInEachCell*((numberOfHorizontalCells-cellNumber)%numberOfHorizontalCells)
        x2= x_min+xDistanceInEachCell*(1+((numberOfHorizontalCells-cellNumber)%numberOfHorizontalCells))

    y1= y_min+yDistanceInEachCell*(cellRowLocation-1)
    y2= y_min+yDistanceInEachCell+yDistanceInEachCell*(cellRowLocation-1)

    return x1, x2, y1, y2

def add_to_shortest_detour(path, new_point):
    """Insert the new point into the path into the location where it requires the shortest detour"""
    if len(path) == 1:
        path.append(new_point)
        return path

    distances = np.array([euclidean_distance(point, new_point) for point in path])
    detours = distances[:-1] + distances[1:]
    # print(detours)
    distances = np.array([euclidean_distance(path[i], path[i+1]) for i in range(len(path)-1)])
    #print(distances)
    cost = detours - distances
    insert = np.argmin(cost)
    new_path = np.concatenate((path[:insert+1], [new_point], path[insert+1:]))
    #print(np.argmin(cost))
    #print(cost)
    return np.asarray(new_path)

def GLR_end(control_points, starting_point, geo, time, seed=0, h_cells=2, v_cells=2):
    """Grid Limited Randomness, with every new point added to the end of the existing path."""
    random= np.random.default_rng(seed)
    path_min_length = geo["velocity"] * time
    waypoints = [(starting_point)]
    waypoints += control_points
    while True:
        pathlength = get_path_length(starting_point, waypoints)
        # print(f"pathlength = {pathlength}")
        if pathlength > path_min_length:
            return waypoints
        for i in range(h_cells * v_cells):
            x1,x2,y1,y2 = GetXYRangeRelatedToCellNumber(i+1, h_cells, v_cells,geo["xmin"], geo["xmax"], geo["ymin"], geo["ymax"])
            x= random.uniform(x1, x2)
            y= random.uniform(y1, y2)
            waypoints.append((x,y))

def GLR_shortest_detour(control_points, starting_point, geo, time, seed=0, h_cells=2, v_cells=2):
    """Lotzi's implementation for the grid limited randomness model. The objective is to make a path, sorted the right way, starting with starting_point which as at least as long as the implementation. We have the control points and the go-round several times."""
    random= np.random.default_rng(seed)
    path_min_length = geo["velocity"] * time
    waypoints = [(starting_point)]
    waypoints += control_points
    while True:
        for i in range(h_cells * v_cells):
            x1,x2,y1,y2 = GetXYRangeRelatedToCellNumber(i+1, h_cells, v_cells,geo["xmin"], geo["xmax"], geo["ymin"], geo["ymax"])
            x = random.uniform(x1, x2)
            y = random.uniform(y1, y2)
            # print(waypoints)
            waypoints = add_to_shortest_detour(waypoints, [x,y])
            # print(waypoints)
            # waypoints.append((x,y))
            pathlength = get_path_length(starting_point, waypoints)
            if pathlength > path_min_length:
                return waypoints    
            
def GLR_CristophidesAlgorithm(control_points, starting_point, geo, time, seed=0, h_cells=2, v_cells=2):
    # we are going to use GLR_end as a subroutine, but resort it with Christophides
    path_min_length = geo["velocity"] * time
    random= np.random.default_rng(seed)

    timex = time
    while True:
        path = GLR_end(control_points, starting_point, geo, timex, seed, h_cells, v_cells)
        # print(get_path_length(starting_point, path))
        pathx = copy.deepcopy(path)
        path2= christofidesV2.compute(pathx)
        pathlength = get_path_length(starting_point, path2)
        # print(f"pathlength = {pathlength} - Christophides")
        if pathlength > path_min_length:
            break
        timex = 2 * timex

    # shuffling everything except the starting point 
    # this tries to avoid a problem with some areas having holes 
    # due to the traversal and cell number too large
    path2 = copy.deepcopy(path[1:])
    random.shuffle(path2)
    path = path[0:0] + path2

    # binary search for the path that just fits
    upper = len(path)
    lower = len(path) // 2
    pathbest = path2
    while upper - 1 > lower:
        subpath = copy.deepcopy(path[:int((upper+lower)/2)])
        path2 = christofidesV2.compute(subpath)
        pathlength = get_path_length(starting_point, path2)
        # print(f"pathlength = {pathlength} - binary search")
        # print(f"{upper} {lower}")
        if pathlength > path_min_length:
            upper = int((upper+lower)/2)
            pathbest = path2
        else:
            lower = int((upper+lower)/2)
    return pathbest

def generate_GLR_EOP(exp_policy: Experiment, 
                                    exp_env: Experiment):
    """Example of how to create a generator for a policy type, in this case fixed budget lawnmower FBLM"""
    geo = get_geometry(exp_env["typename"])
    seed = exp_policy["seed"]
    h_cells = exp_policy["h_cells"]
    v_cells = exp_policy["v_cells"]
    path = GLR_end([], [0,0], geo, time = geo["timesteps-per-day"], seed=seed, h_cells=h_cells, v_cells=v_cells)
    print(f"end end-of-path {h_cells}x{v_cells}")
    policy = FollowPathPolicy(vel = geo["velocity"], waypoints = path, repeat = True)
    policy.name = f"GLR-EOP-{h_cells}x{v_cells}"
    return policy

def generate_GLR_SD(exp_policy: Experiment, 
                                    exp_env: Experiment):
    """Example of how to create a generator for a policy type, in this case fixed budget lawnmower FBLM"""
    geo = get_geometry(exp_env["typename"])
    seed = exp_policy["seed"]
    h_cells = exp_policy["h_cells"]
    v_cells = exp_policy["v_cells"]
    path = GLR_shortest_detour([], [0,0], geo, time = geo["timesteps-per-day"], seed=seed, h_cells=h_cells, v_cells=v_cells)
    print(f"end end-of-path {h_cells}x{v_cells}")
    policy = FollowPathPolicy(vel = geo["velocity"], waypoints = path, repeat = True)
    policy.name = f"GLR-SD-{h_cells}x{v_cells}"
    return policy

def generate_GLR_CA(exp_policy: Experiment, 
                                    exp_env: Experiment):
    """Example of how to create a generator for a policy type, in this case fixed budget lawnmower FBLM"""
    geo = get_geometry(exp_env["typename"])
    seed = exp_policy["seed"]
    h_cells = exp_policy["h_cells"]
    v_cells = exp_policy["v_cells"]
    path = GLR_CristophidesAlgorithm([], [0,0], geo, time = geo["timesteps-per-day"], seed=seed, h_cells=h_cells, v_cells=v_cells)
    print(f"end end-of-path {h_cells}x{v_cells}")
    policy = FollowPathPolicy(vel = geo["velocity"], waypoints = path, repeat = True)
    policy.name = f"GLR-SD-{h_cells}x{v_cells}"
    return policy
