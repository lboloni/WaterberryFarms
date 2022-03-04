from Environment import Environment, EpidemicSpreadEnvironment, DissipationModelEnvironment, PrecalculatedEnvironment
from InformationModel import StoredObservationIM, GaussianProcessScalarFieldIM, DiskEstimateScalarFieldIM, im_score, im_score_weighted
from Robot import Robot
from Policy import GoToLocationPolicy, FollowPathPolicy, RandomWaypointPolicy, \
    AbstractWaypointPolicy, generate_lawnmower, InformationGreedyPolicy, generate_lawnmower
from WaterberryFarm import load_precalculated_wbfe, WaterberryFarm, WaterberryFarmInformationModel, waterberry_score, create_precalculated_wbfe
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from matplotlib import animation
import numpy as np
import math
import logging

# logging.basicConfig(level=logging.WARNING)
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)


# this experiment creates a series of robot trajectories, generates the observations taken by the robots, calculates the score and plots it.

def get_observations(positions, env):
    """Given a list of positions and a static environment, generates a list of observations on the environment. 
    Positions format [[x, y]]
    """
    observations = []
    for pos in positions:
        obs = {StoredObservationIM.X: pos[0], StoredObservationIM.Y: pos[1]}
        tylcv = {StoredObservationIM.X: pos[0], StoredObservationIM.Y: pos[1]}
        tylcv[StoredObservationIM.VALUE] = env.tylcv.get(pos[0], pos[1])
        obs["TYLCV"] = tylcv
        ccr = {StoredObservationIM.X: pos[0], StoredObservationIM.Y: pos[1]}
        ccr[StoredObservationIM.VALUE] = env.ccr.get(pos[0], pos[1])
        obs["CCR"] = ccr
        soil = {StoredObservationIM.X: pos[0], StoredObservationIM.Y: pos[1]}
        soil[StoredObservationIM.VALUE] = env.soil.get(pos[0], pos[1])
        obs["Soil"] = soil
        observations.append(obs)
    return observations

def get_observations_in_time(positions, env):
    """Given a list of positions, and the timepoints where those positions were achieved and a dynamic environment (currently at a time earlier than any of those observations, generate the list of observations in the environment)
    Positions format [[x,y,time]]
    """
    observations = []
    for pos in positions:
        time = pos[2]
        if time < env.time:
            env.proceed(1)
        obs = {StoredObservationIM.X: pos[0], StoredObservationIM.Y: pos[1], obs[StoredObservationIM.TIME]: pos[2]}

        tylcv = {StoredObservationIM.X: pos[0], StoredObservationIM.Y: pos[1],obs[StoredObservationIM.TIME]: pos[2]}
        tylcv[StoredObservationIM.VALUE] = env.tylcv.get(pos[0], pos[1])
        obs["TYLCV"] = tylcv

        ccr = {StoredObservationIM.X: pos[0], StoredObservationIM.Y: pos[1], obs[StoredObservationIM.TIME]: pos[2]}
        ccr[StoredObservationIM.VALUE] = env.ccr.get(pos[0], pos[1])
        obs["CCR"] = ccr

        soil = {StoredObservationIM.X: pos[0], StoredObservationIM.Y: pos[1], obs[StoredObservationIM.TIME]: pos[2]}
        soil[StoredObservationIM.VALUE] = env.soil.get(pos[0], pos[1])
        obs["Soil"] = soil

        observations.append(obs)
    return observations


def run_scenario(robot, time, oneshot = True):    
    """Run the scenario. If one shot, only perform one estimate at the end."""
    # environment
    wbf, wbfe = load_precalculated_wbfe()
    positions = []
    for time in range(time):
        robot.enact_policy()
        robot.proceed(1)
        position = [int(robot.x), int(robot.y), time]
        print(robot)
        positions.append(position)
    if oneshot:
        wbfe.proceed(time)
    else:
        for time in range(time):
            # in this case, advance to the end of the environmemt
            wbfe.proceed(1)
    observations = get_observations(positions, wbfe)
    wbfim = WaterberryFarmInformationModel("wbfi", wbf.width, wbf.height)
    scores = []
    for i, obs in enumerate(observations):
        wbfim.add_observation(obs)
        if not oneshot or i == len(observations)-1:
            wbfim.proceed(1)
            logging.info(f"Calculating score for observation {i}")
            score = waterberry_score(wbfe, wbfim)
            scores.append(score)
    return score, scores, wbfe, wbfim, observations

if __name__ == "__main__":
    if False: 
        create_precalculated_wbfe(200)
    if True:
        robot = Robot("Robi", 0, 0, 0, env=None, im=None)
        path = generate_lawnmower(0, 6000, 0, 5000, 10)
        robot.policy = FollowPathPolicy(None, robot, vel = 100, waypoints = path, repeat = True)
        score, scores, wbfe, wbfim, observations = run_scenario(robot, 1400, oneshot = True)        
        print(f"Score {score}")
        print(f"Scores {scores}")
        ## visualize the env, im and observation list for tylcv
        fig, ((ax_env_tylcv, ax_im_tylcv, ax_obs)) = plt.subplots(3, figsize=(5,15))
        image_env_tylcv = ax_env_tylcv.imshow(wbfe.tylcv.value.T, vmin=0, vmax=1, origin="lower")
        image_im_tylcv = ax_im_tylcv.imshow(wbfim.im_tylcv.value.T, vmin=0, vmax=1, origin="lower")
        obsx = []
        obsy = []
        for obs in observations:
            obsx.append(obs[StoredObservationIM.X])
            obsy.append(obs[StoredObservationIM.Y])
        ax_obs.plot(obsx, obsy)
        ax_obs.set_xlim(xmin = 0, xmax = wbfe.width)
        ax_obs.set_ylim(ymin = 0, ymax = wbfe.height)
        plt.show()

    if False:
        # trying to find some numpy wizardry for disk estimate
        dimx = 20
        dimy = 20
        value = np.zeros([dimx, dimy])
        radius = 3
        # create a mask array
        maskdim = 2 * radius + 1
        mask = np.full((maskdim, maskdim), False, dtype=bool)
        for i in range(-radius, radius):
            for j in range(-radius, radius):
                if (math.sqrt((i*i + j*j)) <= radius):
                    mask[i + radius, j + radius] = True
        print(mask)
        centerx = 20
        centery = 20
        locx = centerx - radius
        locy = centery - radius
        mask2 = np.full((dimx, dimy), False, dtype=bool)

        maskp = mask[max(0, -locx):min(maskdim, dimx-locx), max(0, -locy):min(maskdim,dimy-locy)]
        print(f"maskp = {maskp}")

        mask2[max(0, locx):min(dimx, locx+maskdim), max(0, locy):min(dimy,locy+maskdim)] = maskp
        print(mask2)
        value[mask2==1] = 3
        print(value)

