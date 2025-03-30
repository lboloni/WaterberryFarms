# Multi-day simulate: as there was no current project doing this 
# when the refactoring happened in March 2025, this was not 
# refactored.
# Might be brought back in later.

def simulate_multiday(results):
    """
    Runs the simulation for multiple days. All the parameters and output 
    are in the results dictionary
    """
    wbfe, wbf = results["wbfe"], results["wbf"]
    wbfim = results["estimator-code"]
    results["wbfim-days"] = {}
    results["wbfim-days"][0] = copy.deepcopy(wbfim)
    # keep the positions and remember the length to split by days
    positions = []
    positions_size_last = len(positions)
    results["positions-days"] = {}
    # keep the observations and remember the lenght to split by days    
    observations = []
    observations_size_last = len(observations)
    results["observations-days"] = {}    
    # keep the scores and remember the lenght to split by days    
    scores = []
    scores_size_last = len(scores)
    results["scores-days"] = {}
    for day in range(1,results["days"]): # 1,2,...15 day zero is the beginning
        print(f"day {day}")
        elapsedCount = 0 # how many steps elapsed since we updated im
        for time in range(int(results["timesteps-per-day"])):
            results["robot"].enact_policy()
            results["robot"].proceed(1)
            position = [int(results["robot"].x), int(results["robot"].y), time]
            positions.append(position)
            obs = wbfe.get_observation(position)
            observations.append(obs)
            wbfim.add_observation(obs)
            # determine if we perform the calculation and for how many days
            # FIXME: debug this
            performCount = 0            
            if time == 0: # first time
                performCount = 1
            elif elapsedCount % int(results["score-resolution"]) == 0:
                performCount = elapsedCount
                elapsedCount = 0
            elif time == int(results["timesteps-per-day"])-1:
                performCount = elapsedCount
                elapsedCount = 0
            ## if performcount is not zero update it.
            if performCount > 0:
                wbfim.proceed(performCount)
                score = results["score-code"].score(wbfe, wbfim)
                for i in range(performCount):
                    scores.append(score)
            elapsedCount += 1

        # store the values at the end of the day
        results["wbfe-days"][day] = copy.deepcopy(wbfe)
        results["wbfim-days"][day] = copy.deepcopy(wbfim)
        results["positions-days"][day] = positions[positions_size_last:]
        positions_size_last = len(positions)
        results["observations-days"][day] = observations[observations_size_last:]
        observations_size_last = len(observations)
        results["scores-days"][day] = scores[scores_size_last:]
        scores_size_last = len(scores)
        # move to the next day in the environment
        wbfe.proceed(1)
    # store in results the values at the end of scenario
    results["score"] = score
    results["scores"] = scores
    results["observations"] = observations
    results["positions"] = positions
