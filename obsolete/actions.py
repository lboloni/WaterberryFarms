"""
actions.py

2025-03-22 Obsolete functions. In the first iterations of this code, it was designed to be run through the command line. With the use of the notebooks and the exp/run framework, these are obsolete. Keeping them here, because they might implement flows that are going to be useful later.

"""
def action_precompute_environment(choices):
    results = copy.deepcopy(choices)
    results["action"] = "precompute-environment"
    menuGeometry(results)
    wbf, wbfe, savedir = create_wbfe(False, wbf_prec = None, typename = results["typename"])
    results["wbf"] = wbf
    results["wbfe"] = wbfe
    results["savedir"] = savedir
    if "precompute-time" in results:
        for i in range(results["precompute-time"]):
            logging.info(f"precalculation proceed {i}")
            wbfe.proceed()
    return results 

def action_load_environment(choices):
    results = copy.deepcopy(choices)
    results["action"] = "load-environment"
    menuGeometry(results)
    wbf, wbfe, savedir = create_wbfe(True, wbf_prec = None, typename = results["typename"])
    results["wbf"] = wbf
    results["wbfe"] = wbfe
    results["savedir"] = savedir
    if "precompute-time" in results:
        for i in range(results["precompute-time"]):
            logging.info(f"precalculation proceed {i}")
            wbfe.proceed()
    return results 

def action_visualize(choices):
    results = copy.deepcopy(choices)
    results["action"] = "visualize"
    menuGeometry(results)
    wbf, wbfe, savedir = create_wbfe(True, wbf_prec = None, typename = results["typename"])
    results["wbf"] = wbf
    results["wbfe"] = wbfe
    results["savedir"] = savedir
    timepoint = int(input("Timepoint for visualization:"))
    wbfe.proceed(timepoint)
    wbfe.visualize()
    plt.show()
    return results

def action_animate(choices):
    results = copy.deepcopy(choices)
    results["action"] = "visualize"
    menuGeometry(results)
    wbf, wbfe, savedir = create_wbfe(True, wbf_prec = None, typename = results["typename"])
    wbfe.visualize()
    anim = wbfe.animate_environment()
    plt.show()
    return results


def action_run_1day(choices):
    """
    Implements a single-day experiment with a single robot in the WBF simulator. This is the top level function which is called to run the experiment. 

    choices: a dictionary into which the parameters of the experiments are being loaded. A copy of this will be internally created. 

    results: the return value, which contains both all the input values in the choices, as well as the output data (or references to it.)

    """
    results = copy.deepcopy(choices)
    results["action"] = "run-one-day"
    if "im_resolution" not in results:
        results["im_resolution"] = 1
    menuGeometry(results)
    if "time-start-environment" not in results:
        results["time-start-environment"] = int(input("Time in the environment when the robot starts: "))
    wbf, wbfe, savedir = create_wbfe(saved=True, wbf_prec=None, typename = results["typename"])
    wbfe.proceed(results["time-start-environment"])
    results["wbf"] = wbf
    results["wbfe"] = wbfe
    results["savedir"] = savedir
    results["exp-name"] = results["typename"] + "_" # name of the exp., also dir to save data
    results["days"] = 1
    results["exp-name"] = results["exp-name"] + "1M_"
    get_geometry(results["typename"], results)
    # override the velocity and the timesteps per day, if specified
    if "timesteps-per-day-override" in results:
        results["timesteps-per-day"] = results["timesteps-per-day-override"]
    if "velocity-override" in results:
        results["velocity"] = results["velocity-override"]
    results["robot"] = Robot("Rob", 0, 0, 0, env=None, im=None)
    # Setting the policy
    results["exp-name"] = results["exp-name"] + results["policy-code"].name
    results["robot"].assign_policy(results["policy-code"])

    if "results-filename" in results:
        results_filename = results["results-filename"]
    else:
        results_filename = f"res_{results['exp-name']}"
    if "results-basedir" in results:
        results_path = pathlib.Path(results["results-basedir"], results_filename)
    else:
        results_path = pathlib.Path(results["savedir"], results_filename)    
    results["results-path"] = results_path
    # if dryrun is specified, we return the results without running anything
    if "dryrun" in results and results["dryrun"] == True:
        return results
    # results["oneshot"] = False # calculate one observation score for all obs.
    # running the simulation
    simulate_1day(results)
    #
    logging.info(f"Saving results to: {results_path}")
    with compress.open(results_path, "wb") as f:
        pickle.dump(results, f)
    return results


def action_run_multiday(choices):
    """
    Implements a multi-day (typically 15) experiment with a single robot in the WBF simulator. This is the top level function which is called to run the experiment. 

    choices: a dictionary into which the parameters of the experiments are being loaded. A copy of this will be internally created. 

    results: the return value, which contains both all the input values in the choices, as well as the output data (or references to it.)

    """
    results = copy.deepcopy(choices)
    results["action"] = "run-15-day"
    menuGeometry(results)
    if "time-start-environment" not in results:
        results["time-start-environment"] = int(input("Time in the environment when the robot starts: "))
    wbf, wbfe, savedir = create_wbfe(saved=True, wbf_prec=None, typename = results["typename"])
    wbfe.proceed(results["time-start-environment"])
    results["wbf"] = wbf

    # wbfe will be the environment at the end of experiment
    results["wbfe"] = wbfe
    results["wbfe-days"] = {}
    results["wbfe-days"][0] = copy.deepcopy(wbfe)

    results["savedir"] = savedir
    results["exp-name"] = results["typename"] + "_" # name of the exp., also dir to save data
    if "days" not in results:
        results["days"] = 15
    results["exp-name"] = results["exp-name"] + "1M_"
    get_geometry(results["typename"], results)
    # override the velocity and the timesteps per day, if specified
    if "timesteps-per-day-override" in results:
        results["timesteps-per-day"] = results["timesteps-per-day-override"]
    if "velocity-override" in results:
        results["velocity"] = results["velocity-override"]        
    results["robot"] = Robot("Rob", 0, 0, 0, env=None, im=None)
    # Setting the policy
    results["exp-name"] = results["exp-name"] + results["policy-code"].name
    results["robot"].assign_policy(results["policy-code"])

    if "results-filename" in results:
        results_filename = results["results-filename"]
    else:
        results_filename = f"res_{results['exp-name']}"
    if "results-basedir" in results:
        results_path = pathlib.Path(results["results-basedir"], results_filename)
    else:
        results_path = pathlib.Path(results["savedir"], results_filename)    
    results["results-path"] = results_path
    # if dryrun is specified, we return the results without running anything
    if "dryrun" in results and results["dryrun"] == True:
        return results
    results["oneshot"] = False # calculate one observation score for all obs.
    # running the simulation
    simulate_multiday(results)
    #
    logging.info(f"Saving results to: {results_path}")
    with compress.open(results_path, "wb") as f:
        pickle.dump(results, f)
    return results

