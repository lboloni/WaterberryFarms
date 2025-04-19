"""
This file contains settings for the WBF project. These can be considered as constants for a particular run, but they might have different values on different computers or setups.

They should be accessed through Config().values["value"]

A configuration system that allows porting between different machines and the use of multiple configuration files. 

It starts by loading a file from ~/.config/WaterBerryFarms/mainsettings.yaml" 
from where the "configpath" property points to the path of the 
actual configuration file. 

Template configuration files and mainsettings-sample.yaml are in top directory of the project. However, the actual configuration files should be on the local directory outside the github package, as these configuration files contain local information such as user directory names etc.

The configuration values are a hierarchical dictionary which can be 
accessed  should be accessed through Config()["value"] or 
Config()["group"]["value"]


"""

import yaml
from pathlib import Path

# PROJECTNAME = "WaterBerryFarms"


class Experiment:
    """A class encapsulating an experiment"""
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        return self.values[key]
    
    def __repr__(self):
        """FIXME: print the experiment out in a readable way"""
        return "Experiment: {self.values}"
    
    def clean(self):
        """Programatically clean up an experiment to allow new run"""
        raise Exception("Experiment.clean not implemented yet")

class Config:
    """The overall settings class. """
    _instance = None  # Class-level attribute to store the instance

    PREFIX = "** ExpRun ** : "
    PROJECTNAME = None

    def __new__(cls, *args, **kwargs):
        if Config.PROJECTNAME is None:
            Config.__log(f"Config.PROJECTNAME not set, assign it before using the exp/run framework")
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls)
            home_directory = Path.home()
            main_config = Path(home_directory, ".config", Config.PROJECTNAME, "mainsettings.yaml")
            Config.__log(f"Loading pointer config file:\n\t {main_config}")
            if not main_config.exists():
                raise Exception(f"Missing pointer config file: {main_config}")
            with main_config.open("rt") as handle:
                main_config = yaml.safe_load(handle)
            configpath = main_config["configpath"]
            Config.__log(f"Loading machine-specific config file: {configpath}")
            configpath = Path(configpath)
            if not configpath.exists():
                raise Exception(f"Missing machine-specific config file:\n\t {configpath}")
            with configpath.open("rt") as handle:
                cls._instance.values = yaml.safe_load(handle)
            cls._instance.values["configpath"] = configpath
            cls._instance.values["main_config"] = main_config
        return cls._instance

    def __getitem__(self, key):
        return self.values[key]
    
    @staticmethod
    def __log(pattern):
        """Print exprun messages in a distinguishable form"""
        print(Config.PREFIX, end="")
        print(pattern, flush=True)

    def get_experiment(self, group_name, run_name, subrun_name=None):
        """Returns an experiment configuration, which is the 
        mixture between the system-dependent configuration and the system independent configuration."""
        current_directory = Path(__file__).resolve().parent
        #
        # Load the system independent group configuration
        #
        experiment_group_indep = Path(current_directory, "experiment_configs", group_name, "_" + group_name + ".yaml")
        if not experiment_group_indep.exists():
            raise Exception(f"Missing experiment default config {experiment_group_indep}")
        with experiment_group_indep.open("rt") as handle:
            group_config = yaml.safe_load(handle)
        if group_config == None:
            print(f"Experiment default config {experiment_group_indep} was empty, ok.")
            group_config = {}
        group_config["group_name"] = group_name
        #
        # Load the system independent run configuration
        #
        experiment_run_indep = Path(current_directory, "experiment_configs", group_name, run_name + ".yaml")
        if not experiment_run_indep.exists():
            raise Exception(f"Missing experiment system independent config file {experiment_run_indep}")
        with experiment_run_indep.open("rt") as handle:
            indep_config = yaml.safe_load(handle)
        if indep_config == None:
            print("The system independent config file of the experiment {experiment_run_indep} was empty, this is a likely error, but continuing.")
            indep_config = {}

        indep_config["run_name"] = run_name
        indep_config = group_config | indep_config
        # create the data dir
        if "experiment_data" not in self.values:            
            print(f"The key 'experiment_data' must be present in the configuration. \nThis is the parent directory from where the experiment data should go.")
            print(f"If all the data for all the experiments go to the same directory, \nan appropriate place for this is the system dependent config file, \n in this case: {self['configpath']}")
            raise Exception("Missing experiment_data specification")
        data_dir = Path(self.values["experiment_data"], group_name, run_name)
        data_dir.mkdir(exist_ok=True, parents=True)
        indep_config["data_dir"] = data_dir
        indep_config["exp_run_sys_indep_file"] = experiment_run_indep
        #
        # Load the system dependent run configuration
        #
        if "experiment_system_dependent_dir" not in self.values:            
            print(f"The key 'experiment_system_dependent_dir' must be present in the configuration. \nThis is the parent directory from where system dependent part of the experiment configurations must go.")
            print(f"An appropriate place to specify this is the system dependent config file, \n in this case: {self['configpath']}")
            raise Exception("Missing experiment_system_dependent_dir specification")

        experiment_directory = Path(self.values["experiment_system_dependent_dir"])
        experiment_sys_dep = Path(experiment_directory, group_name, run_name + "_sysdep.yaml")
        if not experiment_sys_dep.exists():
            print(f"No system dependent experiment file\n {experiment_sys_dep},\n that is ok, proceeding.")
            exp_config = indep_config
        else: 
            with experiment_sys_dep.open("rt") as handle:
                dep_config = yaml.safe_load(handle)
            exp_config = indep_config | dep_config
            exp_config["exp_run_sys_dep_file"] = experiment_sys_dep

        print(f"Configuration for experiment: {group_name}/{run_name} successfully loaded", flush=True)

        return Experiment(exp_config)
