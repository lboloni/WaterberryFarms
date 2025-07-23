"""
exp_run_config.py

Implements the Configuration / Experiment / Run / Subrun framework. 

This is a configuration system for software projects that allows the running of projects that need to run on multiple, differently configurated machines, on different platforms. 

It is specifically focusing on cases where multiple different kinds of experiments have to be run and analyzed, under various configurations and assumptions. 

FIXME: clean up the below text

These can be considered as constants for a particular run, but they might have different values on different computers or setups.

They should be accessed through Config()["value"]

A configuration system that allows porting between different machines and the use of multiple configuration files. 

It starts by loading a file from ~/.config/<<ProjectName>>>/mainsettings.yaml" 
from where the "configpath" property points to the path of the 
actual configuration file. 

Template configuration files and mainsettings-sample.yaml are in top directory of the project. However, the actual configuration files should be on the local directory outside the github package, as these configuration files contain local information such as user directory names etc.

The configuration values are a hierarchical dictionary which can be 
accessed  should be accessed through Config()["value"] or 
Config()["group"]["value"]


"""

import yaml
import pathlib
import shutil
import textwrap
from datetime import datetime
from pathlib import Path


class Experiment:
    """A class encapsulating an experiment. It is a dict-like interface, with additional functionality for mandatory fields like data_dir, and support for saving to the data_dir and timers."""
    
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value
        self.save()

    def __contains__(self, key):
        return key in self.values

    def get(self, key, default=None):
        return self.values.get(key, default)

    def __repr__(self):
        text = yaml.dump(self.values)
        text = textwrap.indent(text, prefix="    ")
        text = "Experiment:" + "\n" + text
        return text
    
    def data_dir(self):
        return pathlib.Path(self.values[Config.DATA_DIR])

    def experiment(self):
        return self.values[Config.EXPERIMENT_NAME]

    def run(self):
        return self.values[Config.RUN_NAME]

    def subrun(self):
        return self.values[Config.SUBRUN_NAME]

    def clean(self):
        """Programatically clean up an experiment to allow new run"""
        d = self.data_dir()
        if not d.exists():
            print(f"The experiment directory\n\t{d}\n\tdoes not exist.")
            return
        response = input(f"Cleaning the experiment will remove the directory: \n{d}\n Confirm (y/n): ").strip().lower()
        if response == 'y':
            d = self.data_dir()
            shutil.rmtree(d)
            d.mkdir(exist_ok=False)
        elif response == 'n':
            print(f"You chose NOT to delete the experiment directory\n\t{d}")
            return

    def save(self):
        with open(pathlib.Path(self.data_dir(), "exprun.yaml"), "w") as f:
            yaml.dump(self.values, f)

    def set_time_started(self):
        now = datetime.now()
        self.values[Config.TIME_STARTED] = now.strftime(Config.TIME_FORMAT)

    def get_time_started(self):
        parsed = datetime.strptime(self.values[Config.TIME_STARTED], Config.TIME_FORMAT)
        return parsed

    def start_timer(self, timer_name="default", verbose=True):
        now = datetime.now()
        self.values["timer-" + timer_name + "-start"] = now.strftime(Config.TIME_FORMAT)
        if verbose:
            print(f"***Timer*** {timer_name} started")
        self.save()

    def end_timer(self, timer_name="default", verbose=True):
        now = datetime.now()
        self.values["timer-" + timer_name + "-end"] = now.strftime(Config.TIME_FORMAT)        
        start = datetime.strptime(self.values["timer-" + timer_name + "-start"], Config.TIME_FORMAT)
        seconds = (now - start).total_seconds()
        self.values["timer-" + timer_name + "-seconds"] = seconds   
        if verbose:
            print(f"***Timer*** {timer_name} finished in {seconds} seconds")
        self.save()


    def done(self):
        """Sets the time done variable, and saves"""
        now = datetime.now()
        self.values[Config.TIME_DONE] = now.strftime(Config.TIME_FORMAT)
        self.save()

class Config:
    """The overall settings class. """
    _instance = None  # Class-level attribute to store the instance

    PREFIX = "***ExpRun**: "
    PROJECTNAME = None
    DATA_DIR = "data_dir"
    EXPERIMENT_NAME = "experiment_name"
    RUN_NAME = "run_name"
    SUBRUN_NAME = "subrun_name"
    TIME_STARTED = "time_started"
    TIME_DONE = "time_done"
    TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"

    def __new__(cls, *args, **kwargs):
        if Config.PROJECTNAME is None:
            Config.__log(f"Config.PROJECTNAME not set, assign it before using the exp/run framework")
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls)
            home_directory = pathlib.Path.home()
            main_config = pathlib.Path(home_directory, ".config", Config.PROJECTNAME, "mainsettings.yaml")
            Config.__log(f"Loading pointer config file:\n\t{main_config}")
            if not main_config.exists():
                raise Exception(f"Missing pointer config file: {main_config}")
            with main_config.open("rt") as handle:
                main_config = yaml.safe_load(handle)
            configpath = main_config["configpath"]
            Config.__log(f"Loading machine-specific config file:\n\t{configpath}")
            configpath = pathlib.Path(configpath).expanduser().resolve()
            if not configpath.exists():
                raise Exception(f"Missing machine-specific config file:\n\t {configpath}")
            with configpath.open("rt") as handle:
                cls._instance.values = yaml.safe_load(handle)
            cls._instance.values["configpath"] = configpath
            cls._instance.values["main_config"] = main_config
            # experiment path 
            current_directory = pathlib.Path(__file__).resolve().parent    
            cls._instance.experiment_path_internal = pathlib.Path(current_directory, "experiment_configs")
            cls._instance.experiment_path = cls._instance.experiment_path_internal

        return cls._instance

    def __getitem__(self, key):
        return self.values[key]
    
    @staticmethod
    def __log(pattern):
        """Print exprun messages in a distinguishable form"""
        print(Config.PREFIX, end="")
        print(pattern, flush=True)

    def list_experiments(self):
        """List the experiment directories as strings that are present 
        in the current locations."""
        subdirs = [p.name for p in self.experiment_path.iterdir() if p.is_dir()]
        return subdirs

    def set_experiment_path(self, path: pathlib.Path):
        """Sets the experiment config directories to be an external directory"""
        assert path.exists()
        self.experiment_path = path
        self.__log(f"Experiment config path changed to {self.experiment_path}")
    
    def get_experiment_path(self):
        return self.experiment_path

    def set_experiment_data(self, path: pathlib.Path):
        """Sets the experiment data to be an external directory"""
        assert path.exists()
        self.values["experiment_data"] = path
        self.__log(f"Experiment data path changed to {self.values['experiment_data']}")

    def copy_experiment(self, exp_name, run_name = None):
        """Copy an experiment run, or all the runs, from an internal
        experiment config to the external on
        FIXME Use list runs
        """
        # we cannot do this in the internal domain
        assert self.experiment_path != self.experiment_path_internal
        if run_name:
            source_path = pathlib.Path(self.experiment_path_internal, exp_name, run_name + ".yaml")
            assert source_path.exists()
            target_path = pathlib.Path(self.experiment_path, exp_name, run_name + ".yaml")
            shutil.copy2(source_path, target_path)
            self.__log(f"Exp/run {exp_name}/{run_name} copied to {target_path}")
        else: # copy the full directory
            source_path = pathlib.Path(self.experiment_path_internal, exp_name)
            assert source_path.exists()
            target_path = pathlib.Path(self.experiment_path, exp_name)
            shutil.copytree(source_path, target_path,  dirs_exist_ok=True)
            self.__log(f"Experiment {exp_name} copied to {target_path}")

    def create_exprun_variant(self, exp_name, run_name, changes = {}, new_run_name=None):
        """Creates a variation of the experiment"""
        # we cannot do this in the internal domain
        assert self.experiment_path != self.experiment_path_internal
        exp = self.get_experiment(exp_name, run_name)
        for key in changes:
            exp.values[key] = changes[key]
        if not new_run_name:
            now = datetime.now()
            new_run_name =  run_name + "_" + now.strftime("%Y-%m-%d-%H-%M-%S")
        target_path = pathlib.Path(self.experiment_path, exp_name, new_run_name + ".yaml")
        with open(target_path, "w") as f:
            yaml.dump(exp.values, f)
        self.__log(f"Exp/run variant {exp_name}/{new_run_name} created in {target_path}")


    def list_runs(self, exp_name, done_only=False):
        """List the runs present in the experiment. By default, this is listing the experiment files, not the executed directories. 
        FIXME: add functionality to list the done ones. 
        """
        if not done_only:
            current_directory = pathlib.Path(__file__).resolve().parent    
            runs = pathlib.Path(self.experiment_path, exp_name)
            if not runs.exists():
                raise Exception("No experiment named {exp_name} in the current context")
            runs = [p.stem for p in runs.iterdir() if p.is_file() and p.suffix == ".yaml"]
            return runs
        # if we are specifying done_only, we are listing them from the 
        # data directory
        data_dir = pathlib.Path(self.values["experiment_data"], exp_name).expanduser()
        subdirs = [p.name for p in data_dir.iterdir() if p.is_dir()]
        return subdirs

    def list_subruns(self, exp_name, run_name):
        """List the done subruns in the experiment. """
        data_dir = pathlib.Path(self.values["experiment_data"], exp_name, run_name).expanduser()
        if not data_dir.exists():
            return []
        subdirs = [p.name for p in data_dir.iterdir() if p.is_dir()]
        return subdirs

    def get_experiment(self, experiment_name, run_name, subrun_name=None, creation_style="exist-ok"):
        """Returns an experiment configuration, which is the 
        mixture between the system-dependent configuration and the system independent configuration.
        
        creation_style can be 
            "exist-ok" - reuse the cached values
            "discard-old" - discard the old values, start from scratch
            "version" - create a new version
        
        """
        current_directory = pathlib.Path(__file__).resolve().parent
        #
        # Load the system independent defaults configuration
        #
        experiment_group_indep = pathlib.Path(self.experiment_path, experiment_name, "_defaults_" + experiment_name + ".yaml")
        if not experiment_group_indep.exists():
            raise Exception(f"Missing experiment default config {experiment_group_indep}")
        with experiment_group_indep.open("rt") as handle:
            group_config = yaml.safe_load(handle)
        if group_config == None:
            self.__log(f"Experiment default config {experiment_group_indep} was empty, ok.")
            group_config = {}
        group_config[Config.EXPERIMENT_NAME] = experiment_name
        #
        # Load the system independent run configuration
        #
        experiment_run_indep = pathlib.Path(self.experiment_path, experiment_name, run_name + ".yaml")
        if not experiment_run_indep.exists():
            raise Exception(f"Missing experiment file\n {experiment_run_indep}")
        with experiment_run_indep.open("rt") as handle:
            indep_config = yaml.safe_load(handle)
        if indep_config == None:
            self.__log(f"The system independent config file of the experiment\n\t{experiment_run_indep}\n\t was empty, this is a likely error, but continuing.")
            indep_config = {}

        indep_config[Config.RUN_NAME] = run_name
        indep_config = group_config | indep_config
        # create the data dir
        if "experiment_data" not in self.values:            
            self.__log(f"The key 'experiment_data' must be present in the configuration.\n\tThis is the parent directory from where the experiment data should go.")
            self.__log(f"If all the data for all the experiments go to the same directory, \n\tan appropriate place for this is the system dependent config file, in this case:\n\t {self['configpath']}")
            raise Exception("Missing experiment_data specification")
        indep_config["exp_run_sys_indep_file"] = str(experiment_run_indep)
        #
        # Load the system dependent run configuration
        #
        if "experiment_system_dependent_dir" not in self.values:            
            self.__log(f"The key 'experiment_system_dependent_dir' must be present in the configuration.\n\tThis is the parent directory from where system dependent part of the experiment configurations must go.")
            self.__log(f"An appropriate place to specify this is the system dependent config file, in this case:\n\t{self['configpath']}")
            raise Exception("Missing experiment_system_dependent_dir specification")

        experiment_directory = pathlib.Path(self.values["experiment_system_dependent_dir"])
        experiment_sys_dep = pathlib.Path(experiment_directory, experiment_name, run_name + "_sysdep.yaml")
        if not experiment_sys_dep.exists():
            # don't log this, too common
            # self.__log(f"No system dependent experiment file\n\t {experiment_sys_dep},\n\t that is ok, proceeding.")
            exp_config = indep_config
        else: 
            with experiment_sys_dep.open("rt") as handle:
                dep_config = yaml.safe_load(handle)
            exp_config = indep_config | dep_config
            exp_config["exp_run_sys_dep_file"] = str(experiment_sys_dep)

        self.__log(f"Configuration for exp/run: {experiment_name}/{run_name} successfully loaded")

        # creating the data dir
        if subrun_name is None:
            data_dir = pathlib.Path(self.values["experiment_data"], experiment_name, run_name).expanduser()
        else:
            data_dir = pathlib.Path(self.values["experiment_data"], experiment_name, run_name, subrun_name).expanduser()
        exp_config[Config.DATA_DIR] = str(data_dir)
        exp_config[Config.SUBRUN_NAME] = subrun_name

        if creation_style == "exist-ok":
            # it is ok if the directory exists, but then we don't measure time 
            # or save the values
            exp = Experiment(exp_config)
            if not data_dir.exists():
                data_dir.mkdir(exist_ok=True, parents=True)
                exp.set_time_started()
                exp.save()
        elif creation_style == "version":
            # if the directory exists, move it to a backup
            if data_dir.exists():                
                now = datetime.now()
                formatted = now.strftime("%Y-%m-%d-%H-%M-%S")
                print(formatted)  # Example output: 2025-05-25-17-11            
                backup_dir = data_dir.parent / f"{data_dir.name}_{formatted}"
                self.__log(f"Moving existing experiment directory to {backup_dir}")
                data_dir.rename(backup_dir)
                data_dir.mkdir(exist_ok=True, parents=True)
                exp = Experiment(exp_config)
                exp.set_time_started()
                exp.save()
        elif creation_style == "discard-old":
            # if the directory exists, remove it
            if data_dir.exists():
                #if input(f"Experiment directory {data_dir} already exists. Do you want to remove it? (y/n): ").strip().lower() != 'y':
                #    raise Exception(f"Experiment directory {data_dir} already exists, and you chose not to remove it.")
                self.__log(f"Removing existing experiment directory {data_dir}")
                shutil.rmtree(data_dir)
            data_dir.mkdir(exist_ok=True, parents=True)
            exp = Experiment(exp_config)
            exp.set_time_started()
            exp.save()    
        return exp
