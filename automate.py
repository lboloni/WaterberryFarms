from exp_run_config import Config, Experiment
Config.PROJECTNAME = "WaterBerryFarms"

import sys
import yaml
import pathlib
import papermill as pm
from tqdm import tqdm

def automate_exprun(notebook, name, params):
   """Automates the execution of a notebook. It is assumed that in the notebook there is cell tagged "parameters", and in general that the notebook is idempotent."""

   ext_path = pathlib.Path(Config()["experiment_external"])
   params["external_path"] = str(pathlib.Path(ext_path, "_automate").resolve())
   notebook_path = pathlib.Path(notebook)
   output_path = pathlib.Path(ext_path, "_automation_output")
   output_filename = f"{notebook_path.stem}_{name}_output{ notebook_path.suffix}"
   output = pathlib.Path(output_path, notebook_path.parent, output_filename)
   output.parent.mkdir(exist_ok=True, parents=True)
   print(output)

   try:
      pm.execute_notebook(
         notebook,
         output.absolute(),
         cwd=notebook_path.parent,
         parameters=params
      )
   except Exception as e:
      print(f"There was an exception {e}")

if __name__ == "__main__":
   if len(sys.argv) < 2:
      print("No argument passed, running automate_00")
      experiment = "automate"
      run = "automate_short"
      exp = Config().get_experiment(experiment, run)
   else:
      print(f"Running script {sys.argv[1]}")
      yaml_path = pathlib.Path(sys.argv[1])
      if not yaml_path.is_file():
         print(f"Error: File '{yaml_path}' does not exist.")
         sys.exit(1)
      # Load the YAML file into a dictionary
      with open(yaml_path, 'r') as f:
         exp = yaml.safe_load(f)      

   for item in tqdm(exp["exps_to_run"]):
      print(f"***Automating {item['name']}")
      #notebook = params["notebook"]
      automate_exprun(item["notebook"], item["name"], item["params"])