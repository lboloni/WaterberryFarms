import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wbf_flow import build_flow_entries, run_notebook, setup_flow
from wbf_helper import create_wbfe


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiment_configs"
FLOW_NOTEBOOKS = [
    REPOSITORY_ROOT / "notebooks" / name
    for name in (
        "Environment-Precalc.ipynb",
        "1Robot1Day-Run.ipynb",
        "1Robot1Day-Visualize.ipynb",
        "1Robot1Day-Compare.ipynb",
        "nRobot1Day-Run.ipynb",
        "nRobot1Day-Visualize.ipynb",
        "nRobot1Day-Compare.ipynb",
        "Flow-1Robot1Day.ipynb",
        "Flow-nRobot1Day.ipynb",
    )
]


class FlowConfig:
    def __init__(self, experiment_path, flows_path, results_path):
        self.experiment_path = experiment_path
        self.values = {
            "flows_path": flows_path,
            "experiment_data": results_path,
        }

    def __getitem__(self, key):
        return self.values[key]

    def get_experiment_path(self):
        return self.experiment_path

    def set_experiment_path(self, path):
        self.experiment_path = path

    def set_experiment_data(self, path):
        self.values["experiment_data"] = path


class TestFlowMetadata(unittest.TestCase):
    def test_every_resolved_exprun_has_existing_notebook_entries(self):
        for family_path in sorted(path for path in EXPERIMENT_ROOT.iterdir()
                                  if path.is_dir()):
            family = family_path.name
            defaults_path = family_path / f"_defaults_{family}.yaml"
            if not list(family_path.glob("*.yaml")):
                continue
            self.assertTrue(defaults_path.is_file(), family)
            with defaults_path.open() as handle:
                defaults = yaml.safe_load(handle) or {}
            for run_path in sorted(family_path.glob("*.yaml")):
                if run_path == defaults_path:
                    continue
                with run_path.open() as handle:
                    values = defaults | (yaml.safe_load(handle) or {})
                self.assertIsInstance(
                    values["input-to-notebook"], list,
                    f"{family}/{run_path.stem}")
                for notebook in values["input-to-notebook"]:
                    self.assertTrue(
                        (REPOSITORY_ROOT / notebook).is_file(),
                        f"{family}/{run_path.stem}: {notebook}")

    def test_flow_notebooks_have_no_saved_execution_state(self):
        for path in FLOW_NOTEBOOKS:
            with path.open() as handle:
                notebook = json.load(handle)
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertIsNone(cell["execution_count"], path.name)
                    self.assertEqual(cell["outputs"], [], path.name)

    def test_stage_notebooks_have_standard_parameters(self):
        required = {
            "experiment", "run", "creation_style",
            "expruns_path", "results_path",
        }
        for path in FLOW_NOTEBOOKS[:7]:
            with path.open() as handle:
                notebook = json.load(handle)
            parameter_cells = [
                cell for cell in notebook["cells"]
                if "parameters" in cell["metadata"].get("tags", [])
            ]
            self.assertEqual(len(parameter_cells), 1, path.name)
            source = "".join(parameter_cells[0]["source"])
            for name in required:
                self.assertIn(f"{name} =", source, f"{path.name}: {name}")


class TestFlowHelpers(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary_directory.name)
        self.source = self.root / "source"
        self.flows = self.root / "flows"
        family = self.source / "sample"
        family.mkdir(parents=True)
        (family / "_defaults_sample.yaml").write_text(
            "input-to-notebook: []\n")
        (family / "run.yaml").write_text("name: sample\n")
        self.config = FlowConfig(self.source, self.flows, self.root / "old")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_setup_flow_copies_from_active_experiment_path(self):
        expruns, results, notebooks = setup_flow(
            "sample-flow", ["sample"], config=self.config)
        self.assertTrue((expruns / "sample" / "run.yaml").is_file())
        self.assertEqual(self.config.experiment_path, expruns)
        self.assertEqual(self.config.values["experiment_data"], results)
        self.assertTrue(notebooks.is_dir())

    def test_run_notebook_passes_standard_parameters(self):
        calls = []

        def executor(source, output, **kwargs):
            calls.append((source, output, kwargs))

        notebook_root = self.root / "repository"
        notebook = notebook_root / "notebooks" / "Run.ipynb"
        notebook.parent.mkdir(parents=True)
        notebook.write_text("{}")
        output_root = self.root / "executed"
        output_root.mkdir()
        entry = {
            "notebook": "notebooks/Run.ipynb",
            "experiment": "sample",
            "run": "run",
            "creation_style": "discard-old",
        }

        output = run_notebook(
            entry, self.source, self.root / "results", output_root,
            notebook_root=notebook_root, executor=executor)

        self.assertEqual(output, output_root / "Run_sample_run.ipynb")
        self.assertEqual(len(calls), 1)
        source, called_output, kwargs = calls[0]
        self.assertEqual(source, notebook)
        self.assertEqual(called_output, output)
        self.assertEqual(kwargs["cwd"], notebook.parent)
        self.assertEqual(kwargs["parameters"], {
            "experiment": "sample",
            "run": "run",
            "creation_style": "discard-old",
            "expruns_path": self.source.as_posix(),
            "results_path": (self.root / "results").as_posix(),
        })

    def test_run_notebook_does_not_swallow_failure(self):
        def executor(*args, **kwargs):
            raise RuntimeError("failed")

        entry = {
            "notebook": "missing.ipynb",
            "experiment": "sample",
            "run": "run",
            "creation_style": "exist-ok",
        }
        with self.assertRaisesRegex(RuntimeError, "failed"):
            run_notebook(
                entry, self.source, self.root / "results", self.root,
                notebook_root=self.root, executor=executor)

    def test_build_flow_entries_has_exact_phase_order(self):
        experiments = {
            ("benchmark", "all"): {
                "tocompare": ["a", "b"],
                "input-to-notebook": ["Compare.ipynb"],
            },
            ("benchmark", "a"): {
                "exp_environment": "environment",
                "run_environment": "shared",
                "input-to-notebook": ["Run.ipynb", "Visualize.ipynb"],
            },
            ("benchmark", "b"): {
                "exp_environment": "environment",
                "run_environment": "shared",
                "input-to-notebook": ["Run.ipynb", "Visualize.ipynb"],
            },
            ("environment", "shared"): {
                "input-to-notebook": ["Precompute.ipynb"],
            },
        }

        class RecordingConfig:
            def __init__(self):
                self.create_data_dir_values = []

            def get_experiment(
                    self, experiment, run, create_data_dir=True):
                self.create_data_dir_values.append(create_data_dir)
                return experiments[(experiment, run)]

        config = RecordingConfig()
        entries = build_flow_entries(
            "benchmark", "all", "discard-old", config)
        self.assertTrue(config.create_data_dir_values)
        self.assertFalse(any(config.create_data_dir_values))
        self.assertEqual(
            [(entry["notebook"], entry["run"], entry["creation_style"])
             for entry in entries],
            [
                ("Precompute.ipynb", "shared", "discard-old"),
                ("Run.ipynb", "a", "discard-old"),
                ("Run.ipynb", "b", "discard-old"),
                ("Visualize.ipynb", "a", "exist-ok"),
                ("Visualize.ipynb", "b", "exist-ok"),
                ("Compare.ipynb", "all", "discard-old"),
            ])

    def test_custom_environment_reuses_precalculated_cache(self):
        data_dir = self.root / "custom-environment"
        data_dir.mkdir()
        (data_dir / "farm_geometry").write_bytes(b"cached")
        exp = {
            "data_dir": str(data_dir),
            "custom-tylcv": "custom.png",
        }
        farm = object()
        environment = object()
        with mock.patch("wbf_helper.compress.open", mock.mock_open()), \
                mock.patch("wbf_helper.pickle.load", return_value=farm), \
                mock.patch(
                    "wbf_helper.WaterberryFarmEnvironment",
                    return_value=environment) as constructor:
            loaded_farm, loaded_environment = create_wbfe(exp)
        self.assertIs(loaded_farm, farm)
        self.assertIs(loaded_environment, environment)
        constructor.assert_called_once_with(
            farm, use_saved=True, seed=10, savedir=str(data_dir))


if __name__ == "__main__":
    unittest.main()
