import json
import pathlib
import re
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from wbf_flow import (
    build_flow_entries,
    build_mrmr_2027_flow_entries,
    get_flow_report,
    run_flow,
    run_notebook,
    setup_flow,
)
from wbf_helper import create_wbfe


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = REPOSITORY_ROOT / "data" / "expruns"
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
MRMR_2027_NOTEBOOKS = [
    REPOSITORY_ROOT / "papers" / "y2027_mrmr" / name
    for name in (
        "MRMR-Run.ipynb",
        "MRMR-Visualize-DetectionMap.ipynb",
        "MRMR-Visualize-AgentDetections.ipynb",
        "MRMR-Visualize-Replanning.ipynb",
        "MRMR-Visualize-Comparison.ipynb",
        "MRMR-Visualize-OptimalEPPath.ipynb",
        "MRMR-Flow.ipynb",
    )
]
FLOW_NOTEBOOKS += MRMR_2027_NOTEBOOKS
TOP_LEVEL_FLOW_NOTEBOOKS = [
    REPOSITORY_ROOT / "notebooks" / "Flow-1Robot1Day.ipynb",
    REPOSITORY_ROOT / "notebooks" / "Flow-nRobot1Day.ipynb",
    REPOSITORY_ROOT / "papers" / "y2027_mrmr" / "MRMR-Flow.ipynb",
]


class FlowConfig:
    def __init__(self, exprun_path, flows_path, results_path):
        self.exprun_path = exprun_path
        self.values = {
            "flows_path": flows_path,
            "experiment_data": results_path,
        }

    def __getitem__(self, key):
        return self.values[key]

    def get_exprun_path(self):
        return self.exprun_path

    def set_exprun_path(self, path):
        self.exprun_path = path

    def set_results_path(self, path):
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
        for path in FLOW_NOTEBOOKS[:7] + MRMR_2027_NOTEBOOKS:
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

    def test_mrmr_2027_notebooks_list_every_compatible_exprun(self):
        expected = {}
        for family in (
                "mrmr2027-run", "mrmr2027-figure", "mrmr2027-flow"):
            family_path = EXPERIMENT_ROOT / family
            defaults_path = family_path / f"_defaults_{family}.yaml"
            with defaults_path.open() as handle:
                defaults = yaml.safe_load(handle) or {}
            for run_path in family_path.glob("*.yaml"):
                if run_path == defaults_path:
                    continue
                with run_path.open() as handle:
                    values = defaults | (yaml.safe_load(handle) or {})
                notebook = values["input-to-notebook"][0]
                expected.setdefault(notebook, set()).add(run_path.stem)

        for path in MRMR_2027_NOTEBOOKS:
            relative_path = path.relative_to(REPOSITORY_ROOT).as_posix()
            with path.open() as handle:
                notebook = json.load(handle)
            parameter_cell = next(
                cell for cell in notebook["cells"]
                if "parameters" in cell["metadata"].get("tags", []))
            source = "".join(parameter_cell["source"])
            listed = set(re.findall(
                r'^\s*#?\s*run = "([^"]+)"', source, re.MULTILINE))
            self.assertEqual(listed, expected[relative_path], path.name)

    def test_mrmr_2027_flow_declares_optimal_ep_figure(self):
        path = EXPERIMENT_ROOT / "mrmr2027-flow" / "icc-2027-all.yaml"
        with path.open() as handle:
            collection = yaml.safe_load(handle)
        self.assertEqual(len(collection["figures"]), 17)
        self.assertIn("optimal-ep-path", collection["figures"])

        path = (EXPERIMENT_ROOT / "mrmr2027-figure" /
                "optimal-ep-path.yaml")
        with path.open() as handle:
            figure = yaml.safe_load(handle)
        self.assertEqual(figure["source-runs"], [])
        self.assertEqual(figure["input-to-notebook"], [
            "papers/y2027_mrmr/MRMR-Visualize-OptimalEPPath.ipynb",
        ])

    def test_flow_notebooks_finish_with_result_report(self):
        for path in TOP_LEVEL_FLOW_NOTEBOOKS:
            with path.open() as handle:
                notebook = json.load(handle)
            final_cell = notebook["cells"][-1]
            self.assertEqual(final_cell["id"], "flow-report", path.name)
            source = "".join(final_cell["source"])
            self.assertIn("display_flow_report(", source, path.name)
            self.assertIn("raise flow_error", source, path.name)

    def test_mrmr_flow_embeds_all_generated_figure_previews(self):
        path = (REPOSITORY_ROOT / "papers" / "y2027_mrmr" /
                "MRMR-Flow.ipynb")
        with path.open() as handle:
            notebook = json.load(handle)
        source = "".join(notebook["cells"][-1]["source"])
        self.assertIn("build_figure_previews(", source)
        self.assertIn('flow_exp["figures"]', source)


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

    def test_setup_flow_copies_from_active_exprun_path(self):
        expruns, results, notebooks = setup_flow(
            "sample-flow", ["sample"], config=self.config)
        self.assertTrue((expruns / "sample" / "run.yaml").is_file())
        self.assertEqual(self.config.exprun_path, expruns)
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

    def test_run_flow_reports_completed_and_remaining_notebooks(self):
        entries = [{"name": "first"}, {"name": "second"}]
        calls = []

        class RecordingProgress:
            def __init__(self, **kwargs):
                self.n = 0
                self.options = kwargs
                self.labels = []
                self.closed = False

            def set_postfix_str(self, label):
                self.labels.append(label)

            def update(self, count):
                self.n += count

            def close(self):
                self.closed = True

        progress = RecordingProgress()

        def run(entry, expruns, results, notebooks):
            calls.append((entry, expruns, results, notebooks))

        def progress_factory(**options):
            progress.options = options
            return progress

        run_flow(
            entries, self.source, self.root / "results", self.root,
            notebook_runner=run,
            progress_factory=progress_factory,
        )

        self.assertEqual([call[0] for call in calls], entries)
        self.assertEqual(progress.options, {
            "total": 2,
            "desc": "Overall flow",
            "unit": "notebook",
        })
        self.assertEqual(progress.n, 2)
        self.assertEqual(progress.labels[-1], "0 notebooks left")
        self.assertTrue(progress.closed)

    def test_run_flow_closes_progress_and_propagates_failure(self):
        class RecordingProgress:
            n = 0
            closed = False

            def set_postfix_str(self, label):
                pass

            def update(self, count):
                self.n += count

            def close(self):
                self.closed = True

        progress = RecordingProgress()

        def fail(*args):
            raise RuntimeError("failed")

        with self.assertRaisesRegex(RuntimeError, "failed"):
            run_flow(
                [{"name": "failing"}], self.source,
                self.root / "results", self.root,
                notebook_runner=fail,
                progress_factory=lambda **kwargs: progress,
            )
        self.assertEqual(progress.n, 0)
        self.assertTrue(progress.closed)

    def test_flow_report_distinguishes_complete_and_partial_results(self):
        results = self.root / "results"
        complete = results / "sample" / "complete"
        complete.mkdir(parents=True)
        (complete / "exprun.yaml").write_text(
            "time_done: '2026-09-27 12:00:00.000000'\n")
        entries = [
            {
                "name": "complete",
                "experiment": "sample",
                "run": "complete",
            },
            {
                "name": "missing",
                "experiment": "sample",
                "run": "missing",
            },
        ]

        report = get_flow_report(entries, results)
        self.assertFalse(report["successful"])
        self.assertFalse(report["all-results-present"])
        self.assertEqual(
            [stage["name"] for stage in report["completed"]],
            ["complete"])
        self.assertEqual(
            [stage["name"] for stage in report["missing"]],
            ["missing"])

        (results / "sample" / "missing").mkdir()
        (results / "sample" / "missing" / "exprun.yaml").write_text(
            "time_done: '2026-09-27 12:01:00.000000'\n")
        report = get_flow_report(entries, results)
        self.assertTrue(report["successful"])
        self.assertTrue(report["all-results-present"])

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

    def test_build_mrmr_2027_flow_entries_has_exact_phase_order(self):
        experiments = {
            ("paper-flow", "all"): {
                "run-experiment": "paper-run",
                "runs": ["a", "b"],
                "figure-experiment": "paper-figure",
                "figures": ["map-a", "optimal"],
            },
            ("paper-run", "a"): {
                "exp_environment": "environment",
                "run_environment": "shared",
                "input-to-notebook": ["Run.ipynb"],
            },
            ("paper-run", "b"): {
                "exp_environment": "environment",
                "run_environment": "shared",
                "input-to-notebook": ["Run.ipynb"],
            },
            ("environment", "shared"): {
                "input-to-notebook": ["Precompute.ipynb"],
            },
            ("paper-figure", "map-a"): {
                "source-experiment": "paper-run",
                "source-runs": ["a"],
                "input-to-notebook": ["Map.ipynb"],
            },
            ("paper-figure", "optimal"): {
                "source-experiment": "paper-run",
                "source-runs": [],
                "input-to-notebook": ["Optimal.ipynb"],
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
        entries = build_mrmr_2027_flow_entries(
            "paper-flow", "all", "discard-old", config)

        self.assertFalse(any(config.create_data_dir_values))
        self.assertEqual(
            [(entry["notebook"], entry["experiment"], entry["run"])
             for entry in entries],
            [
                ("Precompute.ipynb", "environment", "shared"),
                ("Run.ipynb", "paper-run", "a"),
                ("Run.ipynb", "paper-run", "b"),
                ("Map.ipynb", "paper-figure", "map-a"),
                ("Optimal.ipynb", "paper-figure", "optimal"),
            ])

    def test_build_mrmr_2027_flow_rejects_undeclared_source(self):
        experiments = {
            ("paper-flow", "all"): {
                "run-experiment": "paper-run",
                "runs": ["a"],
                "figure-experiment": "paper-figure",
                "figures": ["bad"],
            },
            ("paper-run", "a"): {
                "exp_environment": "environment",
                "run_environment": "shared",
                "input-to-notebook": ["Run.ipynb"],
            },
            ("environment", "shared"): {
                "input-to-notebook": ["Precompute.ipynb"],
            },
            ("paper-figure", "bad"): {
                "source-experiment": "paper-run",
                "source-runs": ["missing"],
                "input-to-notebook": ["Figure.ipynb"],
            },
        }

        class ConfigWithBadSource:
            def get_experiment(
                    self, experiment, run, create_data_dir=True):
                return experiments[(experiment, run)]

        with self.assertRaisesRegex(Exception, "undeclared run missing"):
            build_mrmr_2027_flow_entries(
                "paper-flow", "all", "exist-ok",
                ConfigWithBadSource())

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

    def test_custom_environment_cache_has_consistent_geometry(self):
        data_dir = self.root / "custom-environment"
        data_dir.mkdir()
        exprun_dir = self.root / "expruns" / "environment"
        exprun_dir.mkdir(parents=True)
        exprun_path = exprun_dir / "custom.yaml"
        exprun_path.write_text("custom\n")
        (exprun_dir / "custom.png").touch()
        exp = {
            "typename": "Miniberry-10",
            "data_dir": str(data_dir),
            "exp_run_sys_indep_file": str(exprun_path),
            "custom-tylcv": "custom.png",
        }

        with mock.patch(
                "wbf_helper.imageio.imread",
                return_value=np.zeros((11, 11))):
            farm, environment = create_wbfe(exp)

        self.assertEqual(
            farm.type_map.shape, (farm.width, farm.height))
        self.assertEqual(
            environment.my_owner_mask.shape, farm.type_map.shape)
        self.assertFalse(environment.my_strawberry_mask.any())
        self.assertTrue(np.array_equal(
            environment.my_tomato_mask, environment.my_owner_mask))
        np.testing.assert_array_equal(
            environment.tylcv.value, np.zeros((11, 11)))
        environment.proceed(1)
        np.testing.assert_array_equal(
            environment.tylcv.value, np.zeros((11, 11)))

        cached_farm, cached_environment = create_wbfe(exp)
        cached_environment.proceed(1)
        self.assertEqual(
            cached_farm.type_map.shape,
            (cached_environment.width, cached_environment.height))
        np.testing.assert_array_equal(
            cached_environment.tylcv.value, np.zeros((11, 11)))


if __name__ == "__main__":
    unittest.main()
