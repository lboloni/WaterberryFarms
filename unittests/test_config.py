import pathlib
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from exp_run_config import Config


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary_directory.name)
        self.experiments = self.root / "experiments"
        self.system = self.root / "system"
        self.data = self.root / "data"
        (self.experiments / "sample").mkdir(parents=True)
        (self.system / "sample").mkdir(parents=True)
        self.write_yaml(
            self.experiments / "sample" / "_defaults_sample.yaml",
            {"default": 1, "overridden": "default"})
        self.write_yaml(
            self.experiments / "sample" / "run.yaml",
            {"overridden": "run", "run-only": 2})
        self.write_yaml(
            self.system / "sample" / "run_sysdep.yaml",
            {"overridden": "system", "system-only": 3})
        self.config = object.__new__(Config)
        self.config.values = {
            "experiment_data": self.data,
            "experiment_system_dependent_dir": self.system,
        }
        self.config.experiment_path = self.experiments
        self.config.experiment_path_internal = self.experiments

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def write_yaml(path, values):
        with path.open("w") as handle:
            yaml.safe_dump(values, handle)

    def test_configuration_precedence(self):
        exp = self.config.get_experiment("sample", "run")
        self.assertEqual(exp["default"], 1)
        self.assertEqual(exp["run-only"], 2)
        self.assertEqual(exp["system-only"], 3)
        self.assertEqual(exp["overridden"], "system")
        self.assertTrue(exp.data_dir().exists())

    def test_version_creates_new_directory_when_absent(self):
        exp = self.config.get_experiment("sample", "run", "new", creation_style="version")
        self.assertTrue(exp.data_dir().exists())
        self.assertTrue((exp.data_dir() / "exprun.yaml").exists())

    def test_version_moves_existing_directory(self):
        exp = self.config.get_experiment("sample", "run", creation_style="exist-ok")
        marker = exp.data_dir() / "marker"
        marker.write_text("old")
        new_exp = self.config.get_experiment("sample", "run", creation_style="version")
        self.assertFalse((new_exp.data_dir() / "marker").exists())
        backups = list(new_exp.data_dir().parent.glob("run_????-??-??-??-??-??"))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / "marker").read_text(), "old")

    def test_discard_old_removes_previous_content(self):
        exp = self.config.get_experiment("sample", "run", creation_style="exist-ok")
        (exp.data_dir() / "marker").write_text("old")
        exp = self.config.get_experiment("sample", "run", creation_style="discard-old")
        self.assertFalse((exp.data_dir() / "marker").exists())

    def test_unknown_creation_style_fails(self):
        with self.assertRaisesRegex(Exception, "Unknown creation_style"):
            self.config.get_experiment("sample", "run", creation_style="unknown")

    def test_missing_run_fails(self):
        with self.assertRaisesRegex(Exception, "Missing experiment file"):
            self.config.get_experiment("sample", "missing")


if __name__ == "__main__":
    unittest.main()
