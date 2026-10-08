"""Checks the Waterberry Farms exp/run configurations. The exp/run framework
itself is tested in ExpRunFlow."""

import pathlib
import unittest

import yaml


class TestConfig(unittest.TestCase):
    def test_supported_configuration_contains_parameters_not_code_selectors(self):
        src = pathlib.Path(__file__).resolve().parents[1]
        configuration_roots = [
            src.parent / "data" / "expruns",
            src / "papers" / "y2025_mrmr" / "data" / "expruns",
        ]
        code_selectors = {
            "policy-code", "policy-code-generator",
            "estimator-code", "score-code",
        }

        def keys(value):
            if isinstance(value, dict):
                for key, nested in value.items():
                    yield key
                    yield from keys(nested)
            elif isinstance(value, list):
                for nested in value:
                    yield from keys(nested)

        for root in configuration_roots:
            for path in root.rglob("*.yaml"):
                with path.open() as handle:
                    values = yaml.safe_load(handle)
                self.assertFalse(code_selectors.intersection(keys(values)))


if __name__ == "__main__":
    unittest.main()
