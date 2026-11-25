"""Unit tests for pure-Python helpers and CLI syntax in the MD pipeline."""

from __future__ import annotations

import ast
import importlib.util
import json
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_module_if_numpy(script_name: str):
    if importlib.util.find_spec("numpy") is None:
        return None
    spec = importlib.util.spec_from_file_location(script_name, SCRIPTS_DIR / f"{script_name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestPipelineStaticAndHelpers(unittest.TestCase):
    def test_all_python_scripts_parse_cleanly(self) -> None:
        for script in sorted(SCRIPTS_DIR.glob("*.py")):
            src = script.read_text(encoding="utf-8")
            ast.parse(src, filename=str(script))

    def test_colab_notebook_is_valid_json(self) -> None:
        nb_path = REPO_ROOT / "notebooks" / "colab_md_runner.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb.get("nbformat"), 4)
        self.assertGreaterEqual(len(nb.get("cells", [])), 5)

    def test_zaff_parameter_files_present(self) -> None:
        self.assertTrue((REPO_ROOT / "params" / "ZAFF.frcmod").is_file())
        self.assertTrue((REPO_ROOT / "params" / "ZAFF_H.prep").is_file())

    def test_parse_mutations_and_cap_geometry(self) -> None:
        mod = _load_module_if_numpy("01_build_system")
        if mod is None:
            self.skipTest("numpy not installed in base interpreter")
        import numpy as np

        self.assertEqual(mod.parse_mutations("WT"), {})
        self.assertEqual(
            mod.parse_mutations("A:R280K, B:R280T"),
            {"A": ["ARG-280-LYS"], "B": ["ARG-280-THR"]},
        )
        with self.assertRaises(ValueError):
            mod.parse_mutations("R280K")

        n = np.array([1.0, 0.0, 0.0])
        ca = np.array([0.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        ace = mod.compute_cap_coords(n, ca, c, is_nterm=True)
        nme = mod.compute_cap_coords(n, ca, c, is_nterm=False)
        self.assertAlmostEqual(float(np.linalg.norm(ace - n)), 1.33, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(nme - c)), 1.33, places=6)

    def test_block_average_sem(self) -> None:
        mod = _load_module_if_numpy("03_analyze_and_mmpbsa")
        if mod is None:
            self.skipTest("numpy not installed in base interpreter")
        import numpy as np

        vals = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0])
        # 5 blocks with means [1, 2, 3, 4, 5] -> sample SD = sqrt(2.5), SEM = sqrt(2.5 / 5) = sqrt(0.5)
        self.assertAlmostEqual(mod.block_average_sem(vals, n_blocks=5), float(np.sqrt(0.5)), places=10)


if __name__ == "__main__":
    unittest.main()
