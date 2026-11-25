import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class SmokeArtifactTests(unittest.TestCase):
    def test_smoke_cli_reproduces_raw_artifact(self):
        recorded = json.loads((ROOT / "artifacts/sprint-01-smoke.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke.json"
            subprocess.run(
                ["sh", "scripts/smoke.sh", "--output", str(output)],
                cwd=ROOT, check=True,
                env={**os.environ, "PATH": f"{Path(sys.executable).parent}:/usr/bin:/bin"},
            )
            reproduced = json.loads(output.read_text())

        for report in (recorded, reproduced):
            del report["environment"]["platform"]

        self.assertEqual(len(recorded["datasets"]), 10)
        self.assertEqual(len(reproduced["datasets"]), 10)
        for saved, fresh in zip(recorded["datasets"], reproduced["datasets"], strict=True):
            self.assertEqual(len(saved["runs"]), 3)
            for left, right in zip(saved["runs"], fresh["runs"], strict=True):
                for name in left["models"]:
                    a, b = left["models"][name], right["models"][name]
                    np.testing.assert_allclose(
                        a.pop("p1_by_context"), b.pop("p1_by_context"), rtol=0, atol=1e-10,
                    )
                    for split in a["metrics"]:
                        for metric in ("brier", "log_loss"):
                            self.assertAlmostEqual(
                                a["metrics"][split].pop(metric), b["metrics"][split].pop(metric), places=9,
                            )
        self.assertEqual(recorded, reproduced)
