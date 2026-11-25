import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmark import (
    Block, context_targets, constant_digits, evaluate, make_blocks,
    run, scores_for_models, validate_settings,
)


class BenchmarkTests(unittest.TestCase):
    def test_blocks_are_disjoint_and_windows_stay_inside(self):
        blocks = make_blocks((2, 12, 21), (3, 2, 4))
        self.assertEqual([(b.start, b.stop) for b in blocks], [(2, 11), (12, 20), (21, 31)])
        codes, labels = context_targets(np.array([1, 0, 1, 0, 1, 1, 0, 1], dtype=np.uint8))
        np.testing.assert_array_equal(codes, [43, 22])
        np.testing.assert_array_equal(labels, [0, 1])
        with self.assertRaisesRegex(ValueError, "overlap"):
            make_blocks((10, 2, 20), (2, 3, 4))
        with self.assertRaisesRegex(ValueError, "order"):
            make_blocks((12, 2, 20), (2, 3, 4))
        with self.assertRaises(ValueError):
            make_blocks((-1, 20, 40), (2, 3, 4))

    def test_constant_fractional_digit_prefixes(self):
        for source, expected in {"pi": b"141592653589", "e": b"718281828459", "sqrt2": b"414213562373"}.items():
            self.assertEqual(constant_digits(source, 12), expected)

    def test_metrics_and_tie_policy(self):
        summary = evaluate(np.array([0.5, 0.9, 0.25]), np.array([1, 1, 0]))
        self.assertEqual(summary["correct"], 3)
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertAlmostEqual(summary["brier"], (0.25 + 0.01 + 0.0625) / 3)
        self.assertAlmostEqual(summary["log_loss"], -np.log(0.5 * 0.9 * 0.75) / 3)

    def test_train_only_baselines_and_original_style_model(self):
        bits = np.resize(np.array([0, 0, 1, 1], dtype=np.uint8), 406)
        probabilities = scores_for_models(bits, 101)
        self.assertEqual(probabilities["majority"][0], 0.5)
        self.assertEqual(probabilities["ngram"][0], 0.5)
        self.assertEqual(probabilities["ngram"][12], 101 / 102)
        self.assertEqual(probabilities["ngram"][51], 1 / 102)
        codes, y = context_targets(bits)
        for name in ("ngram", "mlp"):
            result = evaluate(probabilities[name][codes], y)
            self.assertEqual(result["accuracy"], 1.0)

    def test_artifact_can_be_recomputed_and_seed_axes_are_explicit(self):
        blocks = make_blocks((5, 65, 120), (32, 20, 24))
        report = run(("iid", "mt", "pi"), blocks, (11, 22), (33, 44))
        self.assertEqual(len(report["datasets"]), 5)
        for dataset in report["datasets"]:
            self.assertEqual(len(dataset["runs"]), 2)
            block = dataset["blocks"]["test"]
            self.assertEqual((block["start"], block["stop"], block["first_target"]), (120, 150, 126))
            bits = bytes(int(bit) for bit in block["bits"])
            self.assertEqual(hashlib.sha256(bits).hexdigest(), block["sha256"])
            codes, labels = context_targets(np.array(list(bits), dtype=np.uint8))
            for row in dataset["runs"]:
                for model in row["models"].values():
                    self.assertEqual(model["metrics"]["test"], evaluate(np.array(model["p1_by_context"])[codes], labels))
        repeated = run(("iid", "mt", "pi"), blocks, (11, 22), (33, 44))
        self.assertEqual(json.dumps(report, sort_keys=True), json.dumps(repeated, sort_keys=True))
        iid, iid_next, mt, *_ = report["datasets"]
        self.assertNotEqual(iid["source_sha256"], iid_next["source_sha256"])
        self.assertNotEqual(iid["source_sha256"], mt["source_sha256"])

    def test_external_decimal_digits_are_validated_and_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "digits.txt"
            path.write_bytes(b" 141592653589\n" * 10)
            blocks = make_blocks((0, 10, 20), (4, 4, 4))
            report = run(("pi",), blocks, (11,), (33,), {"pi": path})
            dataset = report["datasets"][0]
            self.assertEqual(dataset["source_file_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(dataset["blocks"]["train"]["bits"], "0001101101")
            path.write_bytes(b"3.14")
            with self.assertRaisesRegex(ValueError, "decimal digits"):
                run(("pi",), blocks, (11,), (33,), {"pi": path})

    def test_settings_reject_duplicate_seeds_and_invalid_blocks(self):
        for args in [((3, 3), (4,)), ((3,), (-1,)), ((), (4,))]:
            with self.assertRaises(ValueError):
                validate_settings(*args)
        with self.assertRaises(ValueError):
            Block(0, 6)


if __name__ == "__main__":
    unittest.main()
