"""Exploratory six-bit next-symbol benchmark with auditable raw blocks."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform

import mpmath
import numpy as np


CONTEXT = 6
PATTERNS = 2 ** CONTEXT
SOURCES = ("iid", "periodic", "mt", "pi", "e", "sqrt2")
GENERATORS = {"iid": np.random.PCG64, "mt": np.random.MT19937}
SPLITS = ("train", "validation", "test")
PERIOD = np.array([0, 0, 1, 1], dtype=np.uint8)
EPOCHS = 40
LEARNING_RATE = 0.03
LOG_EPSILON = 1e-15


@dataclass(frozen=True)
class Block:
    start: int
    stop: int

    def __post_init__(self):
        if self.start < 0 or self.stop - self.start <= CONTEXT:
            raise ValueError("each block needs a nonnegative offset and at least seven symbols")


def make_blocks(offsets, windows):
    if len(offsets) != len(SPLITS) or len(windows) != len(SPLITS):
        raise ValueError("exactly three blocks are required")
    blocks = [Block(start, start + count + CONTEXT) for start, count in zip(offsets, windows, strict=True)]
    ordered = sorted(blocks, key=lambda block: block.start)
    if any(left.stop > right.start for left, right in zip(ordered, ordered[1:])):
        raise ValueError("raw sequence blocks overlap")
    if blocks != ordered:
        raise ValueError("raw blocks must be in train, validation, test order")
    return blocks


def validate_settings(model_seeds, data_seeds):
    for seeds in (model_seeds, data_seeds):
        if not seeds or len(seeds) != len(set(seeds)) or any(not 0 <= seed < 2**64 for seed in seeds):
            raise ValueError("each seed list must contain distinct unsigned 64-bit integers")


def constant_digits(source, length):
    def render(guard):
        with mpmath.workdps(length + guard):
            value = {"pi": lambda: mpmath.pi, "e": lambda: mpmath.e, "sqrt2": lambda: mpmath.sqrt(2)}[source]()
            fractional = mpmath.nstr(value, length + guard).split(".")[1]
            return fractional[:length].encode("ascii")

    digits = render(16)
    if len(digits) != length or digits != render(32):
        raise ValueError("constant digit precision check failed")
    return digits


def sequence(source, length, seed, path=None):
    if source in GENERATORS:
        generator = GENERATORS[source]
        draws = np.random.Generator(generator(seed)).random(length)
        return (draws >= 0.5).astype(np.uint8), {
            "representation": f"NumPy {generator.__name__} float64 draws >= 0.5",
            "source_sha256": sha256(draws.astype("<f8", copy=False).tobytes()).hexdigest(),
            "source_hash_encoding": "little-endian float64 draws",
        }
    if source == "periodic":
        bits = np.resize(PERIOD, length)
        return bits, {
            "representation": "0011 repeated from offset 0",
            "source_sha256": sha256(bits.tobytes()).hexdigest(),
            "source_hash_encoding": "one uint8 per bit",
        }
    metadata = {"representation": "fractional decimal digits >= 5"}
    if path is None:
        digits = constant_digits(source, length)
        metadata["digit_origin"] = "mpmath with 16 and 32 guard decimal digits"
    else:
        raw = path.read_bytes()
        digits = b"".join(raw.split())
        if not digits.isdigit() or len(digits) < length:
            raise ValueError(f"{path}: expected at least {length} decimal digits with optional ASCII whitespace")
        digits = digits[:length]
        metadata["digit_origin"] = "external fractional-digit file"
        metadata["source_file_sha256"] = sha256(raw).hexdigest()
    metadata["source_sha256"] = sha256(digits).hexdigest()
    metadata["source_hash_encoding"] = "ASCII digits, whitespace removed"
    return (np.frombuffer(digits, dtype=np.uint8) >= ord("5")).astype(np.uint8), metadata


def context_targets(bits):
    count = len(bits) - CONTEXT
    codes = np.zeros(count, dtype=np.uint8)
    for position in range(CONTEXT):
        codes = (codes << 1) | bits[position:position + count]
    return codes, bits[CONTEXT:]


def fit_mlp(totals, positives, seed):
    """Fit a 6-30-20-1 sigmoid/sigmoid/linear MLP on window MSE."""
    contexts = (np.arange(PATTERNS)[:, None] >> np.arange(CONTEXT - 1, -1, -1)) & 1
    rng = np.random.Generator(np.random.PCG64(seed))
    weights = []
    for fan_in, fan_out in ((6, 30), (30, 20), (20, 1)):
        bound = np.sqrt(6 / (fan_in + fan_out))
        weights.extend((rng.uniform(-bound, bound, (fan_in, fan_out)), np.zeros(fan_out)))
    first_moments = [np.zeros_like(parameter) for parameter in weights]
    second_moments = [np.zeros_like(parameter) for parameter in weights]

    def forward():
        hidden1 = 1 / (1 + np.exp(-(contexts @ weights[0] + weights[1])))
        hidden2 = 1 / (1 + np.exp(-(hidden1 @ weights[2] + weights[3])))
        return hidden1, hidden2, hidden2 @ weights[4] + weights[5]

    for epoch in range(1, EPOCHS + 1):
        hidden1, hidden2, predicted = forward()
        # Count-weighted gradients match full-batch gradients over the original windows.
        output_delta = 2 * (totals * predicted[:, 0] - positives)[:, None] / totals.sum()
        hidden2_delta = (output_delta @ weights[4].T) * hidden2 * (1 - hidden2)
        hidden1_delta = (hidden2_delta @ weights[2].T) * hidden1 * (1 - hidden1)
        gradients = (
            contexts.T @ hidden1_delta, hidden1_delta.sum(axis=0),
            hidden1.T @ hidden2_delta, hidden2_delta.sum(axis=0),
            hidden2.T @ output_delta, output_delta.sum(axis=0),
        )
        for parameter, first, second, gradient in zip(weights, first_moments, second_moments, gradients, strict=True):
            first *= 0.9
            first += 0.1 * gradient
            second *= 0.999
            second += 0.001 * gradient**2
            corrected_first = first / (1 - 0.9**epoch)
            corrected_second = second / (1 - 0.999**epoch)
            parameter -= LEARNING_RATE * corrected_first / (np.sqrt(corrected_second) + 1e-8)
    return np.clip(forward()[2][:, 0], 0.0, 1.0)


def scores_for_models(train_bits, seed):
    codes, y = context_targets(train_bits)
    totals = np.bincount(codes, minlength=PATTERNS)
    positives = np.bincount(codes, weights=y, minlength=PATTERNS)
    return {
        "fair": np.full(PATTERNS, 0.5),
        "majority": np.full(PATTERNS, (positives.sum() + 1) / (len(y) + 2)),
        "ngram": (positives + 1) / (totals + 2),
        "mlp": fit_mlp(totals, positives, seed),
    }


def evaluate(probabilities, y):
    correct = int(np.count_nonzero((probabilities >= 0.5) == y))
    safe = np.clip(probabilities, LOG_EPSILON, 1 - LOG_EPSILON)
    return {
        "windows": len(y),
        "correct": correct,
        "accuracy": correct / len(y),
        "log_loss": float(-np.mean(np.where(y, np.log(safe), np.log1p(-safe)))),
        "brier": float(np.mean((probabilities - y)**2)),
    }


def run(sources, blocks, model_seeds, data_seeds, digit_files=None):
    validate_settings(model_seeds, data_seeds)
    offsets = [block.start for block in blocks]
    lengths = [block.stop - block.start - CONTEXT for block in blocks]
    make_blocks(offsets, lengths)
    digit_files = digit_files or {}
    datasets = []
    for source in sources:
        for data_seed in data_seeds if source in GENERATORS else (None,):
            length = max(block.stop for block in blocks)
            bits, metadata = sequence(source, length, data_seed, digit_files.get(source))
            block_bits = {name: bits[block.start:block.stop] for name, block in zip(SPLITS, blocks, strict=True)}
            examples = {name: context_targets(raw) for name, raw in block_bits.items()}
            block_records = {
                name: {
                    "start": block.start, "stop": block.stop,
                    "first_target": block.start + CONTEXT, "windows": count,
                    "sha256": sha256(block_bits[name].tobytes()).hexdigest(),
                    "bits": (block_bits[name] + ord("0")).tobytes().decode("ascii"),
                }
                for name, block, count in zip(SPLITS, blocks, lengths, strict=True)
            }
            runs = []
            for model_seed in model_seeds:
                result = {}
                for model, table in scores_for_models(block_bits["train"], model_seed).items():
                    result[model] = {
                        "p1_by_context": table.tolist(),
                        "metrics": {name: evaluate(table[codes], y) for name, (codes, y) in examples.items()},
                    }
                runs.append({"model_seed": model_seed, "models": result})
            datasets.append({
                "source": source, "data_seed": data_seed, "source_length": length,
                **metadata, "blocks": block_records, "runs": runs,
            })
    return {
        "schema_version": 1, "purpose": "exploratory",
        "implementation_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "mpmath": mpmath.__version__},
        "configuration": {
            "sources": list(sources), "model_seeds": list(model_seeds), "data_seeds": list(data_seeds),
            "context": CONTEXT, "context_order": "oldest bit most significant",
            "offsets": dict(zip(SPLITS, offsets)), "windows": dict(zip(SPLITS, lengths)),
            "tie_predicts": 1, "log_loss_epsilon": LOG_EPSILON, "baseline_pseudocount_per_class": 1,
            "block_hash_encoding": "one uint8 per bit",
            "mlp": {"layers": [6, 30, 20, 1], "activations": ["sigmoid", "sigmoid", "linear"],
                    "loss": "MSE", "optimizer": "full-batch Adam", "epochs": EPOCHS,
                    "learning_rate": LEARNING_RATE, "betas": [0.9, 0.999], "epsilon": 1e-8,
                    "initialization": "Glorot uniform, zero biases, NumPy PCG64",
                    "scores": "linear output clipped to [0,1]"},
        },
        "datasets": datasets,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=SOURCES)
    parser.add_argument("--model-seeds", nargs="+", type=int, default=(101, 202, 303))
    parser.add_argument("--data-seeds", nargs="+", type=int, default=(1101, 2202, 3303))
    for name, count in zip(SPLITS, (40000, 10000, 90000)):
        parser.add_argument(f"--{name}", type=int, default=count, help=f"{name} target count")
        parser.add_argument(f"--{name}-offset", type=int, help=f"{name} raw start (zero-based)")
    for source in ("pi", "e", "sqrt2"):
        parser.add_argument(f"--{source}-digits", type=Path, help="override with a fractional-digit file")
    parser.add_argument("--output", type=Path, help="JSON destination (default: stdout)")
    args = parser.parse_args()
    windows = [getattr(args, name) for name in SPLITS]
    offsets = []
    next_offset = 0
    for name, count in zip(SPLITS, windows):
        specified = getattr(args, f"{name}_offset")
        start = next_offset if specified is None else specified
        offsets.append(start)
        next_offset = start + count + CONTEXT
    digit_files = {source: getattr(args, f"{source}_digits") for source in ("pi", "e", "sqrt2")
                   if getattr(args, f"{source}_digits") is not None}
    try:
        if len(set(args.sources)) != len(args.sources) or not digit_files.keys() <= set(args.sources):
            raise ValueError("select each source once and supply digit files only for selected sources")
        report = run(args.sources, make_blocks(offsets, windows), args.model_seeds, args.data_seeds, digit_files)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    payload = json.dumps(report, allow_nan=False, separators=(",", ":")) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
