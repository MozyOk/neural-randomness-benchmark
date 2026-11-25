# Neural Randomness Benchmark

A controlled benchmark of next-bit prediction on apparently random sequences, based on the experimental task of [Fan & Wang (2018)](https://arxiv.org/abs/1801.01117).

## Reproduce Sprint 01

Use CPython 3.12.16, recorded in `.python-version`. GitHub Actions uses the same runtime and pinned dependencies.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -m unittest discover -s tests -v
sh scripts/smoke.sh --output /tmp/sprint-01-smoke.json
```

The checked-in [raw smoke artifact](artifacts/sprint-01-smoke.json) contains 30 fits across ten source/stream combinations, including three model initializations per combination. All six source types use 512 training, 128 validation and 256 test targets at raw offsets 0, 600 and 800. The test suite reruns the CLI and checks hashes, raw bits, hard-call counts and scores. It allows floating-point tolerance for smooth metrics across platforms.

## Run a custom exploratory benchmark

```sh
python benchmark.py --sources pi e iid mt periodic sqrt2 \
  --train 4000 --train-offset 0 \
  --validation 1000 --validation-offset 5000 \
  --test 10000 --test-offset 10000 \
  --model-seeds 101 202 303 --data-seeds 1101 2202 3303 \
  --output /tmp/benchmark.json
```

Split sizes count targets. An offset identifies the start of the six-bit warm-up in the raw sequence, so a split with `n` targets consumes `n + 6` symbols. Offsets are zero-based. Blocks must be chronological and disjoint. Unspecified offsets follow the end of the preceding raw block. For large constant datasets use `--pi-digits`, `--e-digits` or `--sqrt2-digits` with fractional decimal digit files that contain only ASCII digits and optional ASCII whitespace. The source representation and hash encoding are recorded in the output.

The sources are periodic `0011`, thresholded NumPy PCG64 and MT19937 double draws, and fractional digits of π, e and √2 thresholded at 5. Model seeds change initialization. Data seeds select distinct PCG64 or MT19937 streams. Each fit uses training only, then scores the fixed fair forecast, smoothed train majority, smoothed six-context n-gram and 6–30–20–1 MLP. The output includes the environment, configuration, exact raw binary blocks and per-context probability tables, with enough information to reconstruct every score. Accuracy predicts 1 at a tie. Log loss uses bounded probabilities to keep scores finite.

Read the [original replication gap audit](docs/original-replication-gap.md) for MATLAB settings and saved-model replay. Read the [candidate statistical protocol](docs/research-protocol-v1.md) for the confirmatory design. The benchmark CLI and bundled results are exploratory. Confirmatory testing and independent review are pending.
