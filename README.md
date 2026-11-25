# Neural Randomness Benchmark

Research Garden: a controlled benchmark of next-bit prediction on apparently random sequences. Inspired by Fan & Wang (2018), https://arxiv.org/abs/1801.01117. This is an independent reconstruction, not an exact replication.

## Quickstart

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python benchmark.py --train 4000 --test 10000
```

The default run uses 40,000 training and 90,000 test windows. Mathematical constants are thresholded at decimal digit >= 5. The train and test raw sequences do not overlap. Compare majority, smoothed n-gram, and 6-30-20-1-style MLP (scikit-learn defaults, not the original optimizer). Periodic data is a positive control; IID-like seeded NumPy data is a negative control. MT uses thresholded floating point output, not raw 32-bit state recovery.

## Statistical interpretation

One-sided exact binomial p-values test a fair IID null for a fixed causal predictor, not whether a deterministic constant is mathematically random. Bonferroni adjustment covers the tests in one declared run only. Repeated exploratory runs and correlated comparisons must not be treated as independent evidence. Seeds for mathematical constants change the model only, not the digit stream. Use disjoint digit blocks and held-out seeds before making any claim. The IID control is a PRNG, not physical randomness.

## Roadmap

- Exact original preprocessing/optimizer verification and 900k/9M held-out datasets.
- Preregistered independent blocks and seeds; confidence intervals and log loss.
- Transformer, SSM, LCG, PCG, Philox, ChaCha20, public quantum data.
- Context/compute scaling and reproducible environment lockfile.

No claim of successful prediction of mathematical constants or physical randomness is made.
