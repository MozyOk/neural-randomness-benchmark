# Fan & Wang (2018) source audit

This audit uses [arXiv v1, §§ II.A–C](https://arxiv.org/pdf/1801.01117v1) and the [authors' MATLAB repository at 1ea617b](https://github.com/WANG-AXIS/pseudo-Randomness/tree/1ea617b5d3a5e7a89928e44926ff415bb448fe4e). The repository cites the [IEEE Access version](https://doi.org/10.1109/ACCESS.2018.2826448). The full journal PDF was inaccessible during this audit. Paper details below refer to the preprint. Original scripts and datasets stay at upstream.

## What the preprint specifies

- For π, threshold decimal digits at 5 and slide a width-seven window with stride one. The first six bits feed a 6–30–20–1 feed-forward network and the seventh is the label. The authors specify 40,000 training windows and two π tests, T1 with 900,000 windows and T2 with 9,000,000 windows. Each has nine subgroups. Nets 1 and 2 are trained independently. The reported training accuracy is 51.28% and 51.03% respectively.
- The stated one-based ranges are training 1–40,007, T1 100,000–1,000,006 and T2 999,000–9,999,006. These contain one symbol more than required for the respective window counts. T1 and T2 also overlap if read literally. For e and √2 the text gives the training and T1-size test ranges. The paper leaves the exact leading digit convention and used π target offsets unresolved.
- The preprint specifies Levenberg–Marquardt backpropagation and MATLAB `initnw`, a 40-epoch maximum, momentum 0.95 and learning rate 0.05. It compares in-sample accuracy against a normal approximation to fair guesses, reporting σ=0.025 in proportion units. Under an independent fair correctness benchmark the standard error for 40,000 targets would instead be 0.0025 (0.25 percentage points), and fitting on the same targets invalidates that ordinary out-of-sample reference in any case. It reports one-sided Student-t lower confidence bounds using nine subgroup accuracy rates against 0.5.
- The MT experiment thresholds MATLAB uniform draws at 0.5, trains on 10,000 overlapping windows starting at raw one-based index 1 and scores the single window starting at 10,001. Six raw input symbols of that test window were already present in training windows. The 100 initializations per sequence share the same target and training sequence. Section II.B reports nine streams, although the Figure 3 caption says ten while listing nine counts. The maximum epochs change to 100 and the minimum gradient to 1e−10. Generator seeds, recorded versions and the runnable MT procedure are absent from the retrieved release.
- Section II.C calculates a seven-bit context maximum on the same observations used to identify each context's majority continuation. This in-sample statistic has selection bias relative to a fixed predictor's fair-null accuracy. Overlapping windows and deterministic digit streams do not justify the displayed finite-sample five-sigma bound as a test of asymptotic normality.

## What the released MATLAB evidence specifies

| Item | Evidence |
|---|---|
| Input material | `load2Me.m` takes the first 33,334 whitespace-separated tokens of `e.txt` and concatenates them using the initial width 57 and subsequent width 60. This excludes the integer 2 and leaves the final text token unused. The stored `ESource.mat` is 2,000,037 binary values. The digit transformation `digit >= 5` reproduces every stored bit and the count of 1,000,999 ones. The bundled `pi.txt` has 144,800 fractional digits, insufficient for either stated π test. |
| Training windows | Both e scripts read `ESource.mat`, use 40,000 width-seven windows from one-based starts 1..40,000, first six inputs, seventh target. Actual zero-based raw interval is [0, 40,006). |
| Model | The constructor calls `newff` with 30 and 20 hidden units, `logsig` in both and `purelin` output. Both saved network objects expose version field 8, 851 parameters, `initlay`/layer `initnw`, unregularized MSE and `trainlm`. Saved input processing is `fixunknowns → removeconstantrows → mapminmax` and output processing is `removeconstantrows → mapminmax`. The binary inputs and targets map from [0,1] to [−1,1]. Training/test reports threshold the restored output at 0.5 via `hardlim`. |
| Internal partition | Saved network metadata has `dividerand`, sample mode, ratios 70%/15%/15%. Scripts do not override the division. Window-level random splitting permits shared neighboring raw bits between internal partitions. `Pfm` evaluates all 40,000 supplied windows including internal holdbacks. Exact split indices, RNG state and toolbox release are unavailable. |
| Optimizer and epochs | The scripts assign epochs=40, min_grad=1e−4 and max_fail=200 and also assign mc=0.95 and lr=0.05. The saved `trainlm` parameter record has μ=0.001, factors 0.1 and 10, μ max=1e10 and no momentum or learning-rate field. The [documented LM update](https://www.mathworks.com/help/deeplearning/ref/trainlm.html) uses μ damping. The exact behavior of those assignments in the authors' MATLAB environment is unverified. The saved `net` was constructed again after training and had weights copied into it. Its default epochs=1000, min_grad=1e−7 and max_fail=6 describe the rebuilt object. |
| External test ranges | `trainE7_1M.m` uses zero-based raw intervals [100000j, 100000j+100006), j=1..9, each with 100,000 targets. `trainE7_1Mplus.m` adds 1,000,000 to every offset, giving a further 900,000 e targets. The blocks have disjoint targets but adjacent raw ranges share six symbols. The plus script concerns e and its size differs from π T2. |
| Evaluation and selection | Both e scripts compute mean accuracy and sample variance across nine subgroup rates, use SE = SD/√9, and calculate two bounds with the multipliers 4.502 and 2.897. These approximate one-sided 0.001 and 0.01 t critical values at eight degrees of freedom. Each script restarts the fit while both lower bounds are below 0.5. The same external labels drive model selection and the reported bound. Restart counts, unsuccessful runs and a maximum restart budget are absent. A later calculation also reports mean ±5 SE. `Pfm_pp` compares the previous result array despite a second simulation. |

Script sources at the audited revision: [preprocessing](https://github.com/WANG-AXIS/pseudo-Randomness/blob/1ea617b5d3a5e7a89928e44926ff415bb448fe4e/load2Me.m), [1M training/evaluation](https://github.com/WANG-AXIS/pseudo-Randomness/blob/1ea617b5d3a5e7a89928e44926ff415bb448fe4e/trainE7_1M.m), [1Mplus training/evaluation](https://github.com/WANG-AXIS/pseudo-Randomness/blob/1ea617b5d3a5e7a89928e44926ff415bb448fe4e/trainE7_1Mplus.m).

## Saved-output check

Using SciPy 1.11.4 and NumPy 1.26.2, the audit replayed the stored weights on the stored data. All 18 subgroup accuracies and both aggregate training accuracies exactly matched the saved records. The first nine subgroup accuracies match the preprint's e row. No MATLAB training was rerun.

| Evidence | SHA-256 of upstream file |
|---|---|
| `e.txt` | `926ec9660258c36e9ebfecc7786286d6b6ded3c8a4befbe12db8c2cce4eb39d7` |
| `ESource.mat` | `9070f96d3890954f7fbda6953324ff6cbf9162ca724f0672ff36e89245b030d2` |
| `E7_1M_5sigma_record.mat` | `326515491c488a28b1a7f8eaf77d80000b5228277bda7be42ff454a971364c36` |
| `E7_1Mplus_5sigma_record.mat` | `5bd6d6bc4d7b14464175213f4fb8360024a41163b907618ac4b0ba022324597d` |

To reproduce the check from the upstream checkout with those audit dependencies:

```python
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.special import expit

bits = loadmat("ESource.mat")["ESource"].ravel()
digits = b"".join(Path("e.txt").read_bytes().split()[:33334])
np.testing.assert_array_equal(bits, np.frombuffer(digits, dtype=np.uint8) >= ord("5"))

for tag, offset in (("1M", 0), ("1Mplus", 1_000_000)):
    record = loadmat(f"E7_{tag}_5sigma_record.mat", squeeze_me=True)
    def correct(features, targets):
        x = 2 * features.astype(float) - 1
        a = expit(record["inputWeights"] @ x + record["inputbias"][:, None])
        b = expit(record["layerWeights"] @ a + record["layerbias"][:, None])
        scores = (record["outputWeights"] @ b + record["outputbias"] + 1) / 2
        return np.mean((scores >= 0.5) == targets)

    assert correct(record["P1"], record["T1"]) == record["Pfm"]
    accuracy = []
    for j in range(1, 10):
        start = offset + 100_000 * j
        windows = np.lib.stride_tricks.sliding_window_view(bits[start:start + 100_006], 7)
        accuracy.append(correct(windows[:, :6].T, windows[:, 6]))
    np.testing.assert_array_equal(accuracy, record["Stat_overall"])
    print(tag, record["Pfm"], accuracy)
```

## Python replication gaps

| Area | Sprint 01 Python |
|---|---|
| Historical starting point | Baseline commit `e4e8cf912770a63227751e2a26fb8535935c5772` contained one README, no benchmark code or tests. No prior implementation was located in fetched refs or the workspace search. The README described an intended scikit-learn MLP and 40,000/90,000 windows. Sprint 01 reconstructs the benchmark. |
| Data | Six context bits and decimal threshold match. Fractional digits come from guard-checked mpmath or a caller-supplied digit file. NumPy MT19937 uniform doubles are a separate implementation and seed mapping from MATLAB's. Smoke offsets and counts are deliberately small. |
| Fitting | Same layer widths and activations with MSE, plus Glorot initialization, full-batch Adam, raw 0/1 scale, no window shuffling, and exactly 40 steps. The linear score is clipped for proper scoring. MATLAB-specific processing, optimizer, and random division are unimplemented. |
| Evaluation | Fit once per declared seed, then score raw-disjoint validation and test blocks. Record raw blocks, source and block hashes, p1 by six-bit context, accuracy, log loss and Brier. Fair, train-majority and n-gram references use common targets. Smoke artifacts contain no significance tests. |

Exact historical training and the original π and MT results remain unverified. Resolving them requires the original dataset spans and their conventions, toolbox version, seeds and split indices, and the original run and selection history. The [statistical protocol](research-protocol-v1.md) specifies a separate finite-block confirmatory design.
