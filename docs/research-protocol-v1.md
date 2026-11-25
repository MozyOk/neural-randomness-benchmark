# Statistical research protocol v1

Status: candidate for independent review. The committed smoke run is exploratory. Confirmatory execution requires a signed-off protocol and a frozen analysis manifest before the held-out results are opened.

## Question and estimands

The primary question is whether a frozen six-context, original-style MLP predicts thresholded fractional digits of π above 50% in two new raw blocks. The estimand is next-bit accuracy minus 0.5, in percentage points, reported separately for each block. For each, the sharp null is that the next bit is Bernoulli(0.5) conditional on all earlier bits, the frozen learner and the initial context. The directional alternative is higher predictive accuracy. The primary conclusion requires replication in both blocks. An IID reference model for finite deterministic digits has a narrower scope than mathematical normality.

Two secondary questions are specified for the confirmatory family:

1. **Marginal bias and baseline skill.** In π block B, does the MLP improve paired accuracy over the train-only majority or the train-only six-context n-gram? Each comparison has its own null of nonpositive average conditional improvement and alternative of positive improvement. Compute the difference in per-target correctness on the same targets.
2. **Other constants.** On block B of e and √2, does the same frozen procedure achieve accuracy above 0.5? Apply the primary sharp null separately to each source with its own training data. Report the corresponding effect sizes and all outcomes.

Report log loss in nats and Brier score per target for the MLP, majority, n-gram and fixed probability 0.5. Report differences with a positive sign for reductions in loss. These probability scores and individual model-seed results are descriptive. At probability 0.5 the hard-call rule predicts 1, so the fixed forecast's scored accuracy equals the held-out fraction of ones. The primary accuracy reference is the theoretical 0.5.

## Data and blocks

Use fractional decimal digits from byte-addressable ASCII digit streams, stripped of ASCII whitespace, with digit ≥5 mapped to 1. Record both the original-file SHA-256 and the canonical digit and bit-block hashes. Verify digits covering all declared ranges against an independently generated or independently sourced stream. Record both provenances and versions with the manifest. Smoke fixtures use mpmath and the offsets in the smoke script.

Each example uses six past bits and the seventh as target. All ranges are zero-based, half-open raw offsets. The first six bits of each block are context only. Predictor weights, hyperparameters and thresholds remain frozen across both held-out blocks.

| Split | Raw range | Targets | Role |
|---|---:|---:|---|
| Training | [0, 40,006) | 40,000 | Fit model and baselines |
| Validation | [50,000, 60,006) | 10,000 | Development diagnostics before freezing |
| Held-out A | [11,000,000, 14,000,006) | 3,000,000 | Primary test |
| Held-out B | [15,000,000, 18,000,006) | 3,000,000 | Primary replication test and secondary comparisons |

A and B are disjoint from the earlier experiments in the preprint and the Sprint 01 smoke run. They share a predictor but share no raw bits. Disjointness alone gives no statistical independence guarantee under an alternative. Use the same training and validation offsets for each constant and a distinct fit. For e and √2 only block B is evaluated. Commit source checksums, offset manifest, exact dependencies, commands, test family and software SHA before confirmatory execution. Keep held-out scoring and artifacts under an independent operator until the freeze. Preserve the full prediction table before unblinding. Record any prior inspection of the declared blocks. A compromised block requires a versioned amendment and a fresh declared range before scoring.

The learner is the arithmetic mean of the clipped probability tables from the three frozen model seeds 101, 202 and 303, thresholded at 0.5 with ties predicting 1. All three seeds train for 40 full-batch Adam steps with the implementation's fixed hyperparameters. Identical model and metric definitions apply across constants. Training repeats change initialization and therefore do not increase the number of held-out targets. Use per-seed and per-block tables as diagnostics, with the ensemble as the single declared predictor. Compute MAJ and n-gram tables using the same train block and one Laplace pseudocount per class.

## Controls

- Positive control uses the repeated pattern `0011`, raw training and validation offsets above, and a 100,000-target test block starting at raw index 70,000. The MLP ensemble and six-context n-gram must each score at least 0.99 accuracy. This checks basic learnability, alignment and the comparison pipeline.
- Negative controls use 20 PCG64 streams with data seeds 4101–4120, uniform double draws thresholded at 0.5, and the same offsets, budget and model seeds as the periodic control. For each stream, test the frozen ensemble's accuracy against 0.5 with a two-sided exact binomial check at 0.01/20. Any flagged stream invalidates the preflight until independently investigated. Preserve all outcomes. A clean control suite checks for gross miscalibration or leakage under these synthetic inputs.
- Keep PCG64 data seeds, NumPy MT19937 data seeds and MLP initialization seeds separate in artifacts. MT evaluation in Sprint 01 is exploratory. A later MT confirmatory design needs new independent generator streams, declared output conversion, a fixed target budget per stream and a stream-level analysis. The original repeated-fit experiment scored one target per stream.

## Inference, uncertainty and power

Let `C_t` denote the indicator that the frozen causal call at target `t` was correct. Under the sharp primary null, conditional on earlier bits and the warm-up, `Pr(C_t=1 | past)=0.5`. Therefore the count has an exact binomial null even with sliding contexts. Future data, test-trained parameters or selection on the test invalidate this argument. Test the upper tail for A and B at 0.05 each. The single intersection-union primary decision requires both to pass, which controls size at 0.05 whenever either block satisfies its null. Report both individual p-values and their maximum as the joint p-value.

For the paired secondary contrasts, let `D_t=C_t(MLP)-C_t(baseline)` be in [-1,1]. Under the null that the time-average of `E[D_t | past]` is nonpositive almost surely, use the conservative one-sided bounded martingale p-value `exp(-n max(mean(D),0)^2 / 2)`. Use exact binomial tails for the two other-constant sharp nulls. Test the four secondary hypotheses with Holm at family-wise 0.05 only after primary success. Report all four raw and adjusted p-values regardless of sign when the gate opens. If the gate stays closed, report estimates and intervals and label the secondary results descriptive.

Report simultaneous 95% two-sided confidence intervals for the six declared accuracy effects using a fixed-horizon martingale bound. For score range width `w` and `n` targets, report the observed mean ± `w sqrt(log(2 × 6 / 0.05)/(2n))`, intersected with the attainable effect range. Use `w=1` for accuracy minus 0.5 and `w=2` for paired differences. The interval targets the time-average conditional expected score under the assumed sequence model and does not require independent overlapping windows. Always include numerator, denominator and raw offsets. In descriptive outputs include loss scores and counts without adding inferential tests.

The primary design targets a 0.10 percentage-point increase (accuracy 0.501). Require an observed lift of at least 0.05 percentage points in both blocks in addition to statistical rejection. With 3,000,000 IID Bernoulli correctness indicators, the exact 0.05 critical count is 1,501,425. The effect floor raises the joint decision's per-block cutoff to 1,501,500. Using `Pr[Binom(3_000_000, p) >= 1_501_500]`, power is 0.958 per block at `p=0.501` and 0.919 for both under independent blocks with that same accuracy. At `p=0.50025`, joint power is about 0.037 under the same approximation. These calculations condition on a predictor achieving the assumed accuracy. They do not account for finite training or correlated success indicators under alternatives.

Before unblinding, validate the inferential implementation against exact counts, a synthetic fair null and lag-dependent synthetic streams with an oracle and the frozen learner. Record test size and power estimates with Monte Carlo uncertainty, including dependence stress cases and the planned stopping rule. Treat the secondary family as lower-priority with no power guarantee. Report deviations and the detectability limit with null or inconclusive outcomes.

## Stopping and replication criteria

Run one fixed-budget analysis after controls pass and the operator and reviewer agree the manifest is frozen. Score all prescribed targets once. Prohibit interim looks, optional extension, test-driven restarts, threshold or seed changes, and replacing unfavorable blocks. For an operational failure, retain the failure record and rerun only with the same frozen inputs. Any substantive amendment requires a new protocol version and fresh unexamined blocks. Preserve individual runs, including failures and null outcomes.

Directional replication of the finite-block π result under this revised design requires both primary rejections, both observed effects above the stated floor or equal to it, passing controls, verified lineage and independent reproduction of the scoring. Evidence of MLP value beyond a baseline requires that baseline's gated, adjusted paired contrast to pass. Historical reproduction separately requires matching the authors' original data, MATLAB environment, fit and selection procedure. Replaying the saved e model is a narrower historical check and carries the original selection limitations.

Sprint 01 smoke runs, alternate offsets, seed-specific maxima, additional architectures, alternate thresholds and any analyses after unblinding are exploratory. Exploratory outputs can guide a later preregistration. The scope of either design is prediction on the specified representation and finite blocks.
