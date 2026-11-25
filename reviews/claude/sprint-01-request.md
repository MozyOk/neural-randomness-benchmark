# Independent review request for Claude

- **Candidate commit** `a6c2803ef85753316304801f2b23461d4ae7e18f`
- **Base commit** `e4e8cf912770a63227751e2a26fb8535935c5772`

Review the candidate commit. The follow-up commit pins this request to the candidate. Working materials at the candidate:

- Research protocol — [docs/research-protocol-v1.md](../../docs/research-protocol-v1.md)
- Implementation and environment — [benchmark.py](../../benchmark.py), [.python-version](../../.python-version), [requirements.txt](../../requirements.txt) and [CI workflow](../../.github/workflows/benchmarks.yml)
- Tests and replay command — [tests](../../tests), [scripts/smoke.sh](../../scripts/smoke.sh) and [README](../../README.md)
- Raw predictions and input blocks — [artifacts/sprint-01-smoke.json](../../artifacts/sprint-01-smoke.json)
- Primary sources — [arXiv v1](https://arxiv.org/pdf/1801.01117v1), [IEEE version](https://doi.org/10.1109/ACCESS.2018.2826448), and [MATLAB repository at the audited revision](https://github.com/WANG-AXIS/pseudo-Randomness/tree/1ea617b5d3a5e7a89928e44926ff415bb448fe4e)
- Issue scope — [historical replication #1](https://github.com/MozyOk/neural-randomness-benchmark/issues/1) and [statistical protocol #8](https://github.com/MozyOk/neural-randomness-benchmark/issues/8)

Use a fresh checkout and CPython 3.12.16.

```sh
git checkout --detach a6c2803ef85753316304801f2b23461d4ae7e18f
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -m unittest discover -s tests -v
sh scripts/smoke.sh --output /tmp/sprint-01-review-smoke.json
```

## Standalone review prompt

> Act as the independent scientific and code reviewer for MozyOk/neural-randomness-benchmark Sprint 01. Review commit `a6c2803ef85753316304801f2b23461d4ae7e18f` against base `e4e8cf912770a63227751e2a26fb8535935c5772`. Use the preprint at https://arxiv.org/pdf/1801.01117v1 and the MATLAB repository https://github.com/WANG-AXIS/pseudo-Randomness at `1ea617b5d3a5e7a89928e44926ff415bb448fe4e`. Derive your own account of preprocessing, training, model selection and evaluation before opening the implementer's gap analysis. Compare your account with `docs/original-replication-gap.md` after recording your initial findings.
>
> Audit `docs/research-protocol-v1.md` for well-defined hypotheses and estimands, split integrity, seed and block independence, controls, validity of p-values and intervals, multiplicity, power, stopping and replication criteria. Check the code for offset and window errors, future or held-out information access, metric correctness, baseline comparability and source provenance. Identify what the raw artifacts can substantiate. Attempt to independently reconstruct a metric from the saved raw bits and context scores.
>
> Use CPython 3.12.16 and run `python -m pip install -r requirements.txt`, `python -m unittest discover -s tests -v` and `sh scripts/smoke.sh --output /tmp/sprint-01-review-smoke.json` in a virtual environment with `OPENBLAS_NUM_THREADS=1` and `VECLIB_MAXIMUM_THREADS=1`. Compare the rerun to `artifacts/sprint-01-smoke.json`. Treat smoke outputs as exploratory. Keep the proposed confirmatory blocks unscored.
>
> Return (1) a verdict of accept, revise or block for the foundation and a separate verdict on protocol readiness, (2) severity-ordered findings with file and line or primary-source evidence, (3) commands and observations from any independent checks, (4) unverified assumptions or access limits, and (5) minimum fixes and acceptance checks. Keep the review read-only and return it privately.
