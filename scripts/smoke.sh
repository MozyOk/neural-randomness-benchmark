#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
python benchmark.py \
  --sources iid periodic mt pi e sqrt2 \
  --train 512 --train-offset 0 \
  --validation 128 --validation-offset 600 \
  --test 256 --test-offset 800 \
  --model-seeds 101 202 303 --data-seeds 1101 2202 3303 "$@"
