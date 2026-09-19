#!/usr/bin/env bash
# Correctness gate for the Magpie compare run: the kernel binary itself
# exits nonzero if the fp64 reference check fails.
set -euo pipefail
cd "$(dirname "$0")"
./kernel --check
