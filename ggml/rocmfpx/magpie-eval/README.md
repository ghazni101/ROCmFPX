# Magpie kernel evaluation - ROCmI4 W4A4 MMQ epilogue

Evaluates the W4A4 MMQ epilogue restructure with AMD-AGI/Magpie. The two
variants compile the real kernel headers from two checkouts of this repo:
`baseline` = commit 15acbdbbf (pre-restructure), `optimized` = the current
HEAD. See docs/rocmfpx/plans/rdna3-gfx1100-tg-pp-acceleration.md for the
results and methodology.

Layout:
- kernel.hip        shared standalone harness (correctness + perf modes)
- baseline/         run_test.sh for the pre-restructure tree
- optimized/        run_test.sh for the current tree
- compare.yaml      magpie compare config (paths are host-specific)

Run (inside the ROCm 10 container, Magpie cloned at /magpie):
    pip install -e /magpie
    magpie compare --kernel-config compare.yaml --output-dir results

The correctness gate is bit-exactness against the merged-baseline output
checksum (the restructure is a pure reordering). The nibble-to-k slot
order itself is covered by test-rocmi4-iu4-dot and test-backend-ops.
