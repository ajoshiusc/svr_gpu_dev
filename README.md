# SVR GPU 31-case runner and visual report

This repository runs the public `svr_cli.py` from a neighboring `svr_gpu`
checkout for each case in a local JSON manifest. It saves one Nilearn
`plot_anat` screenshot per reconstruction and builds a standalone HTML report.

The manifest, scans, checkpoints, logs, and generated reconstructions are local
inputs or outputs. They are not stored in this repository. Keep the manifest
outside Git because it may contain private paths or study information.

## Layout

Place the repositories next to each other:

```text
Projects/
  svr_gpu/
  svr_gpu_dev/
```

Use an environment with nibabel, nilearn, matplotlib, and Pillow installed for
the report. Reconstruction runs with the Python environment in `svr_gpu` by
default (`svr_gpu/.venv/bin/python`), or with the interpreter passed through
`--python`.

## Run

The local manifest is a JSON list. Each case has `case` and `raw_stacks`
(or `stacks`) fields containing the case label and a list of NIfTI paths. An
optional `reference` NIfTI path adds a reference screenshot next to the new
reconstruction screenshot. The reference and reconstruction are displayed
independently in their own anatomical coordinates; this is a visual summary,
not a registered voxelwise comparison.

```bash
python run_cohort.py \
  --manifest /path/to/local/cohort_manifest.json \
  --svr-gpu-dir ../svr_gpu \
  --output results/run_YYYYMMDD_HHMMSS \
  --device 0
```

The process runs cases sequentially and records the exact CLI command, elapsed
time, log, return status, and output path in `results.csv`. A completed case is
skipped on resume if its output still exists; use `--force` to rerun it. The
HTML report is `OUTPUT/report/index.html` with per-case PNGs in the same folder.

Use `--limit 1` to check a local setup before starting the full cohort. The
runner does not download data or model weights.
