# Agent guide

The reproduction repository of *Exhaustive hidden-state probing of biological concepts in protein language models*.
`README.md` maps every module to the paper's figures and tables; this file holds what a change here must keep.

## Two import paths

Protify and the neuron study both import packages named `foundry` and `fastplms`, at different versions, so each side
runs on its own path.

| Code | Path |
|---|---|
| The survey's figures and tables (`tap_probing/*.py`) | `PYTHONPATH=protify` |
| The neuron study (`tap_probing/units/`, `scripts/units_atlas.py`) | `PYTHONPATH=.:vendored:vendored/annotation_vocabulary/src` |
| The tests, which import both | `PYTHONPATH=vendored:vendored/annotation_vocabulary/src:protify`, vendored copies first |

A module of the survey never imports from `tap_probing.units`, and the neuron code never imports Protify.

## Invariants

- Results are pinned. `tap_probing/archive.py` (`REVISION`, `POOLING_REVISION`) and `tap_probing/published.py`
  (`SUMMARY_REVISION`) name the archive revisions, and `protify/` is a submodule at `ca7bbd2`. Move a pin only with a
  redraw that reproduces the paper's figures.
- `tap_probing.reproduce` checks each model's rebuilt probe summary against the archive's before it draws. Keep the
  check.
- Every name of a probed state in a figure or table comes from `tap_probing/figure_style.py`, and `figure_style.install`
  refuses an old name when a figure is saved.
- The neuron study follows `preregistration.yaml` of the third archive. Its thresholds, seeds and decision rules do not
  change after the fact; a new analysis is added as its own stage of `scripts/units_atlas.py`.
- `vendored/` holds copies of lab packages. Do not edit them here.
- No probe weight, pooled store or other large or generated file enters the repository. Results go to the archives.

## Commands

```bash
PYTHONPATH=vendored:vendored/annotation_vocabulary/src:protify python -m pytest tests -q
PYTHONPATH=protify python -m tap_probing.reproduce --out reproduced
PYTHONPATH=protify python -m tap_probing.published --out published
PYTHONPATH=.:vendored:vendored/annotation_vocabulary/src python scripts/units_atlas.py --help
```

## Code

- A module docstring first, then `from __future__ import annotations`, direct imports, from-imports, and the
  repository's own imports last.
- Typed functions with one narrow job, domain names, and a shape comment on every array or tensor binding.
- Assertions for invariants and no silent fallback. A test is never skipped or marked as an expected failure.
- Figures are PNG at 300 dpi with lettered panels and no title inside the image.
