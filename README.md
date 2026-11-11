# Exhaustive hidden-state probing of protein language models

Code to reproduce the figures and tables of *Exhaustive hidden-state probing of biological concepts in protein language
models* (Hallee and Gleghorn). The study trains a linear probe on every token-wise state of ESM
Cambrian models with 300 million, 600 million and 6 billion parameters (the token embedding, the queries, keys and
values before and after rotation, the attention context and output, the SwiGLU internals and the residual stream of
every block, and every attention head) on 22 protein concepts and two protein-protein interaction gold standards. It
compares linear with one-hidden-layer MLP probes at three seeds of every model, asks on ESMC-600, through lenses,
write ablations and concept erasure, whether the concepts the probes read are used for the model's own objective, and
reads the feed-forward neurons one at a time, as poles, against random rotations, the residual stream and sparse
autoencoders.

Everything the paper uses is in the Gleghorn Lab collection
[Exhaustive hidden-state probing](https://huggingface.co/collections/GleghornLab/exhaustive-hidden-state-probing-6ad254122b677c479591339c):
the three models (FastPLMs implementations), the 22 concept datasets with their homology splits, the two Bernett
gold standards, and three result archives.

| Archive | What it holds |
|---|---|
| `GleghornLab/esmc_tap_probing_results` | the survey: every score table, training record, selection, bootstrap interval and analysis table, and the tuned pair probes |
| `GleghornLab/esmc_tap_probing_pooling_results` | the three-seed comparison of linear and MLP probes under both poolings, the survey's repetition on the pooled mean, ESMC-600's probe variants and state geometry, its lenses, ablations and erasure, and the familiarity analysis |
| `GleghornLab/esmc_native_units_results` | the neuron study: its preregistration, atlas sample, per-unit scores, decisions and evaluation tables |

Probe weights, pooled stores and the neuron study's activation maxima are not archived: every probe retrains, and every
pass reruns, from the pinned models and datasets.

## Layout

| Path | What it holds |
|---|---|
| `protify/` | [Protify](https://github.com/Synthyra/Protify), pinned as a submodule at `ca7bbd2` on its `tap-probing` branch, whose `interpretability` package trained, scored and drew the survey (the commit of each run is in `docs/runs.md`) |
| `tap_probing/reproduce.py` | downloads the first two archives and redraws the survey's figures and tables with Protify's own drawing code |
| `tap_probing/archive.py` | restores Protify's run layout from the archives and reads their compacted score tables |
| `tap_probing/figure_style.py` | the paper's names for probed states in figures and tables, and the marks its interval panels share |
| `tap_probing/cross_model.py` | the figures and tables that place the three models on one relative depth axis, and the interaction figures' layout |
| `tap_probing/solver_table.py` | the converged-solver table |
| `tap_probing/published.py` | the tuned pair probes beside published predictors on the 2024 gold standard |
| `tap_probing/pooling_study.py` | the three-seed comparison of pooling and probe kinds: gains with intervals, tables |
| `tap_probing/probe_gap.py` | what a nonlinear readout adds: the decomposition of the MLP's lead and ESMC-600's state geometry |
| `tap_probing/causal_figures.py` | ESMC-600's lenses, write ablations and concept erasure |
| `tap_probing/familiarity.py` | probe correctness against each model's familiarity with a protein |
| `tap_probing/units/` | the neuron study: poles and bases, the passes, the protein-level atlas and its decisions, InterPLM's residue-level evaluation on ESM-2 and ESMC, the dictionary test and sparsest-rotation search, ablations, and its figures and tables |
| `scripts/units_atlas.py` | one command per stage of the neuron study, from its inputs to its figures |
| `scripts/units_queue.sh`, `scripts/in_container.sh`, `scripts/units_hub.py`, `Dockerfile` | the neuron study's queue, container entry, uploads and image as they ran through the lab's launcher; a clone runs each stage directly |
| `scripts/units_release.py` | how the third archive was cut from the neuron study's run, and what it leaves out |
| `scripts/publish_collection.py` | how the datasets and the first archive were filed in the collection |
| `docs/runs.md` | the Protify commands that produced every run of the survey, to retrain from scratch |
| `vendored/` | copies of the lab packages the neuron study imports, at the versions it ran with: `foundry` (shared library code), `fastplms` (the ESM++ model and the SAE loaders), `annotation_vocabulary` (association statistics and label views) and `annotation_transformer`, which `foundry.embedding` imports |
| `tests/` | unit tests on synthetic tensors and a synthetic archive |

## Reproduce the survey's figures and tables

No GPU is needed: the archives hold every probe's scores, and the figures are drawn from them.

```bash
git clone https://github.com/Gleghorn-Lab/plm_interpretability
cd plm_interpretability
git submodule update --init protify
pip install -r protify/requirements.txt -r requirements.txt
PYTHONPATH=protify python -m tap_probing.reproduce --out reproduced
```

Fetch the Protify submodule alone, as above: a recursive clone also descends into Protify's own submodules, which
nothing here uses and one of which is not public. Only Protify goes on the import path here: it carries its own copies of the lab packages, and `vendored/` stays off
the path so that the drawing code runs on the versions it was pinned with. The script downloads both archives' tables at their pinned revisions (several thousand small
files; at the Hub's rate limit it pauses and resumes, and a Hugging Face token in `HF_TOKEN` raises that limit),
checks that each model's rebuilt probe summary matches the one written when the paper's figures were drawn, and
writes:

| Folder | Paper items |
|---|---|
| `reproduced/trends` | Figures 2 (emergence), 7 (controls), 9 (model size) and 13 (interaction on the 2024 gold standard), and S12 (the 2026 gold standard) |
| `reproduced/cross_model` | Figures 3 (kinds of state by kinds of concept) and 10 (the structure stream), S1 and S2, Table 2 (the largest gains inside the block) and Table S1 (converged solvers) |
| `reproduced/probe_gap` | Figure 4 (what a nonlinear readout adds), S4 and S5 |
| `reproduced/causal` | Figures 5 and 6 (lenses, write ablations, erasure) and S6 |
| `reproduced/familiarity` | Figures 8 and S7: probe correctness against each model's pseudo pseudo log likelihood of the protein |
| `reproduced/published` | Figure 14: the tuned pair probes beside published predictors, with every plotted value and its source |
| `reproduced/pooling` | Figure S3 (linear and MLP probes along ESMC-6B's stream) and Table S2 (ranking metrics) |
| `reproduced/trends_mean`, `reproduced/cross_model_mean` | Figure S11: the survey on the pooled mean |
| `reproduced/raw`, `reproduced/main` | Figures S15 (principal components) and S16 (phosphomimetic scanning), with ESMC-300's state maps, heads, controls and settings sweep, which the paper leaves out |
| `reproduced/compare`, `reproduced/esmc600`, `reproduced/esmc6b` | raw against unit-norm states, and per-model tables of the larger models, with each task's best blocks |

On Windows, either enable long paths or keep `--out` short (a folder at the drive's root): the Hub client's cache
paths under the archives pass 260 characters otherwise.

Every table the paper's numbers come from is written beside its figure as CSV. Figures 1, 11 and 15 and Tables 1 and
3 to 5 are written in the manuscript. Figure 14 alone redraws in seconds, from one table of the first archive:
`PYTHONPATH=protify python -m tap_probing.published --out published`. Figures S13 and S14 belong to the study that labeled sparse autoencoder
latents with an annotation vocabulary.

## Reproduce the neuron study's figures and tables

The third archive holds every table the neuron figures read, so they redraw on a CPU:

```bash
hf download GleghornLab/esmc_native_units_results --repo-type dataset --local-dir units
pip install -r requirements.txt
export PYTHONPATH=.:vendored:vendored/annotation_vocabulary/src
python scripts/units_atlas.py figures --root units --preregistration units/preregistration.yaml
python scripts/units_atlas.py tables --root units --preregistration units/preregistration.yaml
```

`units/figures/figure_residue.png` is Figure 12, `figure_2.png`, `figure_5.png` and `figure_6.png` are Figures S8 to
S10, and `units/tables/decisions.tex` is Table S3. To run the study itself, each stage of `scripts/units_atlas.py`
takes the same `--root` and `--preregistration`; the passes ran on one NVIDIA GH200 (96 GB) and write about a terabyte of
activation maxima, and `scripts/units_queue.sh` holds the order they ran in. The neuron code needs no Protify: it
imports the packages under `vendored/`. Its inputs are the models and datasets the preregistration pins; one of them,
the annotation compendium `Synthyra/annotation_compendium_v1`, is not public, so the stages that read its labels
need access to it.

## Tests

The tests run on a CPU in under a minute. They import both Protify and the vendored packages, with the vendored
copies first:

```bash
PYTHONPATH=vendored:vendored/annotation_vocabulary/src:protify python -m pytest tests -q
```

## Retrain the survey from scratch

`docs/runs.md` lists the Protify commands for every run: pooled stores, settings sweeps and selections, protein and
residue probes, controls, the randomly initialized model, the baseline encodings, the larger models, the pair tasks and
their bootstrap intervals, principal components, phosphomimetic scans, the pooling and MLP study, ESMC-600's probe
variants and geometry, its lenses, ablations and erasure, and the familiarity analysis. They ran in Docker on NVIDIA
GH200s, the familiarity analysis on NVIDIA H100s; probe weights and pooled stores are not archived, since every probe
retrains from the pinned models and datasets.

## Citation

The concept datasets, splits and control design are those of Whitfield et al., *High-resolution dissection of concept
acquisition in different families of protein language models* (bioRxiv, 2026), and the protein-protein interaction
gold standards those of Bernett et al. (Briefings in Bioinformatics 25:bbae076, 2024; arXiv:2609.10193, 2026). The
neuron study's residue-level evaluation is InterPLM's (Simon and Zou, 2025), run with its code. Please cite them
alongside this work.
