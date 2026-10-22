# Exhaustive hidden-state probing of protein language models

Code to reproduce every figure and table of *Exhaustive hidden-state probing of biological concepts in protein language
models* (Hallee, Rafailidis, Peleg and Gleghorn). The study trains a linear probe on every token-wise state of ESM
Cambrian models with 300 million, 600 million and 6 billion parameters (the token embedding, the queries, keys and
values before and after rotation, the attention context and output, the SwiGLU internals and the residual stream of
every block, and every attention head) on 22 protein concepts and two protein-protein interaction gold standards,
compares linear with one-hidden-layer MLP probes at three seeds of every model, and asks on ESMC-600, through lenses,
write ablations and concept erasure, whether the concepts the probes read are used for the model's own objective.

Everything the paper uses is in the public Gleghorn Lab collection
[Exhaustive hidden-state probing](https://huggingface.co/collections/GleghornLab/exhaustive-hidden-state-probing-6ad254122b677c479591339c):
the three models (FastPLMs implementations), the 22 concept datasets with their homology splits, the two Bernett
gold standards, and the results archive (`GleghornLab/esmc_tap_probing_results`) with every score table, training
record, selection, bootstrap interval and analysis table. The three-seed comparison of linear and MLP probes under
both poolings has an archive of its own (`GleghornLab/esmc_tap_probing_pooling_results`), which also holds the
survey's repetition on the pooled mean, ESMC-600's probe variants and state geometry, and its lenses, ablations and
erasure.

## Layout

| Path | What it holds |
|---|---|
| `protify/` | [Protify](https://github.com/Synthyra/Protify), pinned as a submodule at `ca7bbd2` on its `tap-probing` branch, whose `interpretability` package trained, scored and drew everything (the commit of each run is in `docs/runs.md`) |
| `tap_probing/reproduce.py` | downloads the results archive and redraws every figure and table with Protify's own drawing code |
| `tap_probing/archive.py` | restores Protify's run layout from the archive and reads its compacted score tables |
| `tap_probing/cross_model.py` | the Supplement's figures and tables that place the three models on one relative depth axis |
| `tap_probing/solver_table.py` | the Supplement's converged-solver table |
| `tap_probing/pooling_study.py` | the three-seed comparison of pooling and probe kinds: gains with intervals, tables |
| `tap_probing/probe_gap.py` | what a nonlinear readout adds: the decomposition of the MLP's lead and ESMC-600's state geometry |
| `tap_probing/causal_figures.py` | ESMC-600's lenses, write ablations and concept erasure |
| `docs/runs.md` | the Protify commands that produced every run, to retrain from scratch |
| `scripts/publish_collection.py` | how the datasets and results were published to the collection |

## Reproduce the figures and tables

No GPU is needed: the archive holds every probe's scores, and the figures are drawn from them.

```bash
git clone --recurse-submodules https://github.com/Gleghorn-Lab/tap_probing
cd tap_probing
pip install -r protify/requirements.txt -r requirements.txt
PYTHONPATH=protify:. python -P -m tap_probing.reproduce --out reproduced
```

`-P` (Python 3.11 or later) keeps the clone's own folder off the front of the import path, where its `foundry` package
would hide Protify's. The script downloads both archives at their pinned revisions (about 300 MB and 13,000 small tables; at the Hub's rate
limit it pauses and resumes, and a Hugging Face token in `HF_TOKEN` raises that limit; on Windows, pass a short `--out`
folder, since the archives' paths pass the 260-character limit under a long one), checks that each model's rebuilt probe summary matches the one written when the paper's
figures were drawn, and writes:

| Folder | Paper items |
|---|---|
| `reproduced/trends` | Figures 2, 8, 10 and 11 (emergence, controls, model size, the interaction gold standard) and S14 (the newer gold standard) |
| `reproduced/familiarity` | Figures 9 and S13: probe correctness against each model's pseudo log likelihood of the protein |
| `reproduced/cross_model` | Figures 3 and 4 (kinds of state by kinds of concept, the structure stream), S1 to S3, Tables S1 to S4 |
| `reproduced/probe_gap` | Figures 5 (what a nonlinear readout adds), S8 to S10 |
| `reproduced/causal` | Figures 6 and 7 (lenses, write ablations, erasure), S11 and S12 |
| `reproduced/pooling` | Figures S4 to S7 (MLP probes) and S25 (pooling), Tables S5 and S6: pooling and MLP probes, three seeds of every model |
| `reproduced/main` | ESMC-300 figures of the Supplement: controls, random-init comparison, state maps, heads, rotation and ProtParam targets (Figures S15 to S22), the settings sweep (S27) and phosphomimetic scanning (S29) |
| `reproduced/trends_mean`, `reproduced/cross_model_mean` | Figures S23 and S24: the survey on the pooled mean |
| `reproduced/compare` | raw against unit-norm states (Figure S26) |
| `reproduced/raw` | the same on raw token states, and the principal-component views (Figure S28) |
| `reproduced/esmc600`, `reproduced/esmc6b` | per-model tables of the larger models |

On Windows, either enable long paths or keep `--out` short: the Hub client's cache paths under the pooling archive
approach 260 characters.

Every table the paper's numbers come from is written beside its figure as CSV.

## Retrain from scratch

`docs/runs.md` lists the Protify commands for every run: pooled stores, settings sweeps and selections, protein and
residue probes, controls, the randomly initialized model, the baseline encodings, the larger models, the pair tasks and
their bootstrap intervals, principal components, phosphomimetic scans, the pooling and MLP study, ESMC-600's probe
variants and geometry, its lenses, ablations and erasure, and the familiarity analysis. They ran in Docker on NVIDIA
GH200s, the familiarity analysis on NVIDIA H100s on Modal; probe weights and pooled stores are not archived, since every probe retrains from the pinned models and datasets.

## Citation

The concept datasets, splits and control design are those of Whitfield et al., *High-resolution dissection of concept
acquisition in different families of protein language models* (bioRxiv, 2026), and the protein-protein interaction
gold standards those of Bernett et al. (arXiv:2609.10193, 2026; Briefings in Bioinformatics 25:bbae076, 2024). Please
cite them alongside this work.
