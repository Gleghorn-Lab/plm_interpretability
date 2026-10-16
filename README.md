# Exhaustive hidden-state probing of protein language models

Code to reproduce every figure and table of *Exhaustive hidden-state probing of biological concepts in protein language
models* (Hallee, Rafailidis, Peleg and Gleghorn). The study trains a linear probe on every token-wise state of ESM
Cambrian models with 300 million, 600 million and 6 billion parameters (the token embedding, the queries, keys and
values before and after rotation, the attention context and output, the SwiGLU internals and the residual stream of
every block, and every attention head) on 22 protein concepts and two protein-protein interaction gold standards.

Everything the paper uses is in the public Gleghorn Lab collection
[Exhaustive hidden-state probing](https://huggingface.co/collections/GleghornLab/exhaustive-hidden-state-probing-6ad254122b677c479591339c):
the three models (FastPLMs implementations), the 22 concept datasets with their homology splits, the two Bernett
gold standards, and the results archive (`GleghornLab/esmc_tap_probing_results`) with every score table, training
record, selection, bootstrap interval and analysis table.

## Layout

| Path | What it holds |
|---|---|
| `protify/` | [Protify](https://github.com/Synthyra/Protify), pinned as a submodule at `13c37ae`, the commit whose `interpretability` package trained, scored and drew everything |
| `tap_probing/reproduce.py` | downloads the results archive and redraws every figure and table with Protify's own drawing code |
| `tap_probing/archive.py` | restores Protify's run layout from the archive and reads its compacted score tables |
| `tap_probing/cross_model.py` | the Supplement's figures and tables that place the three models on one relative depth axis |
| `tap_probing/solver_table.py` | the Supplement's converged-solver table |
| `docs/runs.md` | the Protify commands that produced every run, to retrain from scratch |
| `scripts/publish_collection.py` | how the datasets and results were published to the collection |

## Reproduce the figures and tables

No GPU is needed: the archive holds every probe's scores, and the figures are drawn from them.

```bash
git clone --recurse-submodules https://github.com/Gleghorn-Lab/tap_probing
cd tap_probing
pip install -r protify/requirements.txt -r requirements.txt
PYTHONPATH=protify:. python -m tap_probing.reproduce --out reproduced
```

The script downloads about 300 MB of tables from the archive at its pinned revision, checks that each model's rebuilt
probe summary matches the one written when the paper's figures were drawn, and writes:

| Folder | Paper items |
|---|---|
| `reproduced/trends` | Figures 2 to 6 (emergence, state rankings, controls, model size, protein-protein interaction) and Figure S14 |
| `reproduced/cross_model` | Figures S1 to S3, Tables S1 to S4 |
| `reproduced/main` | ESMC-300 figures of the Supplement (state maps, heads, rotation, random-init comparison, controls, ProtParam targets, settings sweep, phosphomimetic scanning) |
| `reproduced/raw` | the same on raw token states, and the principal-component views |
| `reproduced/compare` | raw against unit-norm states and mean-only pooling |
| `reproduced/esmc600`, `reproduced/esmc6b` | per-model tables of the larger models |

Every table the paper's numbers come from is written beside its figure as CSV.

## Retrain from scratch

`docs/runs.md` lists the Protify commands for every run: pooled stores, settings sweeps and selections, protein and
residue probes, controls, the randomly initialized model, the baseline encodings, the larger models, the pair tasks and
their bootstrap intervals, principal components and phosphomimetic scans. They ran in Docker on one NVIDIA GH200;
probe weights and pooled stores are not archived, since every probe retrains from the pinned models and datasets.

## Citation

The concept datasets, splits and control design are those of Whitfield et al., *High-resolution dissection of concept
acquisition in different families of protein language models* (bioRxiv, 2026), and the protein-protein interaction
gold standards those of Bernett et al. (arXiv:2609.10193, 2026; Briefings in Bioinformatics 25:bbae076, 2024). Please
cite them alongside this work.
