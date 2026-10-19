# The runs

Every run used the `interpretability` package of Protify's `tap-probing` branch (`protify/`, pinned at `895a96c`, run
as `python -m src.protify.interpretability` from the submodule): the survey at `13c37ae`, the later runs at the commits
their sections name, in Docker with PyTorch and CUDA 12.6 on one NVIDIA GH200
(96 GB). Below, `P` stands for that command. Each analysis lives in a *root* folder; a root holds its pooled stores
(`stores/`), probe results (`results/<label>/<task>/`), sweeps, interventions and principal coordinates. The results
archive keeps every root's results under `runs/<root>/`. The concept tasks read
`GleghornLab/protein_concept_probing`; the pair tasks read `Synthyra/bernett_gold_ppi_2` (`ppi2`) and
`Synthyra/bernett_gold_ppi` (`ppi`).

The settings below are the ones the paper reports (Table 3 of the paper). On the GH200 the jobs ran as chains that
shared the GPU and waited for free memory; the chains and their timings are in the study's launch record, and none of
them changes what a command computes.

## Roots

| Root | Model | Treatment | Seed |
|---|---|---|---|
| `run`, `run_s24896`, `run_s24897` | ESMC-300 | raw token states | 24895, 24896, 24897 |
| `run_norm`, `run_norm_s24896`, `run_norm_s24897` | ESMC-300 | unit-norm token states (main text) | 24895, 24896, 24897 |
| `run_random*`, `run_random_norm*` | randomly initialized ESMC-300 | raw, unit-norm | as above |
| `run_600_norm` | ESMC-600 | unit-norm | 24895 |
| `run_6b_norm` | ESMC-6B | unit-norm | 24895 |
| `run_onehot`, `run_randvec` | `OneHot-Protein`, `Random` baseline encodings | | 24895 |

Seed replicates add `--seed <seed>` to every command; unit-norm roots add `--normalize`; the randomly initialized model
adds `--random-init --no-controls`; the baselines add `--model <name> --no-controls`.

## Settings sweeps and selection (ESMC-300, `run`, raw states)

Every sweep trains the probes of five sites under 14 settings at once, scored on validation only:

```bash
SWEEP="--sites embedding,resid_post,final_norm,attn_context,mlp_act --lr 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 --wd 0 0.1"
RESIDUE="lipidation,topology,phosphorylation,secondary_structure,ssp_q8"
P sweep --root run --label p_b256 --tasks protein --batch-size 256 --max-epochs 100 --patience 10 $SWEEP
P sweep --root run --label p_b1024 --tasks protein --batch-size 1024 --max-epochs 100 --patience 10 $SWEEP
P sweep --root run --label r_t32768 --tasks $RESIDUE --train-tokens 32768 --max-epochs 5 --patience 3 $SWEEP
P sweep --root run --label r_t131072 --tasks $RESIDUE --train-tokens 131072 --max-epochs 5 --patience 3 $SWEEP
P select --root run --labels p_b256 p_b1024 --loss classification   # lr 3e-2, wd 0, 256 proteins
P select --root run --labels p_b256 p_b1024 --loss regression       # ProtParam: lr 1e-3, wd 0.1, 1,024 proteins
P select --root run --labels r_t32768 r_t131072                     # lr 3e-3, wd 0.1, 32,768 tokens
```

The pair tasks were swept the same way on their own stores (`--label ppi_b1024 --tasks ppi` in `run`,
`--label ppi2_b1024 --tasks ppi2 --normalize` in `run_norm`, 1,024 pairs per step) and selected with
`--loss classification`.

## Every root

```bash
P embed --root <root> --tasks protein,ppi2,ppi --controls --max-tokens 131072
P train --root <root> --tasks protein --label linear --batch-size 256 --lr 0.03 --wd 0 --max-epochs 100 --patience 10
P train --root <root> --tasks prot_param --label linear --batch-size 1024 --lr 1e-3 --wd 0.1 --max-epochs 100 --patience 10
P train --root <root> --tasks residue --label linear --train-tokens 32768 --lr 3e-3 --wd 0.1 --max-epochs 10 --patience 3
P train --root <root> --tasks protein --label linear_mean --pooling mean   # mean-only pooling, same settings
P train --root <root> --tasks ppi2 ...                                     # under each pair task's selection
P validate --root <root> --label linear
P intervals --root <root> --tasks ppi2                                      # 1,000 draws of the test proteins
```

ESMC-600 and ESMC-6B add `--model ESMC-600` or `--model ESMC-6B`; ESMC-6B adds `--no-untrained` (its untrained-model
stores would take about 600 GB) and reads only the residual stream on the residue tasks (`--sites residual`).

## Interventions and latent views (ESMC-300, `run`)

```bash
P phospho --root run --proteins 1000 --max-tokens 65536 --tf32                                   # and --label phospho_s<seed> --seed <seed> --split-seed 284502 / 384502
P pca --root run --tasks interpro_family,interpro_domain,interpro_homologous_superfamily,localization,go_cc --sites resid_post --layers 0,5,10,15,20,25,29 --top 5
```

## Figures

`python -m tap_probing.reproduce` runs the drawing steps (`figures`, `trends`, `compare`) on the archived results, as
the study did on the GH200, with the submodule's commit `895a96c`, the branch's latest: its drawing code (`figures`, `trends`) is the survey
commit's `13c37ae` but for one shortened axis label of the pair figure, and the commits named below add the later
runs' commands.

## The pooling and MLP study (Protify `ddc1d03`)

The Supplement's mean-only pooling and MLP probes ran on the same branch at `ddc1d03`, a later commit than the
submodule's whose drawing code is the same, on 13 protein tasks at seeds 24895, 24896 and 24897. Its archive,
`GleghornLab/esmc_tap_probing_pooling_results`, keeps every root under `runs/<model>_<treatment>_s<seed>/` (models
`esmc300`, `esmc600`, `esmc6b`, `random300`; treatments `norm`, `raw`) and the bootstrap tables under
`runs/tables/<model>_<treatment>/`.

```bash
P embed --tasks protein --sites embedding,resid_post,final_norm,attn_context,mlp_act --no-controls --root <stem>_raw_s24895 --normalized-root <stem>_norm_s24895
P sweep --root esmc300_norm_s<seed> --seed <seed> --normalize --tasks protein --label p_b256 --batch-size 256 --probe linear mlp --pooling mean_var mean \
  --lr 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2 1e-1 --wd 0 0.1 --hidden 256 512 1024 --dropout 0.1 0.3 \
  --sites embedding,resid_post,final_norm,attn_context,mlp_act --no-heads --max-epochs 100 --patience 10 --bank-budget-gb 30 --no-controls
P sweep ... --tasks prot_param --label p_b1024 --batch-size 1024                     # the same grid
P select --root esmc300_norm_s24895 esmc300_norm_s24896 esmc300_norm_s24897 --labels p_b256_<label> [p_b1024_<label>] --loss <loss>
P train --root <root> --seed <seed> [--normalize] --tasks <tasks> --variants <variants.json> --sites embedding,resid_post,final_norm,attn_context,mlp_act \
  --no-heads --max-epochs 100 --patience 10 --bank-budget-gb 30 --no-controls     # the 12 classification tasks, then prot_param
P protein-intervals --roots <three seed roots> --labels linear linear_mean mlp mlp_mean --out tables/<condition>/intervals --tasks protein
P solve --roots <three seed roots> --labels linear linear_mean --out tables/<condition>/solver --tasks protein
```

A variants file lists, per results label (`linear`, `linear_mean`, `mlp`, `mlp_mean`), the probe kind, pooling and
setting `select` chose: for the classification tasks linear probes at learning rate 3e-2 without weight decay and MLP
probes with 256 hidden units, dropout 0.1, learning rate 1e-3 and weight decay 0.1, batch 256; for ProtParam linear
probes at 1e-3 and MLP probes with 512 hidden units, dropout 0.3, learning rate 3e-4, weight decay 0.1 for both, batch
1,024. The randomly initialized model adds `--random-init` and gets its own stores per seed; the trained models' seed
replicates read the seed-24895 stores.

## The survey on the pooled mean (Protify `ddc1d03`)

The Supplement repeats the survey's protein and `ppi` probes on the pooled mean alone (label `linear_mean`) where the
survey's archive lacks them, and adds the baseline encodings on `ppi`, under the survey's settings and controls, on
unit-norm states at all 17 sites with heads. The pooling study's archive keeps them compacted in the survey archive's
layout under `survey/runs/<root>/<label>/<task>/`, and `tap_probing.archive.overlay` adds them to the restored roots.

```bash
MEAN="--normalize --pooling mean --label linear_mean --max-epochs 100 --patience 10"
P embed --root run_norm_s<seed> --seed <seed> --normalize --tasks protein --controls --max-tokens 65536           # seeds 24896, 24897
P embed --root run_random_norm[_s<seed>] --seed <seed> --normalize --tasks protein --random-init --max-tokens 65536  # three seeds
P train --root <root> --seed <seed> --tasks <the 12 classification tasks> --batch-size 256 --lr 0.03 --wd 0 $MEAN
P train --root <root> --seed <seed> --tasks prot_param --batch-size 1024 --lr 1e-3 --wd 0.1 $MEAN               # random-init roots add --random-init --no-controls
P embed --root <root> --seed <seed> --normalize --tasks ppi --controls --max-tokens 131072                       # every trained root; ESMC-6B adds --no-untrained
P train --root <root> --seed <seed> --tasks ppi --batch-size 1024 --lr 3e-4 --wd 0.1 $MEAN
P intervals --root <root> --tasks ppi --label linear_mean
P embed --root run_onehot[_s<seed>] --seed <seed> --model OneHot-Protein --tasks ppi --max-tokens 131072           # and run_randvec with --model Random
P train --root run_onehot[_s<seed>] --seed <seed> --model OneHot-Protein --tasks ppi --batch-size 1024 --lr 3e-4 --wd 0.1 \
  --max-epochs 100 --patience 10 --no-controls --pooling <mean_var|mean> --label <linear|linear_mean>
```

## ESMC-600's probe variants and geometry (Protify `bb29510`)

Two further probes on ESMC-600's unit-norm pooled means at the pooling study's five sites and three seeds (labels
`bottleneck_mean`, the MLP with an identity activation, and `quadratic_mean`, a linear probe on the mean and its
elementwise square), their bootstrap tables beside the other mean-only probes', and the geometry of the seed-24895
store. The archive keeps them under `runs/esmc600_norm_s<seed>/results/`, `runs/tables/esmc600_norm/decomposition/`
and `geometry/esmc600_norm_s24895/`.

```bash
P geometry --root esmc600_norm_s24895 --out geometry/esmc600_norm_s24895
P train --root esmc600_norm_s<seed> --seed <seed> --model ESMC-600 --normalize --tasks <classification> --variants variants_classification.json \
  --sites embedding,resid_post,final_norm,attn_context,mlp_act --no-heads --max-epochs 100 --patience 10 --bank-budget-gb 30 --no-controls
P train ... --tasks prot_param --variants variants_regression.json                  # the same flags
P protein-intervals --roots esmc600_norm_s24895 esmc600_norm_s24896 esmc600_norm_s24897 \
  --labels linear_mean mlp_mean bottleneck_mean quadratic_mean --out tables/esmc600_norm/decomposition --tasks protein
```

The variants files: `bottleneck_mean` is `{"probe": "mlp", "pooling": "mean", "activation": "identity", "hidden": 256,
"dropout": 0.1, "lr": [1e-3], "wd": [0.1], "batch_size": 256}` (ProtParam: hidden 512, dropout 0.3, lr 3e-4, batch
1,024), `quadratic_mean` is `{"probe": "linear", "pooling": "mean_square", "lr": [3e-2], "wd": [0], "batch_size": 256}`
(ProtParam: lr 1e-3, wd 0.1, batch 1,024).

## ESMC-600's lenses, ablations and erasure (Protify `bb29510`)

Three analyses of the masked-language-model objective at seeds 24895, 24896 and 24897, each writing under
`causal/esmc600_s<seed>/` (the archive keeps every table and record):

```bash
P lens --model ESMC-600 --seed <seed> --out causal      # logit and tuned lenses, identity readout
P ablate --model ESMC-600 --seed <seed> --out causal    # mean and zero ablation of every attention and feed-forward write
P erase --model ESMC-600 --seed <seed> --out causal     # LEACE of four concepts at eight depths, with random-direction controls
```
