# The runs

Every run used the `interpretability` package of Protify at the pinned commit (`protify/`, run as
`python -m src.protify.interpretability` from the submodule), in Docker with PyTorch and CUDA 12.6 on one NVIDIA GH200
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
the study did on the GH200.
