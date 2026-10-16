"""The public results archive: download it at its pinned revision, restore Protify's run layout and read its scores.

The archive keeps each probe's full-test score and the mean, standard deviation and count of its scores on the test
halves (`scores.parquet`), and each probe's validation score and loss (`probes.parquet`), under
`runs/<root>/<label>/<task>/`. `restore` puts every root back in the layout Protify's analyses read
(`<root>/results/<label>/<task>/`, beside `sweeps/`, `interventions/`, `latent/` and `stores/`), and `install` swaps
Protify's three score readers for ones that read the compacted tables: `summarize` pools every replicate's test halves
exactly from their means, standard deviations and counts, so Protify's figures, trends and comparisons run unchanged.
"""

from __future__ import annotations

import json
import re
import shutil
import numpy as np
import pandas as pd

from collections.abc import Sequence
from pathlib import Path
from huggingface_hub import snapshot_download
from src.protify.interpretability import figures

from tap_probing.pooling import pool_halves


REPO = "GleghornLab/esmc_tap_probing_results"
REVISION = "9f5e7b31"
KEYS = ["task", "family", "probe_name", "plm", "probe", "control"]
VALIDATION = ["task", "family", "probe_name", "validation_score", "validation_loss"]
RUN_FOLDERS = ("sweeps", "interventions", "latent", "stores", "pilots")
# Every table the figures read; the per-label tables, bootstrap draws and logs stay on the Hub.
PATTERNS = ["runs/*/*/*/run.json", "runs/*/*/*/scores.parquet", "runs/*/*/*/probes.parquet", "runs/*/*/*/ranking.parquet",
            "runs/*/*/*/intervals.parquet", "runs/*/linear/prot_param/elementwise.parquet", "runs/*/sweeps/*/*/*",
            "runs/*/latent/*/*", "runs/*/interventions/*/*.parquet", "runs/*/interventions/*/*.json",
            "runs/*/stores/*/manifest.json", "analyses/*.csv", "analyses/*/*/*"]
# Each model of the paper: the replicate roots (one per seed) of its analysis on unit-norm token states.
MODELS = {
    "ESMC-300": ("run_norm", "run_norm_s24896", "run_norm_s24897"),
    "ESMC-600": ("run_600_norm",),
    "ESMC-6B": ("run_6b_norm",),
}
RANDOM = {"ESMC-300 random init": ("run_random_norm", "run_random_norm_s24896", "run_random_norm_s24897")}
BASELINES = {"OneHot-Protein": ("run_onehot",), "Random": ("run_randvec",)}
RAW = ("run", "run_s24896", "run_s24897")
RANDOM_RAW = ("run_random", "run_random_s24896", "run_random_s24897")
BLOCKS = {"ESMC-300": 30, "ESMC-600": 36, "ESMC-6B": 80}


def download(out: Path) -> Path:
    """The archive's tables at the pinned revision, under `out`."""
    return Path(snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=out, allow_patterns=PATTERNS))


def restore(archive: Path, out: Path) -> Path:
    """Copy every root of the archive into Protify's run layout under `out`; returns `out`."""
    for root in sorted(path for path in (archive / "runs").iterdir() if path.is_dir()):
        for folder in sorted(path for path in root.iterdir() if path.is_dir()):
            target = out / root.name / (folder.name if folder.name in RUN_FOLDERS else Path("results") / folder.name)
            shutil.copytree(folder, target, dirs_exist_ok=True)
    return out


def read_sweeps(root: Path, labels: Sequence[str]) -> pd.DataFrame:
    """Protify's `selection.read_sweeps` without the sweeps' run records, which the archive does not keep: the batch
    comes from the label (`p_b256` is 256 proteins, `r_t32768` 32,768 tokens), each task's kind from its trained run's
    record, and the seconds, which only break ties in a selection the archive records, are left empty."""
    frames = []
    for label in labels:
        match = re.fullmatch(r"(?:p|ppi|r)_([bt])(\d+)", label)
        if match is None:
            raise ValueError(f"Sweep label {label!r} names no batch.")
        paths = sorted((root / "sweeps" / label).glob("*/sweep.parquet"))
        if not paths:
            raise FileNotFoundError(f"No sweep tables under {root / 'sweeps' / label}.")
        batch = int(match.group(2))
        for path in paths:
            record = json.loads((root / "results" / "linear" / path.parent.name / "run.json").read_text(encoding="utf-8"))
            frames.append(pd.read_parquet(path).assign(
                label=label, batch_size=batch if match.group(1) == "b" else None, max_tokens=batch if match.group(1) == "t" else None,
                task_seconds=np.nan, kind=record["task"]["kind"]))
    return pd.concat(frames, ignore_index=True)


def read_validation(replicates: Sequence[tuple[Path, str]]) -> pd.DataFrame:
    """Every replicate's validation score and loss of each probe, from its training records, with `replicate`."""
    return pd.concat([figures.read_table(Path(root), label, "probes")[VALIDATION].assign(replicate=index)
                      for index, (root, label) in enumerate(replicates)], ignore_index=True)


def summarize(scores: pd.DataFrame, validation: pd.DataFrame | None = None) -> pd.DataFrame:
    """Protify's `figures.summarize` on compacted scores: per probe and condition, `full` and `full_std` over
    replicates, `mean` and `std` over every replicate's test halves (pooled from each replicate's mean, standard
    deviation and count), `replicates`, the probe parts and the validation score and loss averaged over replicates."""
    if "replicate" not in scores:
        scores = scores.assign(replicate=0)
    halves = pool_halves(scores, KEYS)
    full = scores.groupby(KEYS)["full"].agg(full="mean", full_std="std", replicates="count")
    full["full_std"] = full["full_std"].fillna(0.0)
    summary = figures.with_probe_parts(halves.join(full).reset_index())
    if validation is not None:
        means = validation.groupby(["task", "family", "probe_name"])[["validation_score", "validation_loss"]].mean()
        summary = summary.join(means, on=["task", "family", "probe_name"])
    return summary


def full_scores(scores: pd.DataFrame, validation: pd.DataFrame | None = None) -> pd.DataFrame:
    """Protify's `figures.full_scores` on compacted scores: each replicate's full-test score as `score`, with the probe
    parts and, with the validation tables, each replicate's validation score and loss."""
    if "replicate" not in scores:
        scores = scores.assign(replicate=0)
    rows = figures.with_probe_parts(scores.rename(columns={"full": "score"}).assign(resample=0))
    if validation is not None:
        keys = ["task", "family", "probe_name", "replicate"]
        rows = rows.join(validation.groupby(keys)[["validation_score", "validation_loss"]].mean(), on=keys)
    return rows


def install() -> None:
    """Point Protify's figures, trends and comparisons at the compacted score tables."""
    figures.read_validation = read_validation
    figures.summarize = summarize
    figures.full_scores = full_scores


def model_summary(roots: Sequence[Path], label: str = "linear") -> pd.DataFrame:
    """The summary of one model's replicate roots, in Protify's run layout."""
    replicates = [(Path(root), label) for root in roots]
    scores = pd.concat([figures.read_table(root, label, "scores").assign(replicate=index) for index, (root, label) in enumerate(replicates)],
                       ignore_index=True)
    return summarize(scores, read_validation(replicates))


def verify(summary: pd.DataFrame, reference: pd.DataFrame, tolerance: float = 1e-4) -> float:
    """The largest absolute difference between a rebuilt summary and the one Protify wrote, over every score column;
    raises when the probes differ or a difference exceeds `tolerance` (the archive stores scores in float32)."""
    merged = summary.merge(reference, on=KEYS, suffixes=("", "_reference"), validate="one_to_one")
    if len(merged) != len(reference) or len(merged) != len(summary):
        raise ValueError(f"{len(summary)} rebuilt probes, {len(reference)} in the reference, {len(merged)} shared.")
    columns = ["mean", "std", "full", "full_std", "validation_score"]
    largest = max(float(np.nanmax(np.abs(merged[column] - merged[f"{column}_reference"]))) for column in columns)
    if largest > tolerance:
        raise ValueError(f"The rebuilt summary differs from the reference by up to {largest:.2e}.")
    return largest
