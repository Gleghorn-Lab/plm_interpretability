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
import time
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

# The three-seed pooling and MLP study: its own archive, in Protify's run layout and not compacted, so its tables are
# read before `install` points Protify's readers at the compacted ones.
POOLING_REPO = "GleghornLab/esmc_tap_probing_pooling_results"
POOLING_REVISION = "73883ced"
POOLING_PATTERNS = ["runs/tables/*/*/*.parquet", "runs/*/results/*/*/scores.parquet", "runs/*/results/*/*/validation.parquet",
                    "runs/*/results/*/*/ranking.parquet", "runs/*/results/*/*/run.json",
                    # ESMC-600's geometry (`geometry/<store>/`) and causal analyses (`causal/esmc600_s<seed>/`)
                    "geometry/*/*.parquet", "causal/*/*.parquet", "causal/*/*.json", "causal/*/*/*.parquet", "causal/*/*/*.json",
                    # per-protein probe outputs and PPLL of every encoder (`familiarity/<encoder>/`), the familiarity analysis
                    "familiarity/*/*.parquet", "familiarity/*/manifest.json"]
POOLING_ROOTS = {"ESMC-300": ("esmc300_norm_s24895", "esmc300_norm_s24896", "esmc300_norm_s24897"),
                 "ESMC-6B": ("esmc6b_norm_s24895", "esmc6b_norm_s24896", "esmc6b_norm_s24897")}
# The survey's mean-only probes that its own archive lacks (ESMC-300 at seeds 24896 and 24897, the randomly initialized
# model, and every model and baseline on the earlier pair task `ppi`), compacted in the layout of `REPO` under
# `survey/` of the pooling study's archive, so `overlay` adds them to the restored roots.
SURVEY_PATTERNS = [f"survey/{pattern}" for pattern in PATTERNS if pattern.startswith("runs/*/*/*/")]
# The Supplement repeats the main text's survey with protein probes and `ppi`'s pair probes on the pooled mean, and
# residue probes, which pool nothing, on token states; `compose` gathers them into this results label, so Protify's
# drawing code reads one run.
MEAN_POOLED = "mean_pooled"
MEAN_ONLY = ("ppi",)  # the pair tasks read on the pooled mean; `ppi2` ran under mean and variance only


def snapshot(repo: str, revision: str, out: Path, patterns: Sequence[str], attempts: int = 12, pause: float = 120.0) -> Path:
    """`snapshot_download` of a dataset's files matching `patterns` at `revision`, under `out`. The pooling archive holds
    thousands of small tables, enough to meet the Hub's rate limit (HTTP 429); the download then pauses and resumes,
    skipping the files it already has."""
    for attempt in range(attempts):
        try:
            return Path(snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=out, allow_patterns=list(patterns), max_workers=2))
        except Exception as error:
            if "429" not in str(error) or attempt == attempts - 1:
                raise
            print(f"The Hub's rate limit; resuming the download in {pause:.0f} s ({attempt + 1} of {attempts - 1}).", flush=True)
            time.sleep(pause)
    raise AssertionError("unreachable")


def download(out: Path) -> Path:
    """The archive's tables at the pinned revision, under `out`."""
    return snapshot(REPO, REVISION, out, PATTERNS)


def download_pooling(out: Path) -> Path:
    """The pooling and MLP study's tables and the survey's mean-only additions at its pinned revision, under `out`."""
    return snapshot(POOLING_REPO, POOLING_REVISION, out, POOLING_PATTERNS + SURVEY_PATTERNS)


def compact_scores(scores: pd.DataFrame, keys: Sequence[str] = KEYS) -> pd.DataFrame:
    """Protify's per-resample scores as the archive keeps them: one row per probe and condition with the full-test score
    (resample 0) and the mean, standard deviation and count of the scores on the test halves."""
    full = scores[scores["resample"] == 0].set_index(list(keys))["score"].rename("full")
    halves = scores[scores["resample"] > 0].groupby(list(keys))["score"].agg(halves_mean="mean", halves_std="std", halves="count")
    table = pd.concat([full, halves], axis=1).reset_index()
    return table.astype({"full": "float32", "halves_mean": "float32", "halves_std": "float32", "halves": "int8"})


def stage_survey(runs: Path, out: Path) -> dict[str, list[dict[str, object]]]:
    """The survey's reruns, as Protify wrote them (`runs/<root>/results/<label>/<task>/`), in the archive's layout under
    `out/runs/<root>/<label>/<task>/`, which `overlay` reads: the run record, the compacted scores, the probe records
    (training records with each probe's validation score and loss), ProtParam's per-target scores of full-width probes,
    compacted the same way, and, for pair tasks, the ranking metrics and bootstrap intervals. A task folder without a run
    record is unfinished and skipped. Writes `out/catalog.json`, the files kept and the files of each task folder left
    out, and returns it."""
    kept, left = [], []
    staged = ("run.json", "scores.parquet", "training.parquet", "validation.parquet", "ranking.parquet", "intervals.parquet")
    for folder in sorted(runs.glob("*/results/*/*")):
        if not (folder / "run.json").exists():
            continue
        root, label, task = folder.parent.parent.parent.name, folder.parent.name, folder.name
        target = out / "runs" / root / label / task
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder / "run.json", target / "run.json")
        compact_scores(pd.read_parquet(folder / "scores.parquet")).to_parquet(target / "scores.parquet", index=False)
        taken = set(staged)
        if task == "prot_param" and (folder / "elementwise.parquet").exists():
            elementwise = pd.read_parquet(folder / "elementwise.parquet")
            elementwise = elementwise[elementwise["family"] == "full"]
            compact_scores(elementwise, ["task", "family", "probe_name", "label_index"]).to_parquet(target / "elementwise.parquet", index=False)
            taken.add("elementwise.parquet")
        probes = pd.read_parquet(folder / "training.parquet")
        if (folder / "validation.parquet").exists():
            probes = probes.merge(pd.read_parquet(folder / "validation.parquet"), on=["task", "family", "probe_name"], how="left", validate="one_to_one")
        probes.to_parquet(target / "probes.parquet", index=False)
        for name in ("ranking.parquet", "intervals.parquet"):
            if (folder / name).exists():
                shutil.copyfile(folder / name, target / name)
        kept += [{"path": path.relative_to(out).as_posix(), "bytes": path.stat().st_size} for path in sorted(target.iterdir())]
        left += [{"path": path.relative_to(runs).as_posix(), "bytes": path.stat().st_size} for path in sorted(folder.iterdir())
                 if path.is_file() and path.name not in taken]
    catalog = {"kept": kept, "omitted": left}
    (out / "catalog.json").write_text(json.dumps(catalog, indent=1), encoding="utf-8", newline="\n")
    return catalog


def overlay(survey: Path, runs: Path) -> Path:
    """Add the survey's mean-only additions (`<pooling archive>/survey`) to the roots `restore` wrote under `runs`;
    raises rather than replace a file the survey's own archive holds."""
    for source in sorted(path for path in (survey / "runs").rglob("*") if path.is_file()):
        root, label, *rest = source.relative_to(survey / "runs").parts
        target = runs / root / "results" / label / Path(*rest)
        if target.exists():
            raise FileExistsError(f"{target} is in the survey's archive and in its mean-only additions.")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return runs


def compose(root: Path, label: str = MEAN_POOLED) -> Path:
    """Gather in `root` the probes the Supplement's mean-only survey reads under one results label: each protein task's and `ppi`'s
    mean-only probes (`linear_mean`) and each residue task's probes (`linear`); `ppi2`, which ran under mean and variance
    pooling only, is left out. Raises when a protein task or `ppi` has no mean-only probes."""
    results = root / "results"
    tasks = {path.parent.name: path for label_folder in ("linear", "linear_mean") for path in (results / label_folder).glob("*/run.json")}
    for name, record in sorted(tasks.items()):
        level = json.loads(record.read_text(encoding="utf-8"))["task"]["level"]
        if level == "residue":
            source = results / "linear" / name
        elif level == "protein" or name in MEAN_ONLY:
            source = results / "linear_mean" / name
        else:
            continue
        if not (source / "run.json").exists():
            raise FileNotFoundError(f"{root.name} has no {source.parent.name} probes of {name}.")
        shutil.copytree(source, results / label / name, dirs_exist_ok=True)
    return root


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
