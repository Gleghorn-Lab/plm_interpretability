"""Redraw every figure and table of the paper from the public results archive.

    python -m tap_probing.reproduce --out reproduced [--archive archive]

Downloads the archive at its pinned revision (or reads `--archive`), restores Protify's run layout, checks each
model's rebuilt probe summary against the one written when the figures were first drawn, then runs Protify's own
drawing code as the study did (`figures` per model, `trends` across models, `compare` across input treatments) and
draws the cross-model Supplement figures and tables. Outputs: `<out>/{trends,main,raw,esmc600,esmc6b,compare,cross_model}`.
"""

from __future__ import annotations

import argparse
import json
import pandas as pd

from collections.abc import Sequence
from pathlib import Path
from src.protify.interpretability import figures, trends
from src.protify.interpretability.selection import rank_settings

from tap_probing import archive, cross_model, solver_table


PROTEIN_SWEEPS = ("p_b256", "p_b1024")
RESIDUE_SWEEPS = ("r_t32768", "r_t131072")
PHOSPHO = ("phospho", "phospho_s24896", "phospho_s24897")
REFERENCES = {"ESMC-300": "main", "ESMC-600": "esmc600", "ESMC-6B": "esmc6b"}


def _roots(runs: Path, names: Sequence[str]) -> list[Path]:
    return [runs / name for name in names]


def draw_model(runs: Path, roots: Sequence[str], out: Path, blocks: int, random_roots: Sequence[str] = (),
               shared: tuple[str, str] | None = None) -> list[Path]:
    """Protify's `figures` command for one model. `shared` names the trained and random-init roots whose sweeps,
    phosphomimetic scans and principal coordinates the figures read (run once, on raw states, and borrowed by every
    input treatment); without it the figures leave them out."""
    first = runs / roots[0]
    source = runs / (shared[0] if shared else roots[0])
    metadata = {path.parent.name: json.loads(path.read_text(encoding="utf-8"))["task"]
                for path in (first / "results" / "linear").glob("*/run.json")}
    rankings = {}
    if shared is not None:
        rankings = {regime: rank_settings(archive.read_sweeps(source, labels)) for regime, labels in (("protein", PROTEIN_SWEEPS), ("residue", RESIDUE_SWEEPS))}
    phospho = [source / "interventions" / label for label in PHOSPHO if (source / "interventions" / label / "scores.parquet").exists()] if shared else []
    latent = {}
    if shared is not None:
        random_source = runs / shared[1]
        latent = {name: root / "latent" / "pca" for name, root in (("trained model", source), ("random-init model", random_source))
                  if (root / "latent" / "pca").exists()}
    return figures.draw(_roots(runs, roots), out, blocks, "linear", _roots(runs, random_roots), phospho, latent, rankings, metadata)


def verify(runs: Path, analyses: Path) -> None:
    """Each model's rebuilt summary against the one Protify wrote when the figures were drawn."""
    for model, folder in REFERENCES.items():
        reference = analyses / folder / "summary.parquet"
        if reference.exists():
            difference = archive.verify(archive.model_summary(_roots(runs, archive.MODELS[model])), pd.read_parquet(reference))
            print(f"{model}: summary matches the archive's to {difference:.1e}")


def cross_model_figures(runs: Path, out: Path) -> list[Path]:
    """The Supplement's figures and tables across ESMC-300, ESMC-600 and ESMC-6B."""
    summaries = {model: archive.model_summary(_roots(runs, roots)) for model, roots in archive.MODELS.items()}
    summaries = {model: summary[~summary["task"].isin(trends.PAIR_TASKS)] for model, summary in summaries.items()}
    random = archive.model_summary(_roots(runs, archive.RANDOM["ESMC-300 random init"]))
    onehot = archive.model_summary(_roots(runs, archive.BASELINES["OneHot-Protein"]))
    best = cross_model.best_table(summaries, archive.BLOCKS)
    out.mkdir(parents=True, exist_ok=True)
    best.to_csv(out / "best_states_all.csv", index=False, float_format="%.4f")
    return [cross_model.depth_all(summaries, archive.BLOCKS, random, onehot, out / "depth_all.png"),
            cross_model.site_gaps_all(summaries, out / "site_gaps_all.png"),
            cross_model.best_depth_all(best, out / "best_depth_all.png"),
            cross_model.best_block_tables(best, out / "best_blocks.tex"), out / "best_states_all.csv"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="folder for the figures and tables")
    parser.add_argument("--archive", type=Path, default=None, help="an archive already downloaded (default: download into <out>/archive)")
    args = parser.parse_args()
    source = args.archive or archive.download(args.out / "archive")
    runs = archive.restore(source, args.out / "runs")
    archive.install()
    verify(runs, source / "analyses" / "final2")
    written = []
    models = [trends.ModelRuns(model, tuple(_roots(runs, roots))) for model, roots in archive.MODELS.items()]
    random = [trends.ModelRuns(model, tuple(_roots(runs, roots))) for model, roots in archive.RANDOM.items()]
    baselines = [trends.ModelRuns(model, tuple(_roots(runs, roots))) for model, roots in archive.BASELINES.items()]
    written += trends.trends(models, args.out / "trends", random, baselines)
    written += draw_model(runs, archive.MODELS["ESMC-300"], args.out / "main", 30, archive.RANDOM["ESMC-300 random init"], shared=("run", "run_random"))
    written += draw_model(runs, archive.RAW, args.out / "raw", 30, archive.RANDOM_RAW, shared=("run", "run_random"))
    written += draw_model(runs, archive.MODELS["ESMC-600"], args.out / "esmc600", 36)
    written += draw_model(runs, archive.MODELS["ESMC-6B"], args.out / "esmc6b", 80)
    variants = {"normalized": [(root, "linear") for root in _roots(runs, archive.MODELS["ESMC-300"])],
                "raw": [(root, "linear") for root in _roots(runs, archive.RAW)],
                "normalized mean-only": [(runs / "run_norm", "linear_mean")], "raw mean-only": [(runs / "run", "linear_mean")]}
    written += figures.compare(variants, args.out / "compare", 30)
    written += cross_model_figures(runs, args.out / "cross_model")
    written.append(solver_table.solver_table(pd.read_csv(source / "analyses" / "solver_check.csv"), args.out / "cross_model" / "solver_table.tex"))
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
