"""From readable to used: the figures, tables and numbers of the lenses, write ablations and concept erasure.

    python -m tap_probing.causal_figures --runs RUNS --out OUT [--data CONCEPTS]

`RUNS` holds one folder per model and seed, `<model>_s<seed>` (`esmc300`, `esmc600`, `esmc6b`), as Protify's `lens`,
`ablate` and `erase` write them: `lens.parquet`, `ablation.parquet` and, per erased concept, `<concept>/mlm.parquet`
and `<concept>/readouts.parquet`. Masked-token quantities are per-protein sums of the test proteins, so each is a
ratio of sums on the whole split and on 1,000 bootstrap draws of the proteins (Protify's `protein_draws`, seed 0, the
draws every model, seed and condition share); readouts come with their draws. Seeds are averaged draw by draw, and an
interval is the 2.5th to 97.5th percentile of the draws.

Outputs: `lens.csv`, `ablation.csv`, `erasure_mlm.csv`, `erasure_readouts.csv`, the main text's `lens_ablation.png`
and `erasure.png`, and the Supplement's `lens_by_structure.png` and `ablation_zero.png`.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from collections.abc import Sequence
from pathlib import Path


MODELS = (("esmc300", "ESMC-300"), ("esmc600", "ESMC-600"), ("esmc6b", "ESMC-6B"))
CONCEPTS = (("ssp_q3", "secondary structure"), ("localization", "localization"), ("prot_param", "composition"),
            ("interpro_family", "InterPro family"))


# -- per-protein sums to estimates with intervals ----------------------------------------------------------------------


def draws_for(proteins: int) -> np.ndarray:
    """Each test protein's count in the whole split and in every shared bootstrap draw: (B + 1, P)."""
    from src.protify.interpretability.scores import protein_draws

    return np.concatenate([np.ones((1, proteins)), protein_draws(proteins).numpy()])


def ratio(frame: pd.DataFrame, numerator: str, denominator: str, counts: np.ndarray) -> np.ndarray:
    """sum(numerator) / sum(denominator) over the proteins of `frame` (one row per protein, in order), on the whole
    split and every draw: (B + 1,)."""
    ordered = frame.sort_values("protein")
    return (counts @ ordered[numerator].to_numpy(np.float64)) / (counts @ ordered[denominator].to_numpy(np.float64))


def summarize(draws: np.ndarray) -> dict[str, float]:
    low, high = np.nanpercentile(draws[1:], [2.5, 97.5])
    return {"estimate": float(draws[0]), "low": float(low), "high": float(high)}


def seed_folders(runs: Path, stem: str) -> list[Path]:
    return sorted(path for path in runs.glob(f"{stem}_s*") if path.is_dir() and not path.name.startswith("pilot"))


def mean_over_seeds(per_seed: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(per_seed), axis=0)


# -- tables -------------------------------------------------------------------------------------------------------------


def lens_table(runs: Path) -> pd.DataFrame:
    """Per model, depth, lens and residue class: masked top-1 accuracy, NLL and KL from the model's prediction per
    masked residue (the identity readout: accuracy and NLL per unmasked residue), seeds averaged, with intervals."""
    rows = []
    for stem, model in MODELS:
        tables = [pd.read_parquet(folder / "lens.parquet") for folder in seed_folders(runs, stem) if (folder / "lens.parquet").exists()]
        if not tables:
            continue
        blocks = int(tables[0]["blocks"].iloc[0])
        counts = draws_for(tables[0]["protein"].nunique())
        for (depth, lens, kind), _ in tables[0].groupby(["depth", "lens", "class"]):
            parts = [table[(table["depth"] == depth) & (table["lens"] == lens) & (table["class"] == kind)] for table in tables]
            for metric, numerator in (("accuracy", "correct"), ("nll", "nll"), ("kl", "kl")):
                if lens == "identity" and metric == "kl":
                    continue
                values = mean_over_seeds([ratio(part, numerator, "masked", counts) for part in parts])
                rows.append({"model": model, "blocks": blocks, "depth": int(depth), "relative_depth": depth / (blocks + 1), "lens": lens,
                             "class": kind, "metric": metric, "seeds": len(parts), **summarize(values)})
    return pd.DataFrame(rows)


def ablation_table(runs: Path) -> pd.DataFrame:
    """Per model, block, write and ablation: the change in masked NLL and top-1 accuracy per masked residue from the
    unablated model, and KL from its prediction, seeds averaged, with intervals (paired by draw)."""
    rows = []
    for stem, model in MODELS:
        tables = [pd.read_parquet(folder / "ablation.parquet") for folder in seed_folders(runs, stem) if (folder / "ablation.parquet").exists()]
        if not tables:
            continue
        blocks = int(tables[0]["blocks"].iloc[0])
        counts = draws_for(tables[0]["protein"].nunique())
        clean = [table[table["block"] == -1] for table in tables]
        clean_nll = [ratio(part, "nll", "masked", counts) for part in clean]
        clean_accuracy = [ratio(part, "correct", "masked", counts) for part in clean]
        for (block, write, kind), _ in tables[0][tables[0]["block"] >= 0].groupby(["block", "write", "ablation"]):
            parts = [table[(table["block"] == block) & (table["write"] == write) & (table["ablation"] == kind)] for table in tables]
            quantities = {
                "delta_nll": mean_over_seeds([ratio(part, "nll", "masked", counts) - base for part, base in zip(parts, clean_nll)]),
                "delta_accuracy": mean_over_seeds([ratio(part, "correct", "masked", counts) - base for part, base in zip(parts, clean_accuracy)]),
                "kl": mean_over_seeds([ratio(part, "kl", "masked", counts) for part in parts]),
            }
            for metric, values in quantities.items():
                rows.append({"model": model, "blocks": blocks, "block": int(block), "relative_depth": (block + 1) / (blocks + 1), "write": write,
                             "ablation": kind, "metric": metric, "seeds": len(parts), **summarize(values)})
        rows += [{"model": model, "blocks": blocks, "block": -1, "relative_depth": np.nan, "write": "none", "ablation": "none", "metric": metric,
                  "seeds": len(clean), **summarize(mean_over_seeds(values))} for metric, values in (("nll", clean_nll), ("accuracy", clean_accuracy))]
    return pd.DataFrame(rows)


def erasure_tables(runs: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per model, concept, erased depth and condition: KL from the unerased prediction and NLL per masked residue, and
    LEACE minus random for both (paired by draw); and each linear readout of the concept, LEACE, random and unerased,
    seeds averaged with intervals."""
    mlm_rows, readout_rows = [], []
    for stem, model in MODELS:
        for concept, name in CONCEPTS:
            folders = [folder / concept for folder in seed_folders(runs, stem) if (folder / concept / "readouts.parquet").exists()]
            if not folders:
                continue
            mlm = [pd.read_parquet(folder / "mlm.parquet") for folder in folders]
            blocks = int(mlm[0]["blocks"].iloc[0])
            counts = draws_for(mlm[0]["protein"].nunique())
            for depth in sorted(mlm[0]["depth"].unique()):
                values = {}
                for condition in ("leace", "random"):
                    parts = [table[(table["depth"] == depth) & (table["condition"] == condition)] for table in mlm]
                    values[condition] = {metric: [ratio(part, metric, "masked", counts) for part in parts] for metric in ("kl", "nll")}
                    for metric in ("kl", "nll"):
                        mlm_rows.append({"model": model, "concept": name, "depth": int(depth), "relative_depth": depth / (blocks + 1),
                                         "condition": condition, "metric": metric, "seeds": len(parts), **summarize(mean_over_seeds(values[condition][metric]))})
                for metric in ("kl", "nll"):
                    difference = mean_over_seeds([a - b for a, b in zip(values["leace"][metric], values["random"][metric])])
                    mlm_rows.append({"model": model, "concept": name, "depth": int(depth), "relative_depth": depth / (blocks + 1),
                                     "condition": "leace_minus_random", "metric": metric, "seeds": len(folders), **summarize(difference)})
            readouts = [pd.read_parquet(folder / "readouts.parquet") for folder in folders]
            for (depth, condition, read), _ in readouts[0].groupby(["depth", "condition", "read"]):
                parts = [table[(table["depth"] == depth) & (table["condition"] == condition) & (table["read"] == read)].sort_values("draw")["value"].to_numpy()
                         for table in readouts]
                readout_rows.append({"model": model, "concept": name, "depth": int(depth), "condition": condition, "read": int(read),
                                     "relative_depth": depth / (blocks + 1) if depth >= 0 else np.nan, "read_relative_depth": read / (blocks + 1),
                                     "final": read == blocks + 1, "seeds": len(parts), **summarize(mean_over_seeds(parts))})
    return pd.DataFrame(mlm_rows), pd.DataFrame(readout_rows)


# -- figures ------------------------------------------------------------------------------------------------------------


def _band(axis, frame: pd.DataFrame, x: str, color: str, style: str = "-", label: str | None = None, width: float = 1.1) -> None:
    frame = frame.sort_values(x)
    axis.plot(frame[x], frame["estimate"], color=color, ls=style, lw=width, label=label)
    axis.fill_between(frame[x], frame["low"], frame["high"], color=color, alpha=0.15, lw=0)


def lens_ablation_figure(lens: pd.DataFrame, ablation: pd.DataFrame, path: Path, unigram: float | None = None) -> Path:
    """A, every depth's logit lens (solid) and tuned lens (dashed): masked top-1 accuracy per model. B, the identity
    readout's NLL of each unmasked residue's own token. C, the masked NLL each block's attention (circles) or
    feed-forward write (squares) costs when set to its mean, by the block's relative depth."""
    from src.protify.interpretability.figures import GRID, INK, INK_SECONDARY, PALETTE, _letter, _pyplot, _save

    plt = _pyplot()
    colors = {model: PALETTE[index] for index, (_, model) in enumerate(MODELS) if (lens["model"] == model).any()}  # the models drawn
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    masked = lens[(lens["class"] == "all") & (lens["metric"] == "accuracy")]
    for model, color in colors.items():
        for kind, style in (("logit", "-"), ("tuned", "--")):
            frame = masked[(masked["model"] == model) & (masked["lens"] == kind) & (masked["relative_depth"] < 1)]
            if len(frame):
                _band(axes[0], frame, "relative_depth", color, style, f"{model}, {kind} lens")
        identity = lens[(lens["model"] == model) & (lens["lens"] == "identity") & (lens["metric"] == "nll")]
        if len(identity):
            _band(axes[1], identity, "relative_depth", color, label=model)
        for write, marker in (("attn", "o"), ("ffn", "s")):
            frame = ablation[(ablation["model"] == model) & (ablation["write"] == write) & (ablation["ablation"] == "mean") & (ablation["metric"] == "delta_nll")]
            if len(frame):
                frame = frame.sort_values("relative_depth")
                axes[2].plot(frame["relative_depth"], frame["estimate"], color=color, lw=0.7, alpha=0.8, marker=marker, ms=2.2,
                             ls="-" if write == "ffn" else ":")
    if unigram is not None:
        axes[0].axhline(unigram, color=INK_SECONDARY, ls=":", lw=0.8)
        axes[0].text(0.02, unigram, "most frequent residue", fontsize=5, color=INK_SECONDARY, va="bottom")
    axes[0].set_ylabel("masked top-1 accuracy")
    axes[1].set_ylabel("identity readout NLL (nats)")
    axes[1].set_yscale("log")
    axes[2].set_ylabel("masked NLL cost (nats)")
    for letter, axis in zip("ABC", axes):
        axis.set_xlabel("relative depth")
        axis.set_xlim(0, 1.0)
        axis.grid(color=GRID, lw=0.5)
        _letter(axis, letter)
    handles = [plt.Line2D([], [], color=color, lw=1.2) for color in colors.values()]
    handles += [plt.Line2D([], [], color=INK, ls="-", lw=1.0), plt.Line2D([], [], color=INK, ls="--", lw=1.0),
                plt.Line2D([], [], color=INK, marker="o", ls=":", ms=3, lw=0.7), plt.Line2D([], [], color=INK, marker="s", ls="-", ms=3, lw=0.7)]
    labels = list(colors) + ["logit lens", "tuned lens", "attention write", "feed-forward write"]
    figure.legend(handles, labels, loc="upper center", ncol=7, frameon=False, fontsize=5.5)
    figure.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.0)
    return _save(figure, path)


def erasure_figure(mlm: pd.DataFrame, readouts: pd.DataFrame, path: Path) -> Path:
    """Per concept (columns): A, KL from the unerased prediction per masked residue after LEACE erasure (solid) and
    after a random subspace of the same rank (dashed) at each relative depth; B, the concept's linear readability at
    the final norm after erasure at each depth (solid LEACE, dashed random), against the unerased stream (dotted)."""
    from src.protify.interpretability.figures import GRID, INK, INK_SECONDARY, PALETTE, _letter, _pyplot, _save, _tag

    plt = _pyplot()
    colors = {model: PALETTE[index] for index, (_, model) in enumerate(MODELS) if (mlm["model"] == model).any()}  # the models drawn
    concepts = [name for _, name in CONCEPTS if name in set(mlm["concept"])]
    figure, axes = plt.subplots(2, len(concepts), figsize=(7.2, 3.6), squeeze=False)
    for column, concept in enumerate(concepts):
        top, bottom = axes[0][column], axes[1][column]
        for model, color in colors.items():
            for condition, style in (("leace", "-"), ("random", "--")):
                frame = mlm[(mlm["model"] == model) & (mlm["concept"] == concept) & (mlm["condition"] == condition) & (mlm["metric"] == "kl")]
                if len(frame):
                    _band(top, frame, "relative_depth", color, style)
                frame = readouts[(readouts["model"] == model) & (readouts["concept"] == concept) & (readouts["condition"] == condition) & readouts["final"]]
                if len(frame):
                    _band(bottom, frame, "relative_depth", color, style)
            clean = readouts[(readouts["model"] == model) & (readouts["concept"] == concept) & (readouts["condition"] == "clean") & readouts["final"]]
            if len(clean):
                bottom.axhline(clean["estimate"].iloc[0], color=color, ls=":", lw=0.8)
        _tag(top, concept)
        for axis in (top, bottom):
            axis.set_xlim(0, 1.0)
            axis.grid(color=GRID, lw=0.5)
        bottom.set_xlabel("relative depth of the erasure")
        if column == 0:
            top.set_ylabel("KL per masked residue (nats)")
            bottom.set_ylabel("readability at the final norm")
            _letter(top, "A")
            _letter(bottom, "B")
    handles = [plt.Line2D([], [], color=color, lw=1.2) for color in colors.values()]
    names = list(colors)
    handles += [plt.Line2D([], [], color=INK, ls="-"), plt.Line2D([], [], color=INK, ls="--"), plt.Line2D([], [], color=INK_SECONDARY, ls=":")]
    figure.legend(handles, names + ["LEACE erasure", "random subspace, same rank", "unerased"], loc="upper center", ncol=6, frameon=False, fontsize=5.5)
    figure.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.6, w_pad=0.6)
    return _save(figure, path)


def lens_by_structure_figure(lens: pd.DataFrame, path: Path, names: Sequence[str] = ("helix", "strand", "coil")) -> Path:
    """The logit lens's masked top-1 accuracy by the masked residue's three-state secondary structure, per model.

    The benchmark gives no integer order for its classes; `names` follows their frequencies among masked residues
    (35, 21 and 44% for classes 0, 1 and 2, as for helix, strand and coil).
    """
    from src.protify.interpretability.figures import GRID, PALETTE, _pyplot, _save, _tag

    plt = _pyplot()
    models = [model for _, model in MODELS if (lens["model"] == model).any()]
    classes = [kind for kind in lens["class"].unique() if kind != "all"]
    figure, axes = plt.subplots(1, len(models), figsize=(7.2, 2.1), sharey=True, squeeze=False)
    for axis, model in zip(axes[0], models):
        for index, kind in enumerate(classes):
            frame = lens[(lens["model"] == model) & (lens["lens"] == "logit") & (lens["class"] == kind) & (lens["metric"] == "accuracy")]
            _band(axis, frame, "relative_depth", PALETTE[index], label=names[index] if index < len(names) else kind)
        _tag(axis, model)
        axis.set_xlabel("relative depth")
        axis.grid(color=GRID, lw=0.5)
    axes[0][0].set_ylabel("logit-lens masked top-1 accuracy")
    axes[0][0].legend(frameon=False, fontsize=5.5)
    figure.tight_layout(w_pad=0.6)
    return _save(figure, path)


def ablation_zero_figure(ablation: pd.DataFrame, path: Path) -> Path:
    """Zero against mean ablation of every write: the masked NLL cost per model."""
    from src.protify.interpretability.figures import GRID, PALETTE, _pyplot, _save, _tag

    plt = _pyplot()
    models = [model for _, model in MODELS if (ablation["model"] == model).any()]
    figure, axes = plt.subplots(1, len(models), figsize=(7.2, 2.1), sharey=True, squeeze=False)
    for axis, model in zip(axes[0], models):
        for index, (write, kind) in enumerate((("attn", "mean"), ("attn", "zero"), ("ffn", "mean"), ("ffn", "zero"))):
            frame = ablation[(ablation["model"] == model) & (ablation["write"] == write) & (ablation["ablation"] == kind) & (ablation["metric"] == "delta_nll")]
            _band(axis, frame, "relative_depth", PALETTE[index], label=f"{'attention' if write == 'attn' else 'feed-forward'}, {kind}", width=0.9)
        _tag(axis, model)
        axis.set_xlabel("relative depth")
        axis.grid(color=GRID, lw=0.5)
    axes[0][0].set_ylabel("masked NLL cost (nats)")
    axes[0][0].legend(frameon=False, fontsize=5.5)
    figure.tight_layout(w_pad=0.6)
    return _save(figure, path)


def unigram_accuracy(data: str | None) -> float:
    """The share of the lens task's test residues that are its most frequent residue: the accuracy of always
    predicting it, the masked prediction's floor."""
    from collections import Counter
    from src.protify.interpretability.causal import LENS_TASK
    from src.protify.interpretability.concepts import load_concept

    residues = Counter("".join(load_concept(LENS_TASK, local_dir=data).splits["test"].sequences))
    return max(residues.values()) / sum(residues.values())


def write_all(runs: Path, out: Path, data: str | None = None) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    lens, ablation = lens_table(runs), ablation_table(runs)
    mlm, readouts = erasure_tables(runs)
    written = []
    for name, table in (("lens", lens), ("ablation", ablation), ("erasure_mlm", mlm), ("erasure_readouts", readouts)):
        table.to_csv(out / f"{name}.csv", index=False, float_format="%.5f")
        written.append(out / f"{name}.csv")
    written.append(lens_ablation_figure(lens, ablation, out / "lens_ablation.png", unigram_accuracy(data)))
    written.append(erasure_figure(mlm, readouts, out / "erasure.png"))
    written.append(lens_by_structure_figure(lens, out / "lens_by_structure.png"))
    written.append(ablation_zero_figure(ablation, out / "ablation_zero.png"))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", default=None, help="a local copy of the concept datasets (default: the Hub's)")
    args = parser.parse_args()
    for path in write_all(args.runs, args.out, args.data):
        print(path)


if __name__ == "__main__":
    main()
