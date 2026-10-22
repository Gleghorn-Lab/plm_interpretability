"""Figures and tables that put ESMC-300, ESMC-600 and ESMC-6B on one relative depth axis, for the Supplement.

Relative depth places the token embedding at 0, the output of block k of n_b at (k + 1)/(n_b + 1) and the final layer
norm at 1. Best states are chosen on validation and reported on test, as Protify's `figures.best_states` does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from collections.abc import Sequence
from pathlib import Path
from src.protify.interpretability import figures
from src.protify.interpretability.figures import GRID, INK, INK_SECONDARY, MUTED, PALETTE, RESIDUAL, SITE_LABELS, STRUCTURE, STRUCTURE_X
from src.protify.interpretability.trends import CATEGORIES, CATEGORY_OF, CONCEPT_ORDER


MODEL_COLORS = {"ESMC-300": PALETTE[0], "ESMC-600": PALETTE[1], "ESMC-6B": PALETTE[2]}
FAMILIES = (("residual", "residual stream"), ("attention", "attention sites"), ("mlp", "feed-forward sites"), ("head", "single head"))
# Short column labels for the kinds of concept.
KIND_LABELS = {"secondary structure": "sec. structure", "sorting and topology": "sorting, topology", "InterPro membership": "InterPro",
               "residue sites and modifications": "residue sites"}


def relative(summary: pd.DataFrame, blocks: int) -> pd.DataFrame:
    """The summary with each state's depth on the residual-stream axis and its relative depth."""
    depths = np.array([figures.depth(site, layer, blocks) for site, layer in zip(summary["site"], summary["layer"], strict=True)], dtype=int)
    return summary.assign(depth=depths, relative_depth=depths / (blocks + 1))


def _stream(summary: pd.DataFrame) -> pd.DataFrame:
    trained = figures.headline(summary)
    return trained[(trained["family"] == "full") & trained["site"].isin(RESIDUAL)].sort_values("depth")


def structure_scores(summary: pd.DataFrame) -> pd.Series:
    """Each task's structure-stream test score, by task; empty where it was not probed."""
    trained = figures.headline(summary)
    return trained[(trained["family"] == "full") & trained["site"].isin(STRUCTURE)].set_index("task")["full"]


def depth_all(summaries: dict[str, pd.DataFrame], blocks: dict[str, int], random: pd.DataFrame, onehot: pd.DataFrame,
              path: Path, tasks: Sequence[str] = CONCEPT_ORDER) -> Path:
    """One panel per task: the residual-stream probe's test score along relative depth for every model (one standard
    deviation over test halves shaded), the randomly initialized ESMC-300's own probes (dashed) and the one-hot encoding
    (dotted level), and, past the final norm, each model's structure stream (diamonds) where it was probed."""
    plt = figures._pyplot()
    streams = {model: _stream(relative(summary, blocks[model])) for model, summary in summaries.items()}
    structures = {model: structure_scores(summary) for model, summary in summaries.items()}
    random_stream = _stream(relative(random, blocks["ESMC-300"]))
    random_structure = structure_scores(random)
    onehot_level = figures.headline(onehot).query("family == 'full'").groupby("task")["full"].max()
    tasks = [task for task in tasks if any((stream["task"] == task).any() for stream in streams.values())]
    figure, axes = figures._grid(plt, len(tasks) + 1, 4, width=7.2, row_height=1.3, sharex=True)
    for axis, task in zip(axes, tasks, strict=False):  # the grid has one panel more, for the legend
        for model, stream in streams.items():
            curve = stream[stream["task"] == task]
            axis.plot(curve["relative_depth"], curve["full"], color=MODEL_COLORS[model], lw=1.1)
            axis.fill_between(curve["relative_depth"], curve["mean"] - curve["std"], curve["mean"] + curve["std"],
                              color=MODEL_COLORS[model], alpha=0.15, lw=0)
            if task in structures[model].index:
                axis.scatter([STRUCTURE_X], [structures[model][task]], marker="D", s=9, color=MODEL_COLORS[model], zorder=3, lw=0)
        curve = random_stream[random_stream["task"] == task]
        axis.plot(curve["relative_depth"], curve["full"], color=MUTED, ls="--", lw=0.9)
        if task in random_structure.index:
            axis.scatter([STRUCTURE_X], [random_structure[task]], marker="D", s=9, facecolors="white", edgecolors=MUTED, zorder=3, lw=0.7)
        if task in onehot_level.index:
            axis.axhline(onehot_level[task], color=INK_SECONDARY, ls=":", lw=0.9)
        axis.grid(axis="y", color=GRID, lw=0.5)
        axis.set_xticks([0.0, 0.5, 1.0])
        figures._tag(axis, figures.SHORT_LABELS.get(task, task))
        if figures._bottom(axis, axes, len(tasks)):
            axis.set_xlabel("relative depth")
            axis.tick_params(labelbottom=True)
    legend = axes[len(tasks)]
    legend.set_visible(True)
    figures._bare(legend)
    legend.set_xticks([])
    legend.set_yticks([])
    handles = [legend.plot([], [], color=color, lw=1.2)[0] for color in MODEL_COLORS.values()]
    handles += [legend.plot([], [], color=MUTED, ls="--")[0], legend.plot([], [], color=INK_SECONDARY, ls=":")[0]]
    labels = list(MODEL_COLORS) + ["random init", "one-hot"]
    if any(not scores.empty for scores in structures.values()):
        handles.append(legend.plot([], [], ls="none", marker="D", ms=3, color=INK)[0])
        labels.append("structure stream")
    legend.legend(handles, labels, loc="center", frameon=False, fontsize=6)
    figure.tight_layout(h_pad=0.9, w_pad=0.5)
    return figures._save(figure, path)


def site_gaps_all(summaries: dict[str, pd.DataFrame], path: Path, tasks: Sequence[str] = CONCEPT_ORDER) -> Path:
    """One map per model: each site's best probe over blocks minus the task's best residual-stream probe, both chosen
    on validation; differences of at least 0.05 in size are printed, and states not probed are hatched gray."""
    plt = figures._pyplot()
    gaps = {model: figures.site_gaps(summary, tasks) for model, summary in summaries.items()}
    sites = [site for site in SITE_LABELS if any(site in gap.columns for gap in gaps.values())]
    rows = [task for task in tasks if any(task in gap.index for gap in gaps.values())]
    figure, axes = plt.subplots(1, len(gaps), figsize=(7.2, 0.17 * len(rows) + 1.3), sharey=True)
    colormap = figures._colormap(figures.DIVERGING, "gaps")
    colormap.set_bad("white")
    image, missing = None, False
    for letter, (axis, (model, gap)) in zip("ABC", zip(axes, gaps.items(), strict=True), strict=False):
        matrix = gap.reindex(index=rows, columns=sites).to_numpy(dtype=float)
        image = axis.imshow(np.ma.masked_invalid(matrix), cmap=colormap, vmin=-0.1, vmax=0.1, aspect="auto")
        missing |= _hatch_missing(axis, matrix)
        _set_apart_structure(axis, sites, "columns")
        for row, column in zip(*np.nonzero(np.abs(np.nan_to_num(matrix)) >= 0.05), strict=True):
            value = matrix[row, column]
            axis.text(column, row, f"{value:+.2f}".replace("0.", "."), ha="center", va="center", fontsize=3.6,
                      color="white" if abs(value) > 0.08 else INK)
        axis.set_xticks(range(len(sites)))
        axis.set_xticklabels([SITE_LABELS[site] for site in sites], rotation=90)
        axis.set_yticks(range(len(rows)))
        axis.set_yticklabels([figures.SHORT_LABELS.get(task, task) for task in rows])
        axis.tick_params(length=0)
        figures._tag(axis, model)
        figures._letter(axis, letter, x=-0.02 if letter != "A" else -0.42)
        figures._bare(axis)
    figure.tight_layout(w_pad=0.4, rect=(0, 0.05, 1, 1))
    bar = figure.colorbar(image, cax=figure.add_axes((0.35, 0.012, 0.3, 0.012)), orientation="horizontal", extend="both")
    bar.set_ticks([-0.1, -0.05, 0.0, 0.05, 0.1])
    bar.set_label("site minus residual stream (best state of each)", fontsize=5.5)
    bar.ax.tick_params(labelsize=5)
    if missing:
        _not_probed_key(figure, (0.7, 0.005, 0.2, 0.025))
    return figures._save(figure, path)


def kind_gaps(summary: pd.DataFrame, tasks: Sequence[str] = CONCEPT_ORDER) -> pd.DataFrame:
    """Sites by kinds of concept: the median over a kind's tasks of each site's best probe over blocks minus the task's
    best residual-stream probe, both chosen on validation (`figures.site_gaps`)."""
    gaps = figures.site_gaps(summary, tasks)
    kinds = [kind for kind, members in CATEGORIES if any(task in gaps.index for task in members)]
    return gaps.groupby(gaps.index.map(CATEGORY_OF)).median().reindex(kinds).T


def state_concepts(summaries: dict[str, pd.DataFrame], path: Path, limit: float = 0.1) -> Path:
    """One map per model of which kinds of state read which kinds of concept: `kind_gaps`, printed where the value
    rounds to a nonzero hundredth, hatched gray where no task of the kind was probed at the site."""
    plt = figures._pyplot()
    gaps = {model: kind_gaps(summary) for model, summary in summaries.items()}
    sites = [site for site in SITE_LABELS if any(site in gap.index for gap in gaps.values())]
    kinds = [kind for kind, _ in CATEGORIES if any(kind in gap.columns for gap in gaps.values())]
    figure, axes = plt.subplots(1, len(gaps), figsize=(7.2, 0.16 * len(sites) + 1.3), sharey=True)
    colormap = figures._colormap(figures.DIVERGING, "kind_gaps")
    colormap.set_bad("white")
    image, missing = None, False
    for letter, (axis, (model, gap)) in zip("ABC", zip(axes, gaps.items(), strict=True), strict=False):
        matrix = gap.reindex(index=sites, columns=kinds).to_numpy(dtype=float)  # (sites, kinds)
        image = axis.imshow(np.ma.masked_invalid(matrix), cmap=colormap, vmin=-limit, vmax=limit, aspect="auto")
        missing |= _hatch_missing(axis, matrix)
        _set_apart_structure(axis, sites, "rows")
        for row, column in zip(*np.nonzero(np.isfinite(matrix)), strict=True):
            value = matrix[row, column]
            if round(abs(value), 2) == 0:
                continue
            axis.text(column, row, f"{value:+.2f}".replace("0.", "."), ha="center", va="center", fontsize=4.2,
                      color="white" if abs(value) > 0.7 * limit else INK)
        axis.set_xticks(range(len(kinds)))
        axis.set_xticklabels([KIND_LABELS.get(kind, kind) for kind in kinds], rotation=45, ha="right", fontsize=5.5)
        axis.set_yticks(range(len(sites)))
        axis.set_yticklabels([SITE_LABELS[site] for site in sites])
        axis.tick_params(length=0)
        figures._tag(axis, model)
        figures._letter(axis, letter, x=-0.02 if letter != "A" else -0.38)
        figures._bare(axis)
    figure.tight_layout(w_pad=0.4, rect=(0, 0.06, 1, 1))
    bar = figure.colorbar(image, cax=figure.add_axes((0.3, 0.015, 0.3, 0.012)), orientation="horizontal", extend="both")
    bar.set_ticks([-limit, -limit / 2, 0.0, limit / 2, limit])
    bar.set_label("site minus residual stream (best state of each)", fontsize=5.5)
    bar.ax.tick_params(labelsize=5)
    if missing:
        _not_probed_key(figure, (0.66, 0.008, 0.2, 0.03))
    return figures._save(figure, path)


def _set_apart_structure(axis, sites: list[str], along: str) -> None:
    """A rule before the structure stream's row or column of a site map, which is not a state of ESMC itself."""
    for index, site in enumerate(sites):
        if site in STRUCTURE and index:
            (axis.axhline if along == "rows" else axis.axvline)(index - 0.5, color=INK, lw=0.9)


def _hatch_missing(axis, matrix: np.ndarray) -> bool:
    """Hatch gray every cell of an `imshow` map whose value is missing, a state not probed; returns whether any was."""
    import matplotlib.patches as patches

    cells = list(zip(*np.nonzero(~np.isfinite(matrix)), strict=True))
    for row, column in cells:
        axis.add_patch(patches.Rectangle((column - 0.5, row - 0.5), 1, 1, facecolor=GRID, edgecolor=MUTED, hatch="////", lw=0))
    return bool(cells)


def _not_probed_key(figure, rect: tuple[float, float, float, float]) -> None:
    """A legend entry for `_hatch_missing`'s cells, in a small axes at `rect` (figure coordinates)."""
    import matplotlib.patches as patches

    key = figure.add_axes(rect)
    key.add_patch(patches.Rectangle((0.0, 0.2), 0.08, 0.6, facecolor=GRID, edgecolor=MUTED, hatch="////", lw=0))
    key.text(0.11, 0.5, "not probed", fontsize=5.5, va="center", ha="left", color=INK)
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    key.axis("off")


def best_table(summaries: dict[str, pd.DataFrame], blocks: dict[str, int]) -> pd.DataFrame:
    """Per model and task, `figures.best_states` with each family's relative depth."""
    tables = []
    for model, summary in summaries.items():
        table = figures.best_states(summary, blocks[model])
        for family, _ in FAMILIES:
            if family == "head":
                table["head_depth"] = table["head_layer"] + 1 if "head_layer" in table else np.nan
            if f"{family}_depth" in table:
                table[f"{family}_relative_depth"] = table[f"{family}_depth"] / (blocks[model] + 1)
        tables.append(table.assign(model=model, blocks=blocks[model]))
    return pd.concat(tables, ignore_index=True)


def best_depth_all(best: pd.DataFrame, path: Path, tasks: Sequence[str] = CONCEPT_ORDER) -> Path:
    """One panel per family of states: the relative depth of each task's best state of that family, chosen on
    validation, for every model."""
    plt = figures._pyplot()
    rows = [task for task in tasks if task in set(best["task"])]
    positions = np.arange(len(rows))
    figure, axes = plt.subplots(1, len(FAMILIES), figsize=(7.2, 0.17 * len(rows) + 1.0), sharey=True)
    offsets = dict(zip(MODEL_COLORS, (-0.22, 0.0, 0.22), strict=True))
    for letter, (axis, (family, title)) in zip("ABCD", zip(axes, FAMILIES, strict=True), strict=True):
        for position in positions[::2]:
            axis.axhspan(position - 0.5, position + 0.5, color="#f6f5f2", lw=0, zorder=0)
        for model, color in MODEL_COLORS.items():
            table = best[best["model"] == model].set_index("task").reindex(rows)
            column = f"{family}_relative_depth"
            if column in table:
                axis.scatter(table[column], positions + offsets[model], s=9, color=color, zorder=3, label=model)
        axis.set_xlim(-0.04, 1.04)
        axis.set_xticks([0.0, 0.5, 1.0])
        axis.set_xlabel("relative depth")
        axis.grid(axis="x", color=GRID, lw=0.5)
        figures._tag(axis, title)
        figures._letter(axis, letter, x=-0.03 if letter != "A" else -0.62)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([figures.SHORT_LABELS.get(task, task) for task in rows])
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    figure.tight_layout(w_pad=0.5, rect=(0, 0, 1, 1 - 0.25 / figure.get_figheight()))
    return figures._save(figure, path)


def structure_table(summaries: dict[str, pd.DataFrame], best: pd.DataFrame) -> pd.DataFrame:
    """Per model and task whose structure stream was probed: its test score (`structure`), and those of the best
    residual-stream state (`residual`), the best state of any of the model's own sites (`any`), each chosen on
    validation, and the final norm (`final_norm`)."""
    rows = []
    for model, summary in summaries.items():
        trained = figures.headline(summary)
        final = trained[(trained["family"] == "full") & (trained["site"] == "final_norm")].set_index("task")["full"]
        table = best[best["model"] == model].set_index("task")
        for task, score in structure_scores(summary).items():
            rows.append({"model": model, "task": task, "category": CATEGORY_OF.get(task, task), "structure": float(score),
                         "residual": float(table.at[task, "residual_score"]), "any": float(table.at[task, "any_score"]),
                         "final_norm": float(final.get(task, np.nan))})
    return pd.DataFrame(rows, columns=["model", "task", "category", "structure", "residual", "any", "final_norm"])


def structure_gaps(table: pd.DataFrame, path: Path, tasks: Sequence[str] = CONCEPT_ORDER) -> Path:
    """Per task, for every model: the structure stream's test score minus that of the best residual-stream state,
    chosen on validation (A), and minus the final norm's (B)."""
    plt = figures._pyplot()
    rows = [task for task in tasks if task in set(table["task"])]
    positions = np.arange(len(rows))
    panels = (("residual", "minus best stream state"), ("final_norm", "minus final norm"))
    figure, axes = plt.subplots(1, len(panels), figsize=(5.6, 0.17 * len(rows) + 1.0), sharey=True)
    offsets = dict(zip(MODEL_COLORS, (-0.22, 0.0, 0.22), strict=True))
    for letter, (axis, (column, title)) in zip("AB", zip(axes, panels, strict=True), strict=True):
        for position in positions[::2]:
            axis.axhspan(position - 0.5, position + 0.5, color="#f6f5f2", lw=0, zorder=0)
        axis.axvline(0.0, color=INK_SECONDARY, lw=0.7, zorder=1)
        for model, color in MODEL_COLORS.items():
            part = table[table["model"] == model].set_index("task").reindex(rows)
            if part["structure"].notna().any():
                axis.scatter(part["structure"] - part[column], positions + offsets[model], s=9, color=color, zorder=3, label=model)
        axis.set_xlabel(f"structure stream {title}")
        axis.grid(axis="x", color=GRID, lw=0.5)
        figures._letter(axis, letter, x=-0.03 if letter != "A" else -0.62)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([figures.SHORT_LABELS.get(task, task) for task in rows])
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    figure.tight_layout(w_pad=0.5, rect=(0, 0, 1, 1 - 0.25 / figure.get_figheight()))
    return figures._save(figure, path)


def _state(site: str, layer: float, head: float | None = None) -> str:
    """A state as the text names it: the site's macro name and its 0-indexed block, `embedding`, `final norm` or
    `structure stream`."""
    if not isinstance(site, str):
        return "--"
    if site in ("embedding", "final_norm", "structural"):
        return SITE_LABELS[site]
    label = r"\site{" + site.replace("_", r"\_") + "}"
    text = f"{label} {int(layer)}"
    return text if head is None else f"{text}, h{int(head)}"


def best_block_tables(best: pd.DataFrame, path: Path, tasks: Sequence[str] = CONCEPT_ORDER) -> Path:
    """A LaTeX table per model: for each task, the best state of each family (site and block), chosen on validation,
    with its test score, and the structure stream's test score where it was probed."""
    lines = []
    for model in MODEL_COLORS:
        table = best[best["model"] == model].set_index("task")
        blocks = int(table["blocks"].iloc[0])
        structure = "structure_score" in table
        lines += [r"\begin{table}[p]", r"\centering\scriptsize",
                  rf"\caption{{\textbf{{The best state of each family in {model}}} ({blocks} blocks, numbered from 0), chosen on validation, "
                  r"with its test score in parentheses. Stream states are the embedding, a block's output and the final norm. "
                  r"Heads are numbered from 0." + (r" The last column is the test score of the structure stream, ESMFold2's "
                                                   r"256-wide mixture of every hidden state, one state per model." if structure else "") + "}",
                  rf"\label{{tab:best_{model.lower().replace('-', '')}}}",
                  r"\begin{tabular}{@{}l l l l l" + (" r" if structure else "") + r"@{}}", r"\toprule",
                  r"Task & Residual stream & Attention & Feed-forward & Single head" + (" & Structure" if structure else "") + r" \\",
                  r"\midrule"]
        for task in tasks:
            if task not in table.index:
                continue
            row = table.loc[task]
            cells = []
            for family, _ in FAMILIES:
                site = row.get(f"{family}_site")
                if not isinstance(site, str):
                    cells.append("--")
                    continue
                layer = row[f"{family}_layer"]
                state = _state(site, layer, row["head"] if family == "head" else None)
                if family == "residual" and site == "resid_post":
                    state = f"block {int(layer)}"
                cells.append(f"{state} ({row[f'{family}_score']:.2f})")
            if structure:
                cells.append(f"{row['structure_score']:.2f}" if np.isfinite(row["structure_score"]) else "--")
            lines.append(f"{figures.TASK_LABELS.get(task, task)} & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
