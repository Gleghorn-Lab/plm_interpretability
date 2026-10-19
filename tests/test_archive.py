"""The survey's raw reruns staged into the archive's compacted layout read back through `overlay` exactly."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from tap_probing import archive


def write_run(folder: Path, task: str, level: str, pair: bool = False) -> pd.DataFrame:
    """A finished task folder as Protify writes it: two probes, the full test split (resample 0) and seven halves."""
    folder.mkdir(parents=True)
    generator = np.random.default_rng(24895)
    rows = [{"task": task, "family": "full", "probe_name": f"resid_post:{layer}", "resample": resample, "plm": "trained",
             "probe": "trained", "control": "original", "score": float(generator.normal(0.5, 0.05))}
            for layer in (1, 2) for resample in range(8)]
    scores = pd.DataFrame(rows)
    scores.to_parquet(folder / "scores.parquet", index=False)
    probes = pd.DataFrame({"task": task, "family": "full", "probe_name": ["resid_post:1", "resid_post:2"], "best_step": [10, 12]})
    probes.to_parquet(folder / "training.parquet", index=False)
    probes[["task", "family", "probe_name"]].assign(validation_score=[0.4, 0.45], validation_loss=[0.6, 0.55]).to_parquet(
        folder / "validation.parquet", index=False)
    pd.DataFrame({"x": [1.0]}).to_parquet(folder / "elementwise.parquet", index=False)
    if pair:
        pd.DataFrame({"metric": ["roc_auc"], "estimate": [0.61]}).to_parquet(folder / "ranking.parquet", index=False)
        pd.DataFrame({"metric": ["roc_auc"], "low": [0.59], "high": [0.63]}).to_parquet(folder / "intervals.parquet", index=False)
    (folder / "run.json").write_text(json.dumps({"task": {"name": task, "level": level}}), encoding="utf-8")
    return scores


def test_staged_survey_reads_back_through_overlay(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    scores = write_run(raw / "run_onehot" / "results" / "linear_mean" / "go_cc", "go_cc", "protein")
    write_run(raw / "run_onehot" / "results" / "linear" / "ppi", "ppi", "pair", pair=True)
    (raw / "run_onehot" / "results" / "linear_mean" / "unfinished").mkdir()
    catalog = archive.stage_survey(raw, tmp_path / "survey")

    staged = tmp_path / "survey" / "runs" / "run_onehot"
    compact = pd.read_parquet(staged / "linear_mean" / "go_cc" / "scores.parquet").set_index("probe_name")
    for name, group in scores.groupby("probe_name"):
        assert compact.loc[name, "full"] == pytest.approx(group.loc[group["resample"] == 0, "score"].iloc[0], abs=1e-6)
        assert compact.loc[name, "halves_mean"] == pytest.approx(group.loc[group["resample"] > 0, "score"].mean(), abs=1e-6)
        assert compact.loc[name, "halves"] == 7
    probes = pd.read_parquet(staged / "linear_mean" / "go_cc" / "probes.parquet")
    assert probes["validation_score"].tolist() == [0.4, 0.45] and probes["best_step"].tolist() == [10, 12]
    assert sorted(path.name for path in (staged / "linear" / "ppi").iterdir()) == [
        "intervals.parquet", "probes.parquet", "ranking.parquet", "run.json", "scores.parquet"]
    assert not (staged / "linear_mean" / "unfinished").exists()
    assert sorted(item["path"] for item in catalog["omitted"]) == [
        "run_onehot/results/linear/ppi/elementwise.parquet", "run_onehot/results/linear_mean/go_cc/elementwise.parquet"]

    runs = tmp_path / "runs"
    (runs / "run_onehot" / "results").mkdir(parents=True)
    archive.overlay(tmp_path / "survey", runs)
    summary = archive.model_summary([runs / "run_onehot"], "linear_mean").set_index("probe_name")
    full = scores[scores["resample"] == 0].set_index("probe_name")["score"]
    np.testing.assert_allclose(summary.loc[full.index, "full"].to_numpy(), full.to_numpy(), atol=1e-6)
    np.testing.assert_allclose(summary.loc[["resid_post:1", "resid_post:2"], "validation_score"].to_numpy(), [0.4, 0.45])
    with pytest.raises(FileExistsError):
        archive.overlay(tmp_path / "survey", runs)  # a staged file never replaces one already there


def test_snapshot_resumes_after_the_rate_limit_and_raises_other_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def flaky(repo, **kwargs):
        calls.append(kwargs["allow_patterns"])
        if len(calls) < 3:
            raise RuntimeError("HTTP status client error (429 Too Many Requests)")
        return str(kwargs["local_dir"])

    monkeypatch.setattr(archive, "snapshot_download", flaky)
    assert archive.snapshot("repo", "rev", tmp_path, ["runs/*"], pause=0.0) == tmp_path
    assert calls == [["runs/*"]] * 3

    def broken(repo, **kwargs):
        calls.append(None)
        raise RuntimeError("404 Not Found")

    monkeypatch.setattr(archive, "snapshot_download", broken)
    with pytest.raises(RuntimeError, match="404"):
        archive.snapshot("repo", "rev", tmp_path, ["runs/*"], pause=0.0)
    assert calls[-2:] == [["runs/*"], None]  # one attempt, no retry


def test_prot_param_targets_are_compacted_for_full_width_probes(tmp_path: Path) -> None:
    folder = tmp_path / "raw" / "run_norm_s24896" / "results" / "linear_mean" / "prot_param"
    write_run(folder, "prot_param", "protein")
    rows = [{"task": "prot_param", "family": family, "probe_name": "resid_post:1", "label_index": target, "resample": resample,
             "score": 0.1 * target + 0.01 * resample} for family in ("full", "head") for target in range(2) for resample in range(8)]
    pd.DataFrame(rows).to_parquet(folder / "elementwise.parquet", index=False)
    catalog = archive.stage_survey(tmp_path / "raw", tmp_path / "survey")
    targets = pd.read_parquet(tmp_path / "survey" / "runs" / "run_norm_s24896" / "linear_mean" / "prot_param" / "elementwise.parquet")
    assert targets["family"].tolist() == ["full", "full"] and targets["label_index"].tolist() == [0, 1]
    np.testing.assert_allclose(targets["full"].to_numpy(), [0.0, 0.1], atol=1e-6)
    np.testing.assert_allclose(targets["halves_mean"].to_numpy(), [0.04, 0.14], atol=1e-6)
    assert not any(item["path"].endswith("elementwise.parquet") for item in catalog["omitted"])
