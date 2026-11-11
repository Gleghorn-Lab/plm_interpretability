"""Run the native-unit passes of experiment 2026-11-06_esmc_native_units, inside the project's Docker image.

    python scripts/units_atlas.py inputs --root RUN --preregistration PREREG --expect-digest SHA256
    python scripts/units_atlas.py pin-sweep --root RUN --preregistration PREREG
    python scripts/units_atlas.py parity --root RUN --preregistration PREREG
    python scripts/units_atlas.py pass --root RUN --preregistration PREREG --experiment e2 [--pilot]
    python scripts/units_atlas.py score --root RUN --preregistration PREREG --experiment e2
    python scripts/units_atlas.py analyze --root RUN --preregistration PREREG
    python scripts/units_atlas.py missing --root RUN --preregistration PREREG
    python scripts/units_atlas.py dictionary --root RUN --preregistration PREREG --model esmc_6b [--pilot]
    python scripts/units_atlas.py causal --root RUN --preregistration PREREG [--pilot]
    python scripts/units_atlas.py near-misses --root RUN --preregistration PREREG
    python scripts/units_atlas.py exemplars --root RUN --preregistration PREREG
    python scripts/units_atlas.py steering --root RUN --preregistration PREREG [--pilot]
    python scripts/units_atlas.py concepts --root RUN --preregistration PREREG --model esmc_600 [--pilot]
    python scripts/units_atlas.py sparse-probes --root RUN --preregistration PREREG --model esmc_600
    python scripts/units_atlas.py attribution --root RUN --preregistration PREREG
    python scripts/units_atlas.py payoff --root RUN --preregistration PREREG --model esmc_600 [--pilot]
    python scripts/units_atlas.py taxonomy-panel --root RUN --preregistration PREREG --model esmc_600 [--pilot]
    python scripts/units_atlas.py subspaces --root RUN --preregistration PREREG --model esmc_600
    python scripts/units_atlas.py retest-inputs --root RUN --preregistration PREREG
    python scripts/units_atlas.py retest --root RUN --preregistration PREREG --model esm2_8m [--pilot]
    python scripts/units_atlas.py retest-esmc --root RUN --preregistration PREREG --model esmc_600 [--pilot]
    python scripts/units_atlas.py strength --root RUN --preregistration PREREG --model esmc_600 [--pilot]
    python scripts/units_atlas.py wiring --root RUN --preregistration PREREG --model esmc_600
    python scripts/units_atlas.py coalitions --root RUN --preregistration PREREG --model esmc_600 [--pilot]
    python scripts/units_atlas.py sae-labels --root RUN --preregistration PREREG
    python scripts/units_atlas.py figures --root RUN --preregistration PREREG [--figure 2]
    python scripts/units_atlas.py tables --root RUN --preregistration PREREG

`inputs` (CPU) downloads the pinned compendium release and the atlas study's held-out shards, draws the atlas sample
(`split.atlas_sample`), refuses a sample whose digest differs from the one the launch record holds, and writes each
protein's labels and local runs and the vocabulary's relations (`tap_probing.units.compendium`); the release holds no
sequence over ESMC's 2,046 residues. `pin-sweep` pins the
revisions of the ESMC-600 layer-27 sweep. `parity` checks, on 256 calibration proteins, that every planned SAE
reconstructs the state it is planned to read better than the neighbouring blocks' states, and whether the 6B layer-60
SAE's pinned and current revisions agree. `pass` loads the pinned model and SAEs and runs one planned pass
(`tap_probing.units.passes.run_pass`), resuming from its last checkpoint; with `--pilot` it times 20 batches after 3
warmup with `foundry.pilot` and stops. `score` writes each planned pair's per-pole specificity and, for `selection`
pairs, the atlas study's frozen labels. `analyze` takes the preregistered decisions H1 to H5 and H7 from the score
tables into `RUN/decisions.json`. `missing` (CPU, after E2 and E3 are scored) gives the test proteins the terms of their
close homologs among the release's validation and test proteins, scores every basis at the sites of H3 and H5 again with
them and takes those two decisions
again (`tap_probing.units.missing`). `dictionary` runs E8 at one model's site: each SAE's latents correlated with the
block's native and rotated poles over 2 million training residues (at ESMC-600's site also the native poles with one
another, for E10), then the sparsest-rotation search from three seeds; with `--pilot` it times the stream's pilot window
and two search steps. `causal` runs E7's ablations on ESMC-600 after
E2 is scored: H6 over block 26's labeled poles against magnitude-matched placebo poles, then the weakest against the
strongest activations; `analyze` takes H6 from its table once it exists. `near-misses` reads Swiss-Prot's mutagenesis
entries of the release's reviewed validation and test proteins in the same labeled poles against placebos (E7c),
`exemplars` draws Figure 4's
exemplar poles from the same pool by the frozen rule and reads their neurons' residue tracks on held-out carriers,
and `steering` steers ESMC-600
along the fitted manifold of a masked residue's surrounding hydropathy against a straight line (E7d). `concepts`
downloads version one's pinned concept datasets and pools one model's units over their proteins
(`tap_probing.units.concepts`), the extraction E6, E9, E10 and E11 read; `sparse-probes` runs E6 on it, `attribution` runs E9 on ESMC-600's once E2 is scored, and `payoff`
runs E11's XGBoost fits on one model's on the GPU, with `--pilot` timing 20 fits after 3 warmup. `taxonomy-panel` pools
one model's units over E10's panel of 12,000 atlas training proteins, one a cluster, which it draws once and keeps in
`RUN/panel/panel.parquet`; `subspaces` runs E10's reduced-rank readouts on the concept and panel extractions and, for
ESMC-600 after its `dictionary` stage, the co-firing blocks. `retest-inputs` builds E1's evaluation set with InterPLM's
own functions, its concept sub-categories enumerated over all of Swiss-Prot as its paper's run did
(`tap_probing.units.retest`), and `retest` scores ESM-2's arms on it after checking its counting, F1 and report
against InterPLM's functions; `analyze` takes H0 once both models have run. `retest-esmc`, E13, added after the
hypotheses were decided, scores an ESMC model's arms at its four depths on the same set, with the same check, into
`RUN/retest_esmc/<model>`; its `--model` also takes E13x's plans (`experiments.esmc_retest_plans`: `esmc_300`, every
ESMC-600 block as `esmc_600_depth`, the feed-forward-write SAEs as `esmc_600_mlp` and `esmc_6b_mlp`), each written to
`RUN/retest_esmc/<plan>`, and E13s's (`esmc_600_search`, `esmc_6b_search`), which first re-run E8's search at the
block and check its sample against E8's receipt. E14's three stages, exploratory and added on 2026-11-10, follow
Goodfire's neuron study: `strength` reads, on E1's set at a model's three-quarter block, each unit's best concept by
firing-strength band, native poles against three rotations (`tap_probing.units.strength`); `wiring` reads from the
weights alone how closely each unit's gate and up rows lie along one writing unit of the block before
(`tap_probing.units.wiring`); and `coalitions`, at one of E8's sites, fits each latent of the SAE on the feed-forward
write with 1 to 64 native poles on half of E8's residues and reads the fit on the other half, after checking its copy
shares against E8's (`tap_probing.units.coalitions`). `sae-labels` (CPU, after E3's scores and
`missing`) summarizes the compendium labels of Biohub's ESMC-6B layer-60 SAE latents and compares them with Biohub's
descriptions by version one's estimator, for Supplement S2 (`tap_probing.units.sae_labels`). `figures` (CPU, after the rest) draws the
paper's figures from the run's outputs into `RUN/figures` (`tap_probing.units.figures`), or one with `--figure`, and
`tables` writes its tables, every preregistered decision among them, into `RUN/tables` (`tap_probing.units.tables`).
Every stage leaves a receipt.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import json
import math
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import yaml

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from torch import Tensor, nn

from foundry.pilot import Pilot
from tap_probing.units import atlas, atlas_set, experiments, passes


PARITY_PROTEINS = 256
FAMILY_CHECK_PROTEINS = 32  # E13x: the first validation proteins on which an all-layer SAE family's read position is checked
PARITY_CHUNK = 8192  # residues an SAE reads at once in the parity check: 4.3 GB a 131,072-latent activation in float32
RETEST_MEMORY_SHARE = 0.5  # of the memory free when E1's label conversion starts, for its workers together
Preregistration = dict[str, Any]  # the parsed preregistration.yaml


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    hashed = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            hashed.update(chunk)
    return hashed.hexdigest()


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _release(root: Path, preregistration: Preregistration) -> Path:
    """The pinned compendium release under `RUN/source` (private: the launcher's HF_TOKEN reads it)."""
    from huggingface_hub import snapshot_download
    from tap_probing.units import compendium

    inputs = preregistration["data"]["atlas_inputs"]
    return Path(snapshot_download(inputs["repo"], repo_type="dataset", revision=inputs["revision"],
                                  allow_patterns=list(compendium.RELEASE_FILES), local_dir=root / "source"))


def stage_inputs(root: Path, preregistration: Preregistration, expect_digest: str) -> None:
    """The atlas set: the release's proteins sampled by `split.atlas_sample`, refused unless the sample's digest is the
    launch record's, then their labels, local runs and the vocabulary's relations (`tap_probing.units.compendium`)."""
    from huggingface_hub import snapshot_download
    from tap_probing.units import compendium, split

    source = _release(root, preregistration)
    examined_pin = preregistration["data"]["examined"]
    examined = compendium.read_examined(Path(snapshot_download(
        examined_pin["repo"], repo_type="dataset", revision=examined_pin["revision"],
        allow_patterns=list(compendium.EXAMINED_FILES), local_dir=root / "examined")))
    sample = split.atlas_sample(compendium.read_pool(source), examined)
    digest = split.assignment_digest(sample)
    if digest != expect_digest:
        raise ValueError(f"The sample's digest {digest} differs from the launch record's {expect_digest}.")
    words, hierarchy = compendium.read_vocabulary(source)
    expected = preregistration["data"]["atlas_inputs"]["tokens"]
    if words.size != int(expected):
        raise ValueError(f"The release has {words.size} tokens, the preregistration {expected}.")
    chosen = set(sample["protein_id"])
    proteins = compendium.read_proteins(source, chosen, atlas_set.PROTEIN_COLUMNS)
    lengths = dict(zip(proteins["protein_id"], proteins["sequence"].str.len(), strict=True))
    signatures = compendium.signature_spans(compendium.read_signature_assertions(source, chosen), lengths,
                                            words.terms["term_id"])
    located, sites, disorder = compendium.read_residue_tables(source, chosen)
    local = compendium.local_runs(located, sites, disorder, signatures.spans, words)
    labels = compendium.global_labels(compendium.read_rows(source, chosen), words)
    built = atlas_set.build_atlas_set(sample, proteins, labels, local, words)
    passes.write_inputs(built, words, compendium.ancestor_matrix(words, hierarchy), root / "inputs",
                        release=preregistration["data"]["atlas_inputs"], examined_accessions=len(examined),
                        signature_spans=len(signatures.spans), signature_spans_off_sequence=signatures.off_sequence,
                        signature_spans_outside_vocabulary=signatures.outside_vocabulary,
                        signature_terms_outside_vocabulary=signatures.terms_outside_vocabulary,
                        proteins_sha256=_sha256(source / "tables" / "proteins.parquet"))
    _log(f"atlas set {len(built.proteins)} proteins, digest {built.digest}; of the signature spans {len(signatures.spans)} kept, "
         f"{signatures.off_sequence} off their sequence, {signatures.outside_vocabulary} of "
         f"{signatures.terms_outside_vocabulary} terms outside the vocabulary")


def stage_pin_sweep(root: Path, preregistration: Preregistration) -> None:
    from huggingface_hub import HfApi

    entry = preregistration["saes"]["esmc_600_layer27_sweep"]
    api = HfApi()
    pins = {experiments.sweep_name(k, codebook): api.model_info(entry["repo_pattern"].format(k=k, codebook=codebook)).sha
            for k in entry["k"] for codebook in entry["codebook"]}
    _json(root / "pins" / "sweep.json", pins)
    _log(f"pinned {len(pins)} sweep SAEs")


def _specs(root: Path, preregistration: Preregistration) -> dict[str, experiments.SaeSpec]:
    pins_path = root / "pins" / "sweep.json"
    pins = json.loads(pins_path.read_text(encoding="utf-8")) if pins_path.exists() else None
    return experiments.sae_specs(preregistration["saes"], pins)


def _load_model(preregistration: Preregistration, name: str, device: torch.device, *, random_init: bool = False) -> nn.Module:
    from fastplms.models.esm_plusplus.modeling_esm_plusplus import ESMplusplusConfig, ESMplusplusForMaskedLM
    from foundry.embedding.canonical import random_init_model

    spec = preregistration["models"][name]
    if random_init:
        config = ESMplusplusConfig.from_pretrained(spec["repo"], revision=spec["revision"])
        model = random_init_model(config, int(preregistration["seeds"][0])).to(torch.bfloat16)
    else:
        model = ESMplusplusForMaskedLM.from_pretrained(spec["repo"], revision=spec["revision"], dtype=torch.bfloat16,
                                                       attn_implementation="sdpa")
    blocks = len(atlas.transformer_blocks(model))
    if blocks != int(spec["blocks"]) or atlas.transformer_units(model) != int(spec["f"]):
        raise ValueError(f"{name} has {blocks} blocks and {atlas.transformer_units(model)} units, not the pinned shape.")
    return model.eval().to(device)


def _load_sae(spec: experiments.SaeSpec, device: torch.device) -> nn.Module:
    from fastplms.models.esm_plusplus.modeling_esm_plusplus_sae import load_esmc_sae_layers

    if spec.revision is None:
        raise ValueError(f"{spec.name} has no pinned revision; run pin-sweep first.")
    return load_esmc_sae_layers(spec.repo, [spec.layer], revision=spec.revision, device=device, dtype=torch.float32)[spec.layer]


def _weights_sha256(model: nn.Module) -> str:
    """SHA-256 of every parameter's bytes in order, the identity of a randomly drawn network."""
    hashed = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        hashed.update(name.encode())
        hashed.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return hashed.hexdigest()


def _reconstruction_loss(sae: nn.Module, states: Tensor, device: torch.device) -> float:
    """An SAE's mean per-residue reconstruction loss over `states` (r, d), PARITY_CHUNK residues at a time, since a
    131,072-latent SAE's activations for every residue of the parity proteins outgrow the GPU; each residue is
    standardized alone, so the chunks change nothing but the order of the sum."""
    # states: (r, d)
    total = 0.0
    with torch.no_grad():
        for start in range(0, states.shape[0], PARITY_CHUNK):
            chunk = states[start:start + PARITY_CHUNK].to(device).float()  # (c, d)
            total += float(sae(chunk, compute_reconstruction_loss=True).reconstruction_loss.double().sum())
    return total / states.shape[0]


def _read_positions(report: dict[str, Any], specs: dict[str, experiments.SaeSpec]) -> dict[str, dict[str, object]]:
    """E0's read-position verdict for each repository family, the SAEs one pattern trained on one block's stream or write
    (`esmc_600_layer27` over its sweep, `esmc_6b_mlp`), from its members' losses in `report`: a family reads the block
    where its members' mean reconstruction loss is lowest, and one read elsewhere stops the run. A member whose own lowest
    loss sits on a neighbouring block is listed, not decisive: a weak member (k = 16 over 8,192 latents) reconstructs two
    adjacent streams about equally, while the family's position is one fact of how its SAEs were trained."""
    families: dict[str, list[str]] = {}
    for name in report:
        families.setdefault(name.rsplit("_k", 1)[0], []).append(name)
    verdicts: dict[str, dict[str, object]] = {}
    for family, members in sorted(families.items()):
        planned = {(specs[name].block, specs[name].reads) for name in members}
        if len(planned) != 1:
            raise ValueError(f"{family}'s members plan different read positions: {sorted(planned)}.")
        ((block, reads),) = planned
        blocks = sorted(set.intersection(*(set(report[name]["losses"]) for name in members)))
        mean = {candidate: float(np.mean([report[name]["losses"][candidate] for name in members])) for candidate in blocks}
        best = min(mean, key=mean.get)
        verdicts[family] = {"planned_block": block, "reads": reads, "members": members, "mean_losses": mean,
                            "best_block": best, "members_best_elsewhere": [name for name in members
                                                                          if report[name]["best_block"] != block]}
        _log(f"{family}: mean losses {mean}, best block {best}; members best elsewhere "
             f"{verdicts[family]['members_best_elsewhere']}")
        if best != block:
            raise ValueError(f"{family} reconstructs block {best}'s {reads} best, not its planned block {block}.")
    return verdicts


def stage_parity(root: Path, preregistration: Preregistration) -> None:
    """Every planned SAE's reconstruction loss on its planned state and on the neighbouring blocks' states, and each
    repository family's read position from them (`_read_positions`)."""
    from huggingface_hub import HfApi

    device = torch.device("cuda")
    proteins = pd.read_parquet(root / "inputs" / "atlas_set.parquet", columns=["sequence", "split"])
    sequences = atlas_set.calibration_sequences(proteins)[:PARITY_PROTEINS]
    specs = _specs(root, preregistration)
    planned = ({name for plan in experiments.experiment_plans().values() for name in plan.saes}
               | {name for site in experiments.dictionary_sites().values() for name in site.saes})
    report: dict[str, Any] = {}
    for model_name in ("esmc_300", "esmc_600", "esmc_6b"):
        names = sorted(name for name in planned if name.startswith(model_name + "_") and specs[name].revision is not None)
        if not names:
            continue
        model = _load_model(preregistration, model_name, device)
        last = int(preregistration["models"][model_name]["blocks"]) - 1
        candidates = sorted({min(max(specs[name].block + shift, 0), last) for name in names for shift in (-1, 0, 1)})
        keep = np.ones(sum(len(sequence) for sequence in sequences), dtype=bool)  # (r_total,)
        states = atlas.calibration_states(model, model.tokenizer, sequences, candidates, keep, stream_blocks=candidates,
                                          write_blocks=candidates, device=device)
        for name in names:
            spec, sae = specs[name], _load_sae(specs[name], device)
            losses = {}
            for block in (block for block in candidates if abs(block - spec.block) <= 1):
                losses[block] = _reconstruction_loss(sae, getattr(states[block], spec.reads), device)
            best = min(losses, key=losses.get)
            report[name] = {"planned_block": spec.block, "reads": spec.reads, "losses": losses, "best_block": best}
            _log(f"{name}: losses {losses}, best block {best}")
            del sae
        del model, states
        torch.cuda.empty_cache()
    families = _read_positions(report, specs)
    report["families"] = families
    pinned = specs["esmc_6b_layer60_k64_16384"]
    head = HfApi().model_info(pinned.repo).sha
    current = experiments.SaeSpec(pinned.name, pinned.repo, head, pinned.layer, pinned.block, pinned.reads)
    first, second = _load_sae(pinned, torch.device("cpu")), _load_sae(current, torch.device("cpu"))
    report["esmc_6b_layer60_current_revision"] = head
    report["esmc_6b_layer60_revisions_equal"] = all(
        torch.equal(getattr(first, name), getattr(second, name)) for name in ("W_enc", "W_dec", "b_dec", "idf", "max"))
    _json(root / "parity" / "receipt.json", report)


def stage_pass(root: Path, preregistration: Preregistration, experiment: str, pilot_run: bool) -> None:
    started = time.perf_counter()
    plan = experiments.experiment_plans()[experiment]
    output = root / ("pilots" if pilot_run else "passes") / experiment
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = passes.PassInputs.read(root / "inputs")
    pilot = None
    if pilot_run:
        lengths = atlas_set.pass_proteins(inputs.frame, subset=plan.proteins).ordered().lengths  # (n,)
        batches = atlas.length_batches(lengths)
        pilot = Pilot(total_steps=len(batches), items=lambda index: int(lengths[batches[index]].sum()), item_unit="residues",
                      budget_seconds=plan.budget_hours * 3600, receipt=output / "pilot.json", started=started)
    specs = _specs(root, preregistration)
    model = _load_model(preregistration, plan.model, device, random_init=plan.random_init)
    trained = None
    if plan.rates_from:
        trained = json.loads((root / "passes" / plan.rates_from / "receipt.json").read_text(encoding="utf-8"))
    saes = {name: _load_sae(specs[name], device) for name in plan.saes if trained is None or name not in plan.references}
    given_rates = None
    if trained is not None:
        # The trained pass may calibrate references this one does not read (e2 also reads the 16,384-latent SAE).
        saved = torch.load(root / "passes" / plan.rates_from / "calibration.pt", weights_only=True)["rates"]
        given_rates = {name: saved[name] for name in plan.references}
    receipt = passes.run_pass(
        model, saes, {name: (specs[name].block, specs[name].reads) for name in plan.references}, plan, inputs, output, device,
        pilot=pilot, log=_log, given_rates=given_rates,
        given_validation_rates=None if trained is None else trained["validation_reference_rates"],
    )
    receipt.update(model=preregistration["models"][plan.model], random_init=plan.random_init,
                   weights_sha256=_weights_sha256(model) if plan.random_init else None,
                   saes={name: vars(specs[name]) for name in plan.saes}, torch=torch.__version__,
                   device=torch.cuda.get_device_name(device))
    _json(output / ("pilot_receipt.json" if pilot_run else "receipt.json"), receipt)
    rejected = [key for key, check in receipt.get("acceptance", {}).items() if not check["accepted"]]
    _log(f"pass {experiment}: {receipt['timing']['residues_per_second']:,.0f} residues/s; rejected matches {rejected}")


def stage_score(root: Path, experiment: str) -> None:
    plan = experiments.experiment_plans()[experiment]
    output = root / "passes" / experiment
    passes.score_pass(plan, passes.PassInputs.read(root / "inputs"), output, device="cuda" if torch.cuda.is_available() else "cpu")
    _json(output / "scores" / "receipt.json", {"experiment": experiment, "pairs": [planned.key for planned in plan.plan]})
    _log(f"scored {experiment}")


def _rejected(root: Path, experiment: str, keys: list[str]) -> list[str]:
    acceptance = json.loads((root / "passes" / experiment / "receipt.json").read_text(encoding="utf-8"))["acceptance"]
    return [key for key in keys if key in acceptance and not acceptance[key]["accepted"]]


def _decide(decisions: list[dict[str, object]], name: str, site: str, rejected: list[str],
            make: Callable[[], Any]) -> None:
    """Append one decision, or its absence when it needs a pair whose rate match was rejected."""
    if rejected:
        decisions.append({"hypothesis": name, "site": site, "taken": False, "rejected_matches": rejected})
        return
    decision = make()
    decisions.append({"hypothesis": name, "site": site, "taken": True, "passed": decision.passed,
                      "details": decision.details})


def stage_analyze(root: Path) -> None:
    """H1 to H5 and H7 from the score tables, and H6 from E7's ablation table; a decision that needs a pair whose rate
    match was rejected is not taken."""
    from tap_probing.units import hypotheses

    decisions: list[dict[str, object]] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decide = functools.partial(_decide, decisions)
    for site, (experiment, block, reference) in experiments.HYPOTHESIS_SITES.items():
        keys = experiments.site_keys(block, reference)
        native, sign, residual, sae, rotated = keys.native, keys.sign, keys.residual, keys.sae, list(keys.rotated)
        tables = {key: experiments.score_table(root, experiment, key) for key in (native, sign, residual, sae, *rotated)}
        rotations = [tables[key] for key in rotated]
        # Each decision on the gate-tagged poles, then on the sign-only reading of the same products, which splits z by
        # sign alone as the rotated and residual bases are split, so a lead the gate's tag alone gives shows apart.
        for suffix, units in (("", native), ("_sign", sign)):
            decide("H1" + suffix, site, _rejected(root, experiment, [units, residual]),
                   lambda: hypotheses.h1_native_over_residual(tables[units], tables[residual]))
            decide("H2" + suffix, site, _rejected(root, experiment, [units, *rotated]),
                   lambda: hypotheses.h2_native_privileged(tables[units], rotations))
            decide("H3" + suffix, site, _rejected(root, experiment, [units]),
                   lambda: hypotheses.h3_native_near_sae(tables[units], tables[sae]))
            decide("H5" + suffix, site, _rejected(root, experiment, [units, *rotated]),
                   lambda: hypotheses.h5_generalization(tables[units], rotations))
    inputs = passes.PassInputs.read(root / "inputs")
    kinds = hypotheses.input_kinds(root / "inputs")
    plans = experiments.experiment_plans()
    for trained_name, random_name in experiments.RANDOM_INIT_PASSES.items():
        control = plans[random_name]
        keys = [planned.key for planned in control.plan]
        rejected = _rejected(root, random_name, keys) + _rejected(root, trained_name, keys)

        def pretraining(trained_name: str = trained_name, control: experiments.PassPlan = control,
                        keys: list[str] = keys) -> hypotheses.Decision:
            trained = passes.subset_scores(plans[trained_name], int(control.proteins), inputs, root / "passes" / trained_name,
                                           keys, device=device)
            random = pd.concat([experiments.score_table(root, control.name, key) for key in keys], ignore_index=True)
            return hypotheses.h4_pretraining(pd.concat(trained.values(), ignore_index=True), random, kinds)

        decide("H4", control.model, rejected, pretraining)
    scale = experiments.SCALE_SITES
    decide("H7", "three models", [key for _, experiment, key in scale for key in _rejected(root, experiment, [key])],
           lambda: hypotheses.h7_scale([(model, experiments.score_table(root, experiment, key))
                                        for model, experiment, key in scale]))
    ablations = root / "causal" / "h6.parquet"
    if ablations.exists():
        decide("H6", f"{experiments.CAUSAL_MODEL} block {experiments.CAUSAL_BLOCK}", _rejected(root, "e2", [_causal_key()[0]]),
               lambda: hypotheses.h6_used(pd.read_parquet(ablations)))
    else:
        decisions.append({"hypothesis": "H6", "site": experiments.CAUSAL_MODEL, "taken": False, "reason": "E7 has not run"})
    retests = [root / "retest" / model for model in experiments.RETEST_BUDGET_HOURS]
    if all((folder / "e1_summary.parquet").exists() for folder in retests):
        summary = pd.concat([pd.read_parquet(folder / "e1_summary.parquet") for folder in retests], ignore_index=True)
        per_concept = pd.concat([pd.read_parquet(folder / "e1_per_concept.parquet") for folder in retests], ignore_index=True)
        decide("H0", "ESM-2 8M and 650M", [], lambda: hypotheses.h0_mlp_over_residual(summary, per_concept))
    else:
        decisions.append({"hypothesis": "H0", "site": "ESM-2", "taken": False, "reason": "E1 has not run on both models"})
    _json(root / "decisions.json", decisions)
    for entry in decisions:
        _log(f"{entry['hypothesis']} at {entry['site']}: " + (f"passed={entry['passed']}" if entry["taken"] else "not taken"))


def _scored_basis(root: Path, experiment: str, rows: np.ndarray, own: Any, propagated: Any,
                  planned: atlas.PlannedBasis) -> pd.DataFrame:
    """One basis's poles on the test proteins before and after inheritance (`missing.basis_scores`)."""
    # rows: (n,) the pass's test rows; own, propagated: (n, t)
    from tap_probing.units import missing

    maxima = np.load(root / "passes" / experiment / "maxima" / f"{planned.stem}.npy", mmap_mode="r")  # (n_pass, p)
    return missing.basis_scores(np.asarray(maxima[rows]), own, propagated,
                                experiments.score_table(root, experiment, planned.key))


def stage_missing(root: Path) -> None:
    """The missing-annotation correction (`tap_probing.units.missing`), CPU only, after E2 and E3 are scored: one search
    of the test proteins against the release's validation and test proteins, every basis at each site of H3 and H5 scored
    before and after the test proteins inherit their homologs' terms, and H3 and H5 taken again on the propagated held-out
    AUROC."""
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import asdict, replace
    from foundry.datasets.homology import matching_references
    from tap_probing.units import compendium, hypotheses, missing
    from tap_probing.units.observed import TermLabels

    started = time.perf_counter()
    output = root / "missing"
    source = root / "source"
    inputs = passes.PassInputs.read(root / "inputs")
    splits = pd.read_parquet(source / "tables" / "split.parquet", columns=["protein_id", "split"])
    split_of = dict(zip(splits["protein_id"], splits["split"], strict=True))
    targets = splits.loc[splits["split"].isin(missing.TARGET_SPLITS), "protein_id"]
    entries = compendium.read_proteins(source, set(targets), ("protein_id", "sequence"))
    test = inputs.frame[inputs.frame["split"] == "test"]
    threads = int(os.environ.get("OMP_NUM_THREADS") or os.cpu_count() or 1)
    search = replace(missing.SEARCH, threads=threads)
    matches = matching_references(dict(zip(test["protein_id"], test["sequence"], strict=True)),
                                  dict(zip(entries["protein_id"], entries["sequence"], strict=True)), output / "search",
                                  search, minimum_identity=missing.MIN_IDENTITY)
    homologs = np.array([sum(match != query for match in matches[query]) for query in test["protein_id"]])  # (q,)
    _log(f"{int((homologs > 0).sum())} of {len(test)} test proteins have a homolog at {missing.MIN_IDENTITY} identity")
    matched = {match for query in test["protein_id"] for match in matches[query] if match != query}
    released = compendium.read_rows(source, matched)
    annotations = dict(zip(released["protein_id"], released["tokens"], strict=True))
    inherit = missing.inheritable(pd.read_parquet(root / "inputs" / "terms.parquet"))
    plans = experiments.experiment_plans()
    decisions: list[dict[str, object]] = []
    sites: dict[str, object] = {}
    for site, (experiment, block, reference) in experiments.HYPOTHESIS_SITES.items():
        plan = plans[experiment]
        proteins = atlas_set.pass_proteins(inputs.frame, subset=plan.proteins).ordered()
        rows = np.flatnonzero(proteins.split == "test")  # (n,)
        accessions = [str(accession) for accession in np.asarray(proteins.accession)[rows]]
        order = inputs.frame.set_index("accession").loc[accessions]
        own = TermLabels.from_frame(order, inputs.vocabulary_size, inputs.unit)  # (n, t)
        propagated = missing.with_inherited(own, missing.inherited_positives(order["protein_id"].tolist(), matches,
                                                                             annotations, inherit))
        planned = {item.key: item for item in plan.plan if item.basis.block == block}
        score = functools.partial(_scored_basis, root, experiment, rows, own, propagated)
        with ThreadPoolExecutor(max_workers=max(1, min(len(planned), threads // 4))) as pool:
            scored = dict(zip(planned, pool.map(score, planned.values()), strict=True))
        for key, table in scored.items():
            path = missing.table_path(root, experiment, key)
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(path, index=False)
        keys = experiments.site_keys(block, reference)
        native, sae, rotated = keys.native, keys.sae, list(keys.rotated)
        after = {key: experiments.score_table(root, experiment, key).assign(held_out_auroc=scored[key]["auroc_after"].to_numpy())
                 for key in (native, sae, *rotated)}
        _decide(decisions, "H3", site, _rejected(root, experiment, [native]),
                functools.partial(hypotheses.h3_native_near_sae, after[native], after[sae]))
        _decide(decisions, "H5", site, _rejected(root, experiment, [native, *rotated]),
                functools.partial(hypotheses.h5_generalization, after[native], [after[key] for key in rotated]))
        sites[site] = {"test_proteins": len(rows), "inherited_pairs": int(propagated.positives.nnz - own.positives.nnz),
                       "terms_changed": int(missing.changed_terms(own, propagated).sum()),
                       "bases": missing.missing_summary(scored)}
    found = homologs > 0  # (q,)
    _json(output / "receipt.json", {
        "search": asdict(search), "min_identity": missing.MIN_IDENTITY, "target_splits": list(missing.TARGET_SPLITS),
        "uninherited_aspects": list(missing.UNINHERITED_ASPECTS), "queries": len(test), "entries": len(entries),
        "matched_by_split": {str(name): int(count) for name, count in
                             pd.Series([split_of[match] for match in matched], dtype=object).value_counts().items()},
        "queries_with_homologs": int(found.sum()), "median_homologs": float(np.median(homologs[found])) if found.any() else 0.0,
        "sites": sites, "decisions_after_inheritance": decisions, "seconds": time.perf_counter() - started,
    })
    for entry in decisions:
        _log(f"{entry['hypothesis']} at {entry['site']} after inheritance: "
             + (f"passed={entry['passed']}" if entry["taken"] else "not taken"))


def stage_dictionary(root: Path, preregistration: Preregistration, model_name: str, pilot_run: bool) -> None:
    """E8 at one site: the dictionary test and the sparsest-rotation search (`passes.run_dictionary`)."""
    started = time.perf_counter()
    site = experiments.dictionary_sites()[model_name]
    output = root / "pilots" / f"dictionary_{model_name}" if pilot_run else root / "dictionary" / model_name
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = passes.PassInputs.read(root / "inputs")
    pilot = None
    if pilot_run:
        lengths = atlas_set.residue_budget_proteins(inputs.frame, passes.DICTIONARY_RESIDUES).ordered().lengths  # (n,)
        batches = atlas.length_batches(lengths, passes.DICTIONARY_TOKEN_BUDGET)
        pilot = Pilot(total_steps=len(batches), items=lambda index: int(lengths[batches[index]].sum()), item_unit="residues",
                      budget_seconds=site.budget_hours * 3600, receipt=output / "pilot.json", started=started)
    specs = _specs(root, preregistration)
    model = _load_model(preregistration, model_name, device)
    saes = {name: _load_sae(specs[name], device) for name in site.saes}
    receipt = passes.run_dictionary(model, saes, {name: specs[name].reads for name in site.saes}, site.block, inputs, output,
                                    device, cofiring=site.cofiring, pilot=pilot, log=_log)
    receipt.update(model=preregistration["models"][model_name], saes={name: vars(specs[name]) for name in site.saes},
                   torch=torch.__version__, device=torch.cuda.get_device_name(device))
    _json(output / ("pilot_receipt.json" if pilot_run else "receipt.json"), receipt)


def _causal_key() -> tuple[str, str]:
    """The E2 pair whose labels E7 ablates, and its file stem."""
    key = f"block{experiments.CAUSAL_BLOCK}/native@{experiments.sweep_name(64, 8192)}"
    return key, {planned.key: planned.stem for planned in experiments.experiment_plans()["e2"].plan}[key]


def stage_causal(root: Path, preregistration: Preregistration, pilot_run: bool) -> None:
    """E7's ablations on ESMC-600: H6 over block 26's labeled poles, then the weak-against-strong test."""
    from tap_probing.units import ablation, causal

    started = time.perf_counter()
    output = root / ("pilots" if pilot_run else "causal")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = passes.PassInputs.read(root / "inputs")
    _, stem = _causal_key()
    scores_dir = root / "passes" / "e2" / "scores"
    labels = pd.read_parquet(scores_dir / f"{stem}.labels_local.parquet")
    scores = pd.read_parquet(scores_dir / f"{stem}.parquet")
    model = _load_model(preregistration, experiments.CAUSAL_MODEL, device)
    calibration = atlas_set.calibration_sequences(inputs.frame)
    keep = atlas.residue_sample(sum(len(sequence) for sequence in calibration))  # (r_calibration,)
    states = atlas.calibration_states(model, model.tokenizer, calibration, [experiments.CAUSAL_BLOCK], keep,
                                      device=device)[experiments.CAUSAL_BLOCK]
    values = causal.nonfiring_means(states.gate, states.up).to(device)  # (p,)
    pilot = None
    if pilot_run:
        pilot = Pilot(total_steps=len(ablation.labeled_poles(labels, scores)), items=lambda index: 1, item_unit="poles",
                      budget_seconds=experiments.CAUSAL_BUDGET_HOURS * 3600, receipt=output / "pilot.json", started=started)
    table = ablation.run_h6(model, experiments.CAUSAL_BLOCK, labels, scores, inputs.frame, values, device, pilot=pilot,
                            log=_log)
    table.to_parquet(output / "h6.parquet", index=False)
    receipt = {"model": preregistration["models"][experiments.CAUSAL_MODEL], "block": experiments.CAUSAL_BLOCK,
               "labels": stem, "statuses": {str(status): int(count) for status, count in table["status"].value_counts().items()},
               "torch": torch.__version__, "device": torch.cuda.get_device_name(device),
               "seconds": time.perf_counter() - started}
    if not pilot_run:
        test = inputs.frame[inputs.frame["split"] == "test"]
        picked = np.sort(np.random.default_rng(ablation.SEED).choice(len(test), size=min(ablation.READOUT_PROTEINS, len(test)),
                                                                     replace=False))
        _json(output / "weak_strong.json", ablation.weak_strong(model, calibration, test["sequence"].iloc[picked].tolist(),
                                                                device, log=_log))
        receipt["seconds"] = time.perf_counter() - started
    _json(output / ("pilot_receipt.json" if pilot_run else "receipt.json"), receipt)


def _download(url: str, target: Path, attempts: int = 3) -> dict[str, object]:
    """A UniProt stream saved once to `target`, with the release UniProt names in its header recorded beside it.

    A stream cut short starts over, up to `attempts` times in all: all of Swiss-Prot's stream was cut once at 03:36 UTC
    on 2026-11-07. The last failure raises.
    """
    import http.client
    import shutil
    import urllib.error
    import urllib.request

    record = target.with_name(target.name + ".release.json")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
                    release = response.headers.get("X-UniProt-Release")
                    shutil.copyfileobj(response, handle)
                break
            except (http.client.IncompleteRead, urllib.error.URLError, ConnectionError) as error:
                if attempt == attempts:
                    raise
                _log(f"UniProt stream cut short ({error!r}); starting over, attempt {attempt + 1} of {attempts}")
                time.sleep(30 * attempt)
        partial.replace(target)
        _json(record, {"release": release, "url": url})
    return {**json.loads(record.read_text(encoding="utf-8")), "sha256": _sha256(target)}


def _held_out_reviewed(source: Path, splits: Iterable[str]) -> pd.DataFrame:
    """Every reviewed release protein of some splits with its accessions, split, cluster, sequence and local positive
    runs (an empty run column where it has none), as `near_misses.held_out_substitutions` reads them."""
    from tap_probing.units import compendium

    pool = compendium.read_pool(source)
    held = pool[pool["split"].isin(list(splits)) & pool["reviewed"].astype(bool)]
    chosen = set(held["protein_id"])
    words, _ = compendium.read_vocabulary(source)
    sequences = compendium.read_proteins(source, chosen, ("protein_id", "sequence"))
    lengths = dict(zip(sequences["protein_id"], sequences["sequence"].str.len(), strict=True))
    spans = compendium.signature_spans(compendium.read_signature_assertions(source, chosen), lengths, words.terms["term_id"]).spans
    local = compendium.local_runs(*compendium.read_residue_tables(source, chosen), spans, words)
    proteins = (held.merge(sequences, on="protein_id", validate="one_to_one")
                .merge(local, on="protein_id", how="left", validate="one_to_one")
                .sort_values("protein_id", kind="mergesort").reset_index(drop=True))
    empty = np.zeros(0, dtype=np.int64)
    for column in compendium.LOCAL_COLUMNS:
        proteins[column] = [cell if isinstance(cell, np.ndarray) else empty for cell in proteins[column]]
    return proteins


def stage_near_misses(root: Path, preregistration: Preregistration) -> None:
    """E7c on ESMC-600's block 26: Swiss-Prot's mutagenesis entries of the release's reviewed validation and test
    proteins, read in the poles E2 labels with annotated local terms against placebo poles
    (`tap_probing.units.near_misses`)."""
    from tap_probing.units import ablation, near_misses

    started = time.perf_counter()
    output = root / "causal" / "near_misses"
    source = _download(near_misses.UNIPROT_STREAM, output / "uniprot_mutagenesis.tsv.gz")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = passes.PassInputs.read(root / "inputs")
    proteins = _held_out_reviewed(root / "source", near_misses.HELD_OUT)
    entries = pd.read_csv(output / "uniprot_mutagenesis.tsv.gz", sep="\t")
    substitutions, filters = near_misses.held_out_substitutions(entries, proteins)
    _, stem = _causal_key()
    scores_dir = root / "passes" / "e2" / "scores"
    scores = pd.read_parquet(scores_dir / f"{stem}.parquet")
    labeled = ablation.labeled_poles(pd.read_parquet(scores_dir / f"{stem}.labels_local.parquet"), scores, limit=len(scores))
    annotated = labeled[labeled["term"] < inputs.vocabulary_size]  # the release's terms, the intrinsic ones aside
    covering = near_misses.covering_poles(substitutions, proteins, annotated)
    inside = np.array([len(poles) > 0 for poles in covering], dtype=bool)  # (m,)
    substitutions = substitutions[inside].reset_index(drop=True)
    covering = [poles for poles, kept in zip(covering, inside, strict=True) if kept]
    if substitutions.empty:
        raise ValueError("No held-out substitution lies inside a labeled pole's annotated term.")
    model = _load_model(preregistration, experiments.CAUSAL_MODEL, device)
    wild_sequences = [proteins["sequence"].iloc[row] for row in substitutions["row"]]
    sites = substitutions["site"].tolist()
    mutant_sequences = [sequence[:site] + mutant + sequence[site + 1:]
                        for sequence, site, mutant in zip(wild_sequences, sites, substitutions["mutant"], strict=True)]
    wild = near_misses.site_poles(model, experiments.CAUSAL_BLOCK, wild_sequences, sites, device)  # (m, p)
    mutant = near_misses.site_poles(model, experiments.CAUSAL_BLOCK, mutant_sequences, sites, device)  # (m, p)
    table = near_misses.near_miss_table(substitutions, covering, wild, mutant)
    table.to_parquet(output / "near_misses.parquet", index=False)
    summary = near_misses.near_miss_summary(table)
    _json(output / "receipt.json", {
        "model": preregistration["models"][experiments.CAUSAL_MODEL], "block": experiments.CAUSAL_BLOCK, "labels": stem,
        "uniprot": source, "held_out_reviewed": len(proteins), "filters": filters, "labeled_poles": len(annotated),
        "inside_a_term": len(substitutions),
        "summary": summary, "torch": torch.__version__, "device": torch.cuda.get_device_name(device),
        "seconds": time.perf_counter() - started})
    labeled_auroc, placebo_auroc = summary["labeled_auroc"], summary["placebo_auroc"]
    _log(f"E7c: {summary['tested']} substitutions tested; loss against no effect, labeled poles AUROC "
         f"{labeled_auroc['estimate']:.3f} [{labeled_auroc['low']:.3f}, {labeled_auroc['high']:.3f}], placebos "
         f"{placebo_auroc['estimate']:.3f} [{placebo_auroc['low']:.3f}, {placebo_auroc['high']:.3f}]")


def stage_exemplars(root: Path, preregistration: Preregistration) -> None:
    """Figure 4's exemplar units on ESMC-600's block 26, drawn by the frozen rule from the poles H6 ablates, and the
    residue tracks of their neurons on held-out carriers (`tap_probing.units.exemplars`); a few GPU seconds."""
    from tap_probing.units import ablation, exemplars, hypotheses

    started = time.perf_counter()
    output = root / "exemplars"
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = passes.PassInputs.read(root / "inputs")
    key, stem = _causal_key()
    scores_dir = root / "passes" / "e2" / "scores"
    scores = pd.read_parquet(scores_dir / f"{stem}.parquet")
    labeled = ablation.labeled_poles(pd.read_parquet(scores_dir / f"{stem}.labels_local.parquet"), scores, limit=len(scores))
    annotated = labeled[labeled["term"] < inputs.vocabulary_size]  # the release's terms, the intrinsic ones aside
    kinds = hypotheses.input_kinds(root / "inputs")
    chosen = exemplars.draw_exemplars(annotated, kinds)
    proteins = atlas_set.pass_proteins(inputs.frame, subset=experiments.experiment_plans()["e2"].proteins).ordered()
    order = inputs.frame.set_index("accession").loc[proteins.accession].reset_index()
    maxima = np.load(root / "passes" / "e2" / "maxima" / f"{stem}.npy", mmap_mode="r")  # (n, p)
    pairs = exemplars.exemplar_carriers(chosen, order, maxima).merge(chosen, on="exemplar", validate="many_to_one")
    model = _load_model(preregistration, experiments.CAUSAL_MODEL, device)
    units = atlas.transformer_units(model)
    tracks = exemplars.residue_tracks(model, experiments.CAUSAL_BLOCK, order["sequence"].iloc[pairs["row"]].tolist(),
                                      (pairs["pole"] % units).tolist(), device)
    thresholds = torch.load(root / "passes" / "e2" / "calibration.pt", weights_only=True)["thresholds"][key].numpy()  # (p,)
    output.mkdir(parents=True, exist_ok=True)
    exemplars.track_table(pairs, order, tracks, thresholds, units).to_parquet(output / "tracks.parquet", index=False)
    _json(output / "receipt.json", {
        "model": preregistration["models"][experiments.CAUSAL_MODEL], "block": experiments.CAUSAL_BLOCK, "labels": stem,
        "pool": {str(kind): int(count) for kind, count in annotated["term"].map(kinds).value_counts().items()},
        "exemplars": chosen.to_dict(orient="records"), "carriers": pairs[["exemplar", "row", "role", "maximum"]].to_dict(orient="records"),
        "torch": torch.__version__, "device": torch.cuda.get_device_name(device), "seconds": time.perf_counter() - started})
    _log(f"exemplars: {len(chosen)} poles ({', '.join(f'{kind} {count}' for kind, count in chosen['kind'].value_counts().items())}), "
         f"{len(pairs)} tracks")


def stage_steering(root: Path, preregistration: Preregistration, pilot_run: bool) -> None:
    """E7d on ESMC-600: the manifolds of a masked residue's surrounding hydropathy from calibration proteins, then every
    ordered pair of its bins steered on test proteins by each arm (`tap_probing.units.steering`)."""
    from tap_probing.units import steering

    started = time.perf_counter()
    output = root / ("pilots" if pilot_run else "causal") / "steering"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = passes.PassInputs.read(root / "inputs")
    model = _load_model(preregistration, experiments.CAUSAL_MODEL, device)
    pilot = None
    if pilot_run:
        pilot = Pilot(total_steps=len(steering.steering_plan()), items=len(steering.ARMS) * (steering.STEPS - 1),
                      item_unit="steered passes", budget_seconds=experiments.STEERING_BUDGET_HOURS * 3600,
                      receipt=output / "pilot.json", started=started)
    run = steering.run_e7d(model, atlas_set.calibration_sequences(inputs.frame),
                           inputs.frame.loc[inputs.frame["split"] == "test", "sequence"].tolist(), device,
                           block=experiments.CAUSAL_BLOCK, pilot=pilot, log=_log)
    receipt = {"model": preregistration["models"][experiments.CAUSAL_MODEL], "block": experiments.CAUSAL_BLOCK,
               "isometry": run.isometry, "pairs": len(run.pairs), "torch": torch.__version__,
               "device": torch.cuda.get_device_name(device), "seconds": time.perf_counter() - started}
    if pilot_run:
        _json(output / "pilot_receipt.json", receipt)
        return
    run.pairs.to_parquet(output / "steering.parquet", index=False)
    np.savez(output / "manifolds.npz", edges=run.manifolds.edges, knots=run.manifolds.knots,
             centroids=run.manifolds.centroids, behavior=run.manifolds.behavior)
    summary = steering.steering_summary(run)
    _json(output / "receipt.json", {**receipt, "summary": summary, "seconds": time.perf_counter() - started})
    difference = summary["linear_minus_manifold"]
    _log(f"E7d: path energy, linear minus manifold {difference['estimate']:+.3f} [{difference['low']:+.3f}, "
         f"{difference['high']:+.3f}] over {summary['pairs']} bin pairs")


def _concept_data(root: Path, preregistration: Preregistration) -> Path:
    """The pinned concept datasets, downloaded once under `RUN/concepts/data`."""
    from huggingface_hub import snapshot_download

    pin = preregistration["data"]["concepts"]
    target = root / "concepts" / "data"
    snapshot_download(pin["repo"], repo_type="dataset", revision=pin["revision"], local_dir=target)
    return target


def stage_concepts(root: Path, preregistration: Preregistration, model_name: str, pilot_run: bool) -> None:
    """The concept extraction of one model (`concepts.ConceptExtraction`), resuming from its last saved batch."""
    from tap_probing.units import concepts

    started = time.perf_counter()
    plan, budget_hours = experiments.concept_plans()[model_name]
    output = root / "pilots" / f"concepts_{model_name}" if pilot_run else root / "concepts" / model_name
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    proteins = concepts.concept_proteins(concepts.load_tasks(_concept_data(root, preregistration)))
    model = _load_model(preregistration, model_name, device)
    sae = _load_sae(_specs(root, preregistration)[plan.sae], device) if plan.sae else None
    run = concepts.ConceptExtraction(model, plan, proteins.sequences, output, device, sae=sae, log=_log)
    pilot = None
    if pilot_run:
        lengths = np.array([len(proteins.sequences[row]) for row in run.order], dtype=np.int64)  # (n,)
        pilot = Pilot(total_steps=len(run.batches), items=lambda index: int(lengths[run.batches[index]].sum()),
                      item_unit="residues", budget_seconds=budget_hours * 3600, receipt=output / "pilot.json", started=started)
    timing = run.run(pilot)
    _json(output / ("pilot_receipt.json" if pilot_run else "receipt.json"), {
        "model": preregistration["models"][model_name], "plan": vars(plan), "proteins": len(proteins.sequences),
        "concepts": preregistration["data"]["concepts"], "timing": timing, "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device)})


def stage_sparse_probes(root: Path, preregistration: Preregistration, model_name: str) -> None:
    """E6 on one model's concept extraction: k-sparse probes in the native basis and under the ten rotations."""
    from tap_probing.units import concepts, sparse_probes

    plan, _ = experiments.concept_plans()[model_name]
    extraction = root / "concepts" / model_name
    tasks = concepts.load_tasks(_concept_data(root, preregistration))
    table = sparse_probes.run_e6(extraction, concepts.concept_proteins(tasks), tasks, plan, log=_log)
    table.to_parquet(extraction / "e6_scores.parquet", index=False)
    summary = sparse_probes.rotation_summary(table)
    summary.to_parquet(extraction / "e6_rotation_summary.parquet", index=False)
    _json(extraction / "e6_receipt.json", {"targets": int(table[["task", "target"]].drop_duplicates().shape[0]),
                                           "rows": len(table), "plan": vars(plan)})
    _log(f"E6 {model_name}: native beats every rotation by more than their range at "
         f"{int(summary['privileged'].sum())} of {len(summary)} (block, k) cells")


def stage_attribution(root: Path, preregistration: Preregistration) -> None:
    """E9 on ESMC-600: stream probes at block 26 read through the neurons E2 labels with their concept."""
    from tap_probing.units import attribution, concepts
    from tap_probing.units.causal import transformer_blocks
    from tap_probing.units.capture import DOWN_PROJECTION_INDEX

    block = experiments.CAUSAL_BLOCK
    reference = experiments.sweep_name(64, 8192)
    model = _load_model(preregistration, experiments.CAUSAL_MODEL, torch.device("cpu"))
    blocks = transformer_blocks(model)
    down = {index: blocks[index].ffn[DOWN_PROJECTION_INDEX].weight.detach() for index in range(block + 1)}  # (d, f) each
    scores = {index: experiments.score_table(root, "e2", f"block{index}/native@{reference}") for index in range(block + 1)}
    inputs = passes.PassInputs.read(root / "inputs")
    tasks = concepts.load_tasks(_concept_data(root, preregistration))
    extraction = root / "concepts" / experiments.CAUSAL_MODEL
    table = attribution.run_e9(extraction, concepts.concept_proteins(tasks), tasks, block, scores, down,
                               float(blocks[0].scaling_factor), pd.read_parquet(root / "inputs" / "terms.parquet"),
                               inputs.ancestors, log=_log)
    table.to_parquet(extraction / "e9_attribution.parquet", index=False)
    _json(extraction / "e9_receipt.json", {"block": block, "labels_from": f"e2 native@{reference}", "targets": len(table),
                                           "residual_scale": float(blocks[0].scaling_factor)})
    _log(f"E9: {len(table)} mappable targets; labeled share above the null interval for "
         f"{int((table['labeled_share'] > table['null_high']).sum())}")


def stage_payoff(root: Path, preregistration: Preregistration, model_name: str, pilot_run: bool) -> None:
    """E11 on one model's concept extraction: XGBoost on its native, rotated and SAE fingerprints at the 75% block."""
    import xgboost

    from tap_probing.units import concepts, payoff

    started = time.perf_counter()
    plan, _ = experiments.concept_plans()[model_name]
    if plan.rotation_block is None or plan.sae_block != plan.rotation_block:
        raise ValueError(f"E11 needs {model_name}'s rotated extremes and SAE latents at one block.")
    extraction = root / "concepts" / model_name
    output = root / "pilots" / f"payoff_{model_name}" if pilot_run else extraction
    tasks = concepts.load_tasks(_concept_data(root, preregistration))
    proteins = concepts.concept_proteins(tasks)
    fits = payoff.payoff_plan(extraction, tasks, plan.rotation_block)
    pilot = None
    if pilot_run:
        output.mkdir(parents=True, exist_ok=True)
        pilot = Pilot(total_steps=len(fits.fits), items=lambda fit: fits.work(fit, proteins), item_unit="protein-features",
                      budget_seconds=experiments.PAYOFF_BUDGET_HOURS[model_name] * 3600, receipt=output / "pilot.json",
                      started=started)
    table = payoff.run_e11(fits, proteins, device="cuda", pilot=pilot, log=_log)
    receipt = {"model": preregistration["models"][model_name], "block": fits.block, "sae": plan.sae, "fits": len(fits.fits),
               "fitted": len(table), "targets": {task: len(target.names) for task, target in fits.targets.items()},
               "fingerprints": [f"{fingerprint.features}/{fingerprint.basis}" for fingerprint in fits.prints],
               "settings": payoff.SETTINGS, "rounds": payoff.ROUNDS, "xgboost": xgboost.__version__,
               "device": torch.cuda.get_device_name(0), "seconds": time.perf_counter() - started}
    if pilot_run:
        _json(output / "pilot_receipt.json", receipt)
        return
    table.to_parquet(extraction / "e11_scores.parquet", index=False)
    summary = payoff.payoff_summary(table)
    summary.to_parquet(extraction / "e11_summary.parquet", index=False)
    _json(extraction / "e11_receipt.json", receipt)
    for row in summary[summary["metric"] == "pr_auc"].itertuples():
        _log(f"E11 {model_name} PR AUC, {row.comparison}: {row.difference:+.3f} [{row.low:+.3f}, {row.high:+.3f}] "
             f"over {row.targets} targets")


def _convert_shards(annotations: Path, shards: Iterable[int], options: dict[str, list[str]], workers: int) -> None:
    """InterPLM's `convert_shard_to_amino_acid_features` on each shard, `workers` spawned processes at a time; a failure
    cancels the shards not yet started and stops the stage (InterPLM's `main` logs it and carries on)."""
    import multiprocessing

    from concurrent.futures import ProcessPoolExecutor, as_completed

    from interplm.analysis.concepts import extract_annotations

    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(extract_annotations.convert_shard_to_amino_acid_features, shard_id=shard,
                               input_path=annotations / f"shard_{shard}" / "protein_data.tsv", output_dir=annotations,
                               categorical_options=options, binary_cols=extract_annotations.binary_meta_cols,
                               interaction_cols=extract_annotations.paired_binary_cols, overwrite=True)
                   for shard in shards]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise


def _free_bytes() -> int:
    """Memory free to this process now (Linux): the machine's free pages, or less inside a container whose cgroup caps
    its memory."""
    free = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    limit = Path("/sys/fs/cgroup/memory.max")
    if limit.exists() and limit.read_text(encoding="utf-8").strip() != "max":
        used = int(Path("/sys/fs/cgroup/memory.current").read_text(encoding="utf-8"))
        free = min(free, int(limit.read_text(encoding="utf-8")) - used)
    return free


def _shard_workers(shards: int) -> tuple[int, int]:
    """Workers for the remaining shards from the first one's peak memory (Linux): half of the memory free now, at most
    half the cores; and that peak in bytes."""
    import resource

    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024  # kilobytes on Linux
    return max(1, min(shards, (os.cpu_count() or 2) // 2, int(RETEST_MEMORY_SHARE * _free_bytes() / peak))), peak


def _evaluation_set(source: Path, annotations: Path, *, proteins: int, shards: int, min_instances: int,
                    min_residues: int, min_domains: int, workers: int | None = None) -> dict[str, object]:
    """InterPLM's `extract_annotations.main` and `make_eval_subset` at the pinned commit, step by step: its preprocessing
    of every protein in `source`, the concept sub-categories enumerated over all of them (as the paper's run over
    Swiss-Prot did), the first `proteins` split by position into `shards` shards and converted to residue labels by
    InterPLM's own function, and the first half of the shards for validation, the rest for test. `main` splits with
    `np.array_split` on the frame, which pandas 3 returns as arrays. The first shard is converted alone and timed;
    `workers`, when None, is set from its peak memory."""
    import shutil

    from interplm.analysis.concepts import extract_annotations, prepare_eval_set
    from tap_probing.units import retest

    usable = extract_annotations.preprocess_proteins(pd.read_csv(source, sep="\t"), retest.MAX_LENGTH)
    options = extract_annotations.enumerate_protein_subcategories(usable, min_instances)
    sample = usable.head(proteins)
    if len(sample) != proteins:
        raise ValueError(f"{source} gives {len(usable)} usable proteins, fewer than {proteins}.")
    built = {"usable": len(usable), "proteins": proteins, "residues": int(sample["Length"].sum()),
             "accessions_sha256": hashlib.sha256("\n".join(sample["Entry"]).encode()).hexdigest(),
             "sub_categories": {column: len(values) for column, values in options.items()}}
    if annotations.exists():
        shutil.rmtree(annotations)
    for shard, positions in enumerate(np.array_split(np.arange(proteins), shards)):  # positions: (p_s,)
        folder = annotations / f"shard_{shard}"
        folder.mkdir(parents=True)
        sample.iloc[positions].reset_index(drop=True).to_csv(folder / "protein_data.tsv", sep="\t", index=False)
    del usable, sample
    started = time.perf_counter()
    _convert_shards(annotations, [0], options, 1)
    first = time.perf_counter() - started
    if workers is None:
        workers, built["first_shard_peak_bytes"] = _shard_workers(shards - 1)
    _log(f"E1 labels: the first shard took {first:.0f} s; {shards - 1} more on {workers} workers, about "
         f"{first * math.ceil((shards - 1) / workers) / 60:.0f} min")
    _convert_shards(annotations, range(1, shards), options, workers)
    half = shards // 2
    prepare_eval_set.make_eval_subset(uniprot_dir=annotations, valid_shard_range=(0, half - 1),
                                      test_shard_range=(half, shards - 1), min_aa_per_concept=min_residues,
                                      min_domains_per_concept=min_domains)
    return {**built, "first_shard_seconds": first, "workers": workers, "seconds": time.perf_counter() - started}


def stage_retest_inputs(root: Path) -> None:
    """E1's evaluation set: all of Swiss-Prot from UniProt's current release with InterPLM's fields, through
    InterPLM's procedure (`_evaluation_set`). CPU only; it can run beside a GPU stage."""
    from tap_probing.units import retest

    folder = root / "retest"
    source = folder / "uniprot_sprot.tsv.gz"
    downloaded = _download(retest.UNIPROT_STREAM, source)
    built = _evaluation_set(source, folder / "annotations", proteins=retest.EVAL_PROTEINS, shards=retest.SHARDS,
                            min_instances=retest.MIN_INSTANCES, min_residues=retest.MIN_RESIDUES,
                            min_domains=retest.MIN_DOMAINS)
    metadata = json.loads((folder / "annotations" / "valid" / "metadata.json").read_text(encoding="utf-8"))
    _json(folder / "inputs_receipt.json", {"uniprot": downloaded, **built, "concepts": metadata["n_concepts"],
                                           "shards": retest.SHARDS})
    _log(f"E1 evaluation set: {built['proteins']} proteins, {metadata['n_concepts']} concepts")


def _retest_data(root: Path) -> tuple[Any, list[Any]]:
    """E1's concepts and shards from InterPLM's evaluation set under `RUN/retest/annotations`."""
    import scipy.sparse as sp

    from interplm.analysis.concepts.concept_constants import is_aa_level_concept
    from tap_probing.units import retest

    annotations = root / "retest" / "annotations"
    metadata = {split: json.loads((annotations / split / "metadata.json").read_text(encoding="utf-8")) for split in retest.SPLITS}
    names = tuple((annotations / "valid" / "aa_concepts_columns.txt").read_text(encoding="utf-8").split("\n"))
    kept = metadata["valid"]["indices_of_concepts_to_keep"]
    concepts = retest.ConceptSet(names, np.array([not is_aa_level_concept(name) for name in names]),
                                 {split: np.asarray(entry["n_positive_aa_per_concept"]) for split, entry in metadata.items()},
                                 {split: np.asarray(entry["n_positive_domains_per_concept"]) for split, entry in metadata.items()})
    shards = [retest.RetestShard(split, tuple(pd.read_csv(annotations / f"shard_{shard}" / "protein_data.tsv", sep="\t")["Sequence"]),
                                 sp.load_npz(entry["path_to_shards"][str(shard)]).tocsr()[:, kept])
              for split, entry in metadata.items() for shard in entry["shard_source"]]
    return concepts, shards


def _interplm_sae(preregistration: Preregistration, model_name: str, layer: int, device: torch.device) -> nn.Module:
    """InterPLM's released SAE of one layer at its pinned revision, loaded as `load_sae_from_hf` loads it (the image
    completes InterPLM's install with `interplm.train` and `sae/migration`, which its wheel leaves out)."""
    import shutil

    from huggingface_hub import hf_hub_download
    from interplm.sae import inference
    from tap_probing.units import retest

    pin = preregistration["saes"][f"interplm_{model_name}"]
    weights = Path(hf_hub_download(pin["repo"], f"layer_{layer}/ae_normalized.pt", revision=pin["revision"]))
    if not (weights.parent / "config.yaml").exists():
        shutil.copy(Path(inference.__file__).parent / "migration" / f"dummy_config_{retest.ESM2[model_name][2]}.yaml",
                    weights.parent / "config.yaml")
    return inference.load_sae(weights.parent, model_name="ae_normalized.pt", device=device).eval()


def _retest_parity(batches: list[Any], ranges: Any, concepts: Any, shards: list[Any]) -> dict[str, object]:
    """Counts, F1 and the report here against InterPLM's `calc_metrics_dense`, `calculate_metrics` and report functions,
    on the first layer's residual arm over a validation and a test batch; any difference stops the stage."""
    from interplm.analysis.concepts.calculate_f1 import calculate_metrics
    from interplm.analysis.concepts.compare_activations import calc_metrics_dense
    from interplm.analysis.concepts.report_metrics import find_all_top_heldout_feats, find_top_heldout_feat_per_concept
    from tap_probing.units import retest

    is_aa = [not bool(flag) for flag in concepts.domain_level]
    position = {name: index for index, name in enumerate(concepts.names)}
    arrays, frames = {}, {}
    for batch in batches:  # one validation and one test batch, the first layer's arms
        scaled = ranges.scale((batch.layer, "residual"), batch.arms["residual"])[:, :retest.PARITY_FEATURES]  # (n, 64)
        labels = shards[batch.shard].labels[batch.rows]  # (n, C)
        counts = retest.ConceptCounts(len(concepts.names), scaled.shape[1], device=scaled.device)
        counts.update(scaled, retest.batch_labels(labels, concepts.domain_level, scaled.device))
        tp, fp, per_domain = calc_metrics_dense(scaled, labels.toarray(), list(retest.THRESHOLDS), is_aa)
        mine = (counts.positive.cpu().numpy(), (counts.fired[None] - counts.positive).cpu().numpy(), counts.domains.cpu().numpy())
        for name, theirs, ours in zip(("tp", "fp", "tp_per_domain"), (tp, fp, per_domain), mine, strict=True):
            if not np.array_equal(np.asarray(theirs), ours):
                raise ValueError(f"E1 parity: {name} differs from InterPLM's on the {batch.split} batch.")
        frames[batch.split] = calculate_metrics(tp, fp, per_domain, concepts.positives[batch.split], concepts.domains[batch.split],
                                                list(concepts.names), list(retest.THRESHOLDS), is_aa)
        arrays[batch.split] = retest.f1_scores(counts, concepts, batch.split)
        present, f1, f1_domain = (array.cpu().numpy() for array in arrays[batch.split])
        rows = frames[batch.split]
        index = ([position[name] for name in rows["concept"]], rows["feature"].to_numpy(),
                 [retest.THRESHOLDS.index(value) for value in rows["threshold_pct"]])
        if len(rows) != int(present.sum()) or not (np.allclose(f1[index], rows["f1"]) and np.allclose(f1_domain[index], rows["f1_per_domain"])):
            raise ValueError(f"E1 parity: F1 differs from InterPLM's on the {batch.split} batch.")
    report = retest.interplm_report(arrays["valid"], arrays["test"], concepts.reported)
    best = find_top_heldout_feat_per_concept(frames["valid"], frames["test"])
    pairs = find_all_top_heldout_feats(frames["valid"], frames["test"])
    expected = dict(zip(best["concept"], best["f1_per_domain"], strict=True))
    ours = {concepts.names[index]: value for index, value in enumerate(report.best_test_f1) if np.isfinite(value)}
    if set(ours) != set(expected) or not all(np.isclose(ours[name], expected[name]) for name in ours):
        raise ValueError("E1 parity: the best test F1 per concept differs from InterPLM's report.")
    if {(int(feature), concept) for feature, concept in zip(pairs["feature"], pairs["concept"], strict=True)} != {
            (int(feature), concepts.names[concept]) for feature, concept in report.pairs}:
        raise ValueError("E1 parity: the associated pairs differ from InterPLM's report.")
    return {"features": retest.PARITY_FEATURES, "entries": {split: len(frame) for split, frame in frames.items()},
            "concepts_scored": len(ours), "pairs": len(report.pairs)}


def stage_retest(root: Path, preregistration: Preregistration, model_name: str, pilot_run: bool) -> None:
    """E1 on one ESM-2 model: the ranges pass, the parity check against InterPLM's functions, the counting pass and
    InterPLM's report for every arm and layer."""
    from transformers import AutoTokenizer, EsmForMaskedLM

    from tap_probing.units import retest

    started = time.perf_counter()
    repo, layers, _ = retest.ESM2[model_name]
    spec = preregistration["models"][model_name]
    if spec["repo"] != repo:
        raise ValueError(f"{model_name} is pinned at {spec['repo']}, the re-test reads {repo}.")
    concepts, shards = _retest_data(root)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True  # counts compare thresholded activations; recorded in the receipt
    model = EsmForMaskedLM.from_pretrained(repo, revision=spec["revision"]).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=spec["revision"])
    shuffled = retest.shuffled_copy(model)
    saes = {layer: _interplm_sae(preregistration, model_name, layer, device) for layer in layers}
    # InterPLM's comparison reads latents through encode_feat_subset, which applies the SAE's input normalization
    encoders = {layer: functools.partial(sae.encode_feat_subset, feat_list=slice(None)) for layer, sae in saes.items()}
    output = root / "pilots" / f"retest_{model_name}" if pilot_run else root / "retest" / model_name
    output.mkdir(parents=True, exist_ok=True)

    def stream(batches: Any = None) -> Any:
        return retest.stream_arms(model, shuffled, tokenizer, layers, shards, encoders, device=device, batches=batches)

    receipt = {"model": spec, "layers": layers, "interplm_saes": preregistration["saes"][f"interplm_{model_name}"]}
    _run_retest(stream, concepts, shards, layers, output, experiment="e1", model_name=model_name, device=device,
                budget_hours=experiments.RETEST_BUDGET_HOURS[model_name], token_budget=retest.TOKEN_BUDGET,
                pilot_run=pilot_run, started=started, receipt=receipt, logged=("residual", "mlp_poles", "sae"))


def stage_retest_esmc(root: Path, preregistration: Preregistration, plan_name: str, pilot_run: bool) -> None:
    """E13, exploratory, added after H0 to H7 were decided: E1's evaluation of one ESMC plan, by default the model's
    four depths, the native poles against rotated units, the stream and the SAEs at the hypothesis block
    (`retest.stream_esmc_arms`). Outputs go to RUN/retest_esmc/<plan>, whose tables' `model` column names the plan."""
    from tap_probing.units import retest

    started = time.perf_counter()
    plan = experiments.esmc_retest_plans()[plan_name]
    concepts, shards = _retest_data(root)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True  # as E1
    model = _load_model(preregistration, plan.model, device)
    shuffled = retest.shuffled_copy(model) if plan.shuffled else None
    depth_specs = experiments.depth_sae_specs()
    specs = {**_specs(root, preregistration), **depth_specs}
    saes: dict[int, dict[str, nn.Module]] = {}
    for name in plan.saes:
        if specs[name].block not in plan.blocks:
            raise ValueError(f"{name} reads block {specs[name].block}, which {plan_name} does not read.")
        saes.setdefault(specs[name].block, {})[name] = _load_sae(specs[name], device)
    family = [specs[name] for name in plan.saes if name in depth_specs]
    position = _family_position(model, family, saes, shards[0].sequences[:FAMILY_CHECK_PROTEINS], device) if family else None
    searched, search = _searched_bases(root, plan_name, plan, model, device, pilot_run) if plan.searched else (None, None)
    rotation_seeds = retest.ROTATION_SEEDS if plan.rotation_seeds is None else plan.rotation_seeds
    output = root / "pilots" / f"retest_esmc_{plan_name}" if pilot_run else root / "retest_esmc" / plan_name
    output.mkdir(parents=True, exist_ok=True)

    def stream(batches: Any = None) -> Any:
        return retest.stream_esmc_arms(model, shuffled, plan.blocks, shards, saes, device=device,
                                       token_budget=plan.token_budget, rotation_seeds=rotation_seeds,
                                       sae_reads={name: specs[name].reads for name in plan.saes}, arms=plan.arms,
                                       searched=searched, batches=batches)

    receipt = {"model": preregistration["models"][plan.model], "plan_name": plan_name, "plan": dataclasses.asdict(plan),
               "saes": {name: dataclasses.asdict(specs[name]) for name in plan.saes}, "family_position": position,
               "search": search}
    logged = ("residual", "mlp_native", "mlp_poles", "mlp_rotation0", *(f"sae:{name}" for name in plan.saes),
              *(arm for arm in plan.arms or () if arm.startswith(("mlp_standard", "mlp_start", "mlp_searched"))))
    _run_retest(stream, concepts, shards, plan.blocks, output, experiment="e13", model_name=plan_name, device=device,
                budget_hours=plan.budget_hours, token_budget=plan.token_budget, pilot_run=pilot_run, started=started,
                receipt=receipt, logged=logged)


def _family_position(model: nn.Module, specs: Sequence[experiments.SaeSpec], saes: Mapping[int, Mapping[str, nn.Module]],
                     sequences: Sequence[str], device: torch.device) -> dict[str, object]:
    """E13x's check of an all-layer SAE family's read position, by E0's rule for a family (`_read_positions`): each
    member's reconstruction loss on the stream its planned block outputs and on its neighbours', over a few validation
    proteins, and the offset from the planned block at which the members' mean loss is lowest; any offset but 0 stops
    the stage. A member whose own lowest loss sits elsewhere is listed, not decisive."""
    last = len(atlas.transformer_blocks(model)) - 1
    candidates = sorted({block for spec in specs for block in (spec.block - 1, spec.block, spec.block + 1) if 0 <= block <= last})
    keep = np.ones(sum(len(sequence) for sequence in sequences), dtype=bool)  # (r_total,)
    states = atlas.calibration_states(model, model.tokenizer, sequences, candidates, keep, stream_blocks=candidates,
                                      device=device)
    losses = {spec.name: {offset: _reconstruction_loss(saes[spec.block][spec.name], states[spec.block + offset].stream, device)
                          for offset in (-1, 0, 1) if spec.block + offset in states} for spec in specs}
    mean = {offset: float(np.mean([member[offset] for member in losses.values() if offset in member])) for offset in (-1, 0, 1)}
    best = min(mean, key=mean.get)
    _log(f"all-layer SAE family: mean losses by offset from the planned block {mean}, best {best:+d}")
    if best != 0:
        raise ValueError(f"The all-layer SAEs reconstruct best {best:+d} blocks from their planned block: {mean}.")
    return {"proteins": len(sequences), "mean_losses_by_offset": mean, "best_offset": best, "losses": losses,
            "members_best_elsewhere": [name for name, member in losses.items() if min(member, key=member.get) != 0]}


def _searched_bases(root: Path, plan_name: str, plan: experiments.EsmcRetestPlan, model: nn.Module, device: torch.device,
                    pilot_run: bool) -> tuple[dict[int, Any], dict[str, object]]:
    """E13s's bases at the plan's one block: E8's pass and sparsest-rotation search re-run as E8 ran them (TF32 off, its
    proteins, sample, seeds and steps), keeping the rotations, into RUN/retest_esmc/<plan>/search. The sample must be
    E8's: its native objective and each search's start objective must equal E8's receipt, or the stage stops; where
    each search lands is reported beside E8's. A pilot times the counting with each search's Haar start standing in for
    the rotation the search would find, in unscaled coordinates."""
    from tap_probing.units import retest
    from tap_probing.units.rotation_search import SEARCH_SEEDS, Standardization
    from tap_probing.units.rotations import haar_rotation

    (block,) = plan.blocks
    reference = json.loads((root / "dictionary" / plan.model / "report.json").read_text(encoding="utf-8"))
    if reference["block"] != block:
        raise ValueError(f"E8 searched block {reference['block']} of {plan.model}, not {plan_name}'s {block}.")
    units = atlas.transformer_units(model)
    starts = {seed: haar_rotation(units, seed, dtype=torch.float32, device=device) for seed in SEARCH_SEEDS}  # (f, f) each
    if pilot_run:
        standard = Standardization(torch.zeros(units, device=device), torch.ones(units, device=device))
        found, receipt = starts, {"pilot": "each search's Haar start stands in for the rotation it would find"}
    else:
        output = root / "retest_esmc" / plan_name / "search"
        torch.backends.cuda.matmul.allow_tf32 = False  # as E8 ran
        try:
            report = passes.run_dictionary(model, {}, {}, block, passes.PassInputs.read(root / "inputs"), output, device,
                                           keep_rotations=True, log=_log)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = True  # as E1 and E13 count
        ours, theirs = report["rotation_search"], reference["rotation_search"]
        starts_ours = {entry["seed"]: entry["start_objective"] for entry in ours["searches"]}
        starts_theirs = {entry["seed"]: entry["start_objective"] for entry in theirs["searches"]}
        same = math.isclose(ours["native_objective"], theirs["native_objective"], rel_tol=1e-6) and starts_ours.keys() == starts_theirs.keys()
        if not same or not all(math.isclose(starts_ours[seed], starts_theirs[seed], rel_tol=1e-6) for seed in starts_ours):
            raise ValueError(f"E13s's sample is not E8's: native objective {ours['native_objective']} against "
                             f"{theirs['native_objective']}, start objectives {starts_ours} against {starts_theirs}.")
        for entry, earlier in zip(ours["searches"], theirs["searches"], strict=True):
            _log(f"E13s search from seed {entry['seed']}: objective {entry['objective']:.1f} after {entry['steps']} steps; "
                 f"E8's {earlier['objective']:.1f} after {earlier['steps']}; native {ours['native_objective']:.1f}")
        kept = np.load(output / "searched.npz")
        standard = Standardization(torch.from_numpy(kept["mean"]), torch.from_numpy(kept["scale"])).to(device, torch.float32)
        found = {seed: torch.from_numpy(kept[f"rotation{seed}"]).to(device) for seed in SEARCH_SEEDS}  # (f, f) each
        receipt = {"search": ours, "e8": theirs, "seconds": report["seconds"]}
    rotations = {**{f"start{seed}": starts[seed] for seed in SEARCH_SEEDS}, **{f"searched{seed}": found[seed] for seed in SEARCH_SEEDS}}
    return {block: retest.SearchedBases(standard, rotations)}, receipt


def _run_retest(stream: Callable[[Any], Any], concepts: Any, shards: list[Any], layers: Sequence[int], output: Path, *,
                experiment: str, model_name: str, device: torch.device, budget_hours: float, token_budget: int,
                pilot_run: bool, started: float, receipt: dict[str, object], logged: Sequence[str]) -> None:
    """The passes of a re-test (E1, E13): the ranges pass, the parity check against InterPLM's functions on the first
    layer's residual arm, the counting pass, watched by a pilot when piloting, and InterPLM's report for every arm."""
    from tap_probing.units import retest

    window = (lambda plan: plan[:retest.PILOT_BATCHES]) if pilot_run else None
    timed = time.perf_counter()
    ranges = retest.collect_ranges(stream(window))
    first_pass = time.perf_counter() - timed
    first_test = next(index for index, shard in enumerate(shards) if shard.split == "test")
    parity_batches = [batch for batch in stream(lambda plan: [entry for entry in plan if entry[1] == 0 and entry[0] in (0, first_test)])
                      if batch.layer == layers[0]]
    parity = _retest_parity(parity_batches, ranges, concepts, shards)
    label = f"{experiment.upper()} {model_name}"
    _log(f"{label} parity with InterPLM: {parity}")
    pilot = None
    if pilot_run:
        total = sum(len(retest.protein_batches([len(sequence) for sequence in shard.sequences], token_budget)) for shard in shards)
        pilot = Pilot(total_steps=total, items=lambda entry: len(entry[2]), item_unit="proteins",
                      budget_seconds=budget_hours * 3600, receipt=output / "pilot.json", started=started)
    timed = time.perf_counter()
    counts = retest.collect_counts(stream(pilot.watch if pilot is not None else None), shards, ranges, concepts, device=device)
    second_pass = time.perf_counter() - timed
    receipt = {**receipt, "concepts": len(concepts.names), "shards": len(shards), "parity": parity, "tf32": True,
               "first_pass_seconds": first_pass, "second_pass_seconds": second_pass, "torch": torch.__version__,
               "device": torch.cuda.get_device_name(device)}
    if pilot_run:
        receipt["projected_hours"] = (first_pass + second_pass) / retest.PILOT_BATCHES * pilot.total_steps / 3600
        _json(output / "pilot_receipt.json", receipt)
        return
    summary, per_concept = retest.retest_tables(counts, concepts)
    summary.assign(model=model_name).to_parquet(output / f"{experiment}_summary.parquet", index=False)
    per_concept.assign(model=model_name).to_parquet(output / f"{experiment}_per_concept.parquet", index=False)
    _json(output / "receipt.json", {**receipt, "seconds": time.perf_counter() - started})
    for row in summary[summary["arm"].isin(list(logged))].itertuples():
        _log(f"{label} layer {row.layer} {row.arm}: {row.associated_features} of {row.features} features "
             f"({row.share:.3f}) on {row.identified_concepts} concepts")


def stage_strength(root: Path, preregistration: Preregistration, model_name: str, pilot_run: bool) -> None:
    """E14a, exploratory: at one model's three-quarter block, on E1's evaluation set, the precision of each unit's
    best concept by firing-strength band, the native poles against three rotations (`tap_probing.units.strength`)."""
    from tap_probing.units import retest, strength

    started = time.perf_counter()
    block, token_budget, budget_hours = experiments.STRENGTH_SITES[model_name]
    concepts, shards = _retest_data(root)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True  # as E13
    model = _load_model(preregistration, model_name, device)
    output = root / "pilots" / f"strength_{model_name}" if pilot_run else root / "strength" / model_name
    output.mkdir(parents=True, exist_ok=True)

    def stream(batches: Any = None) -> Any:
        return retest.stream_esmc_arms(model, None, (block,), shards, {}, device=device, token_budget=token_budget,
                                       rotation_seeds=strength.ROTATION_SEEDS, arms=strength.ARMS, batches=batches)

    window, rate = strength.sample_window(shards, token_budget)
    timed = time.perf_counter()
    sample = strength.collect_sample(stream(window), rate)
    sampled = int(sample[strength.ARMS[0]].shape[0])
    thresholds = {arm: strength.tail_thresholds(values) for arm, values in sample.items()}  # (B, F) each
    del sample
    sample_seconds = time.perf_counter() - timed
    _log(f"E14a {model_name}: thresholds from {sampled} validation residues in {sample_seconds:.0f} s")
    total = sum(len(retest.protein_batches([len(sequence) for sequence in shard.sequences], token_budget)) for shard in shards)
    pilot = Pilot(total_steps=total, items=lambda entry: len(entry[2]), item_unit="proteins", budget_seconds=budget_hours * 3600,
                  receipt=output / "pilot.json", started=started) if pilot_run else None
    timed = time.perf_counter()
    counts = strength.collect_bands(stream(pilot.watch if pilot is not None else None), shards, thresholds, concepts)
    receipt = {"model": preregistration["models"][model_name], "block": block, "token_budget": token_budget,
               "arms": list(strength.ARMS), "tails": list(strength.TAILS), "sampled_residues": sampled, "sample_rate": rate,
               "min_fired": strength.MIN_FIRED, "sample_seconds": sample_seconds, "count_seconds": time.perf_counter() - timed,
               "tf32": True, "torch": torch.__version__, "device": torch.cuda.get_device_name(device)}
    if pilot_run:
        _json(output / "pilot_receipt.json", receipt)
        return
    residues = sum(shard.labels.shape[0] for shard in shards if shard.split == "test")
    base_rate = np.asarray(concepts.positives["test"], dtype=np.float64) / residues  # (C,)
    readings = {arm: strength.band_reading(counts[arm, "valid"], counts[arm, "test"], concepts.reported, base_rate)
                for arm in strength.ARMS}
    names = np.array(concepts.names)
    units = []
    for arm, reading in readings.items():
        frame = pd.DataFrame({"arm": arm, "unit": np.arange(reading.concept.shape[0])})
        for band, label in enumerate(strength.BANDS):
            frame[f"concept:{label}"] = np.where(reading.concept[:, band] >= 0, names[np.maximum(reading.concept[:, band], 0)], "")
            frame[f"precision:{label}"] = reading.precision[:, band]
            frame[f"lift:{label}"] = reading.lift[:, band]
            frame[f"test_residues:{label}"] = reading.fired[:, band]
        units.append(frame)
    pd.concat(units, ignore_index=True).assign(model=model_name, block=block).to_parquet(output / "units.parquet", index=False)
    table = strength.band_table(readings).assign(model=model_name, block=block)
    table.to_parquet(output / "bands.parquet", index=False)
    narrowing = strength.narrowing(readings)
    _json(output / "summary.json", {"bands": table.to_dict("records"), "narrowing": narrowing})
    _json(output / "receipt.json", {**receipt, "test_residues": residues, "seconds": time.perf_counter() - started})
    for arm in strength.ARMS:
        rows = table[table["arm"] == arm]
        _log(f"E14a {model_name} {arm}: median precision by band {[round(value, 3) for value in rows['median_precision']]}, "
             f"median rise {narrowing[arm]['median_rise']} over {narrowing[arm]['units']} units")


def stage_wiring(root: Path, preregistration: Preregistration, model_name: str) -> None:
    """E14b, exploratory: from one model's weights alone, how closely each unit's gate and up rows lie along one
    writing unit of the block before, against rotations of those units (`tap_probing.units.wiring`)."""
    from tap_probing.units import wiring

    started = time.perf_counter()
    device = torch.device("cuda")
    model = _load_model(preregistration, model_name, device)
    table = wiring.wiring_table(model)
    output = root / "wiring" / model_name
    output.mkdir(parents=True, exist_ok=True)
    table.assign(model=model_name).to_parquet(output / "links.parquet", index=False)
    summary = wiring.wiring_summary(table)
    _json(output / "summary.json", {"model": preregistration["models"][model_name], "strong_link": wiring.STRONG_LINK,
                                    "rotation_seeds": list(wiring.ROTATION_SEEDS), "sides": summary,
                                    "seconds": time.perf_counter() - started, "torch": torch.__version__})
    for side, entry in summary.items():
        _log(f"E14b {model_name} {side}: strong links for {entry['native_strong_share']:.3f} of units "
             f"(pairs {entry['native_strong_share_range']}), rotations at most {entry['rotation_strong_share_max']:.4f}")


COPY_TOLERANCE = 0.002  # E14c's copy shares, from float32 sums, against E8's, from float64


def stage_coalitions(root: Path, preregistration: Preregistration, model_name: str, pilot_run: bool) -> None:
    """E14c, exploratory: at one of E8's sites, each latent of the SAE on the feed-forward write fit with 1 to 64
    native poles on half of E8's residues and read on the other half, against rotated poles
    (`tap_probing.units.coalitions`). It stops unless its copy shares equal E8's."""
    from tap_probing.units import coalitions

    started = time.perf_counter()
    site = experiments.dictionary_sites()[model_name]
    specs = _specs(root, preregistration)
    name = next(candidate for candidate in site.saes if specs[candidate].reads == "write")
    output = root / "pilots" / f"coalitions_{model_name}" if pilot_run else root / "coalitions" / model_name
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False  # as E8
    inputs = passes.PassInputs.read(root / "inputs")
    pilot = None
    if pilot_run:
        lengths = atlas_set.residue_budget_proteins(inputs.frame, passes.DICTIONARY_RESIDUES).ordered().lengths  # (n,)
        batches = atlas.length_batches(lengths, passes.DICTIONARY_TOKEN_BUDGET)
        pilot = Pilot(total_steps=len(batches), items=lambda index: int(lengths[batches[index]].sum()), item_unit="residues",
                      budget_seconds=experiments.COALITION_BUDGET_HOURS[model_name] * 3600, receipt=output / "pilot.json",
                      started=started)
    model = _load_model(preregistration, model_name, device)
    sae = _load_sae(specs[name], device)
    frame, receipt = coalitions.run_coalitions(model, sae, site.block, inputs, device, pilot=pilot, log=_log)
    receipt.update(model=preregistration["models"][model_name], sae=vars(specs[name]), torch=torch.__version__,
                   device=torch.cuda.get_device_name(device))
    if frame is None:
        _json(output / "pilot_receipt.json", receipt)
        return
    earlier = json.loads((root / "dictionary" / model_name / "report.json").read_text(encoding="utf-8"))["dictionaries"][name]
    for threshold, share in receipt["copy_shares"].items():
        if abs(share - earlier["native_shares"][threshold]) > COPY_TOLERANCE:
            raise ValueError(f"{name}'s copy share at {threshold} is {share:.4f}, E8's {earlier['native_shares'][threshold]:.4f}.")
    report = coalitions.coalition_report(frame)
    per_concept = pd.read_parquet(root / "retest_esmc" / f"{model_name}_mlp" / "e13_per_concept.parquet")
    concepts = coalitions.concept_latents(per_concept, f"sae:{name}", "mlp_poles", frame)
    frame.to_parquet(output / f"{name}.parquet", index=False)
    _json(output / "report.json", {**receipt, "e8_copy_shares": earlier["native_shares"], "shares": report, "concepts": concepts,
                                   "seconds": time.perf_counter() - started})
    for basis in coalitions.BASES:
        _log(f"E14c {model_name} {basis}: activity reached at 0.7 by fit size {report['0.7'][basis]['mass_share']}")


def _taxonomy_panel(root: Path) -> Any:
    """E10's taxonomy panel, drawn from the atlas set with the release's lineages at eight ranks; the first draw is written
    to `RUN/panel/panel.parquet`, and a later one that differs from it stops the stage."""
    from tap_probing.units import subspaces

    atlas = passes.PassInputs.read(root / "inputs").frame
    table = pd.read_parquet(root / "source" / "tables" / "lineage.parquet", columns=["protein_id", *subspaces.TAXONOMIC_RANKS])
    lineage_of = subspaces.rank_lineages(table[table["protein_id"].isin(set(atlas["protein_id"]))])
    panel = subspaces.taxonomy_panel(atlas, {accession: lineage_of.get(protein, ()) for accession, protein
                                             in zip(atlas["accession"], atlas["protein_id"], strict=True)})
    frame = pd.DataFrame({"accession": panel.accessions, "split": panel.split,
                          "lineage": [list(lineage) for lineage in panel.lineages]})
    path = root / "panel" / "panel.parquet"
    if path.exists():
        if not pd.read_parquet(path)[["accession", "split"]].equals(frame[["accession", "split"]]):
            raise ValueError(f"{path} differs from the panel the atlas set draws now.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    return panel


def stage_taxonomy_panel(root: Path, preregistration: Preregistration, model_name: str, pilot_run: bool) -> None:
    """E10's taxonomy panel through one model, pooled at the concept extraction's depths."""
    from tap_probing.units import concepts

    started = time.perf_counter()
    panel = _taxonomy_panel(root)
    plan = concepts.ExtractionPlan(concepts.DEPTHS[model_name])
    output = root / "pilots" / f"panel_{model_name}" if pilot_run else root / "panel" / model_name
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    model = _load_model(preregistration, model_name, device)
    run = concepts.ConceptExtraction(model, plan, list(panel.sequences), output, device, log=_log)
    pilot = None
    if pilot_run:
        lengths = np.array([len(panel.sequences[row]) for row in run.order], dtype=np.int64)  # (n,)
        pilot = Pilot(total_steps=len(run.batches), items=lambda index: int(lengths[run.batches[index]].sum()),
                      item_unit="residues", budget_seconds=experiments.PANEL_BUDGET_HOURS[model_name] * 3600,
                      receipt=output / "pilot.json", started=started)
    timing = run.run(pilot)
    _json(output / ("pilot_receipt.json" if pilot_run else "receipt.json"), {
        "model": preregistration["models"][model_name], "plan": vars(plan), "proteins": len(panel.sequences),
        "atlas_inputs": preregistration["data"]["atlas_inputs"], "timing": timing, "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device)})


def stage_subspaces(root: Path, preregistration: Preregistration, model_name: str) -> None:
    """E10 on one model: reduced-rank readouts of the concept families and of taxonomy at every depth."""
    from tap_probing.units import concepts, subspaces

    started = time.perf_counter()
    tasks = concepts.load_tasks(_concept_data(root, preregistration))
    families = [*subspaces.concept_families(tasks, concepts.concept_proteins(tasks)),
                subspaces.taxonomy_family(_taxonomy_panel(root))]
    extractions = {"concepts": root / "concepts" / model_name, "panel": root / "panel" / model_name}
    table = subspaces.run_e10(extractions, families, concepts.DEPTHS[model_name], device="cuda", log=_log)
    table.to_parquet(extractions["concepts"] / "e10_subspaces.parquet", index=False)
    _json(extractions["concepts"] / "e10_receipt.json", {
        "families": {family.name: family.targets for family in families}, "depths": concepts.DEPTHS[model_name],
        "features": subspaces.FEATURES, "ranks": subspaces.RANKS, "penalties": subspaces.PENALTIES,
        "panel_proteins": subspaces.PANEL_PROTEINS, "rows": len(table), "seconds": time.perf_counter() - started})
    site = table[(table["block"] == concepts.DEPTHS[model_name][2]) & (table["features"] == "z_mean") & table["full"]]
    for row in site.itertuples():
        _log(f"E10 {model_name} {row.family}: dimension {row.dimension} of {row.targets}, full test {row.test:.3f}, "
             f"stable rank {row.stable_rank:.1f}")
    if model_name != experiments.CAUSAL_MODEL:
        return
    correlation = np.load(root / "dictionary" / model_name / "pole_cofiring.npy")  # (2f, 2f), from E8's pass
    blocks = subspaces.run_e10_blocks(extractions["concepts"], [family for family in families if family.source == "concepts"],
                                      experiments.CAUSAL_BLOCK, correlation, device="cuda", log=_log)
    blocks.to_parquet(extractions["concepts"] / "e10_blocks.parquet", index=False)
    for row in blocks.itertuples():
        _log(f"E10 co-firing {row.family}: best block of {row.best_size} poles {row.best_test:.3f} against random "
             f"{row.null_mean:.3f} [{row.null_low:.3f}, {row.null_high:.3f}] and every pole {row.all_poles_test:.3f}")


def stage_figures(root: Path, name: str | None) -> None:
    """The paper's figures from the run's outputs (`tap_probing.units.figures`), each saved to `RUN/figures`."""
    import matplotlib

    from tap_probing.units import figures

    names = list(figures.FIGURES) if name is None else [name]
    unknown = [chosen for chosen in names if chosen not in figures.FIGURES]
    if unknown:
        raise ValueError(f"No figure {unknown}; the run draws figures {list(figures.FIGURES)}.")
    drawn = {}
    for chosen in names:
        drawn[chosen] = str(figures.save_figure(figures.FIGURES[chosen](root), root / "figures" / f"figure_{chosen}.png"))
        _log(f"figure {chosen}: {drawn[chosen]}")
    receipt = root / "figures" / "receipt.json"
    earlier = json.loads(receipt.read_text(encoding="utf-8"))["figures"] if receipt.exists() else {}
    _json(receipt, {"figures": {**earlier, **drawn}, "matplotlib": matplotlib.__version__})


def stage_sae_labels(root: Path) -> None:
    """Supplement S2 on the compendium (CPU, after E3 is scored and `missing` has run): Biohub's ESMC-6B layer-60 SAE
    latents' labels summarized, and compared with Biohub's descriptions by version one's estimator
    (`tap_probing.units.sae_labels`)."""
    from huggingface_hub import hf_hub_download
    from tap_probing.units import missing, sae_labels

    started = time.perf_counter()
    output = root / "sae_labels"
    experiment, block, reference = experiments.HYPOTHESIS_SITES["esmc_6b"]
    key = experiments.site_keys(block, reference).sae
    stem = experiments.stem(experiment, key)
    scores = experiments.score_table(root, experiment, key)
    labels = pd.read_parquet(root / "passes" / experiment / "scores" / f"{stem}.labels_local.parquet")
    inputs = passes.PassInputs.read(root / "inputs")
    terms = (pd.read_parquet(root / "inputs" / "terms.parquet", columns=["token", "term_id", "label"]).sort_values("token")
             .merge(pd.read_parquet(root / "source" / "tables" / "vocabulary.parquet", columns=["term_id", "definition"]),
                    on="term_id", how="left", validate="one_to_one"))
    pin = sae_labels.FEATURE_TABLE
    table = Path(hf_hub_download(pin["repo"], pin["file"], repo_type="dataset", revision=pin["revision"],
                                 local_dir=output / "biohub"))
    described = sae_labels.descriptions(pd.read_parquet(table, columns=["feature_id", *sae_labels.DESCRIPTION_FIELDS, "category"]))
    chosen = sae_labels.representatives(scores, labels, inputs.vocabulary_size)
    frame, summary = sae_labels.agreement(described, chosen, sae_labels.term_documents(terms["label"], terms["definition"]))
    chosen.to_parquet(output / "representatives.parquet", index=False)
    frame.to_parquet(output / "agreement.parquet", index=False)
    labeled = sae_labels.label_summary(scores, labels, pd.read_parquet(missing.table_path(root, experiment, key)),
                                       inputs.vocabulary_size)
    _json(output / "receipt.json", {"latents": key, "feature_table": {**pin, "sha256": _sha256(table)}, "labels": labeled,
                                    "agreement": summary, "seconds": time.perf_counter() - started})
    arms = summary["arms"]
    _log("S2: matched win probability " + ", ".join(
        f"{arm} {arms[arm]['mean_matched_win_probability']:.4f} over {arms[arm]['included']}" for arm in sae_labels.ARMS
        if arms[arm]["included"]))


def stage_tables(root: Path) -> None:
    """The paper's tables from the run's outputs (`tap_probing.units.tables`), each a LaTeX tabular body in `RUN/tables`."""
    from tap_probing.units import tables

    written = tables.write_tables(root, root / "tables")
    _json(root / "tables" / "receipt.json", {"tables": [str(path) for path in written]})
    _log(f"tables: {', '.join(path.name for path in written)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("inputs", "pin-sweep", "parity", "pass", "score", "analyze", "missing", "dictionary",
                                          "causal", "near-misses", "exemplars", "steering", "concepts", "sparse-probes",
                                          "attribution", "payoff", "taxonomy-panel", "subspaces", "retest-inputs", "retest",
                                          "retest-esmc", "strength", "wiring", "coalitions", "sae-labels", "figures",
                                          "tables"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--experiment", choices=sorted(experiments.experiment_plans()))
    parser.add_argument("--model", choices=sorted({*experiments.dictionary_sites(), *experiments.RETEST_BUDGET_HOURS,
                                                   *experiments.esmc_retest_plans()}))
    parser.add_argument("--expect-digest")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--figure", help="the figure `figures` draws, `2` or `depth_atlas`; every figure when left out")
    options = parser.parse_args(argv)
    preregistration = yaml.safe_load(options.preregistration.read_text(encoding="utf-8"))
    if options.stage in ("pass", "score") and options.experiment is None:
        parser.error(f"{options.stage} needs --experiment")
    esmc_stages = ("dictionary", "concepts", "sparse-probes", "payoff", "taxonomy-panel", "subspaces", "coalitions")
    if options.stage in esmc_stages and options.model not in experiments.dictionary_sites():
        parser.error(f"{options.stage} needs --model {' or '.join(sorted(experiments.dictionary_sites()))}")
    if options.stage == "retest" and options.model not in experiments.RETEST_BUDGET_HOURS:
        parser.error(f"retest needs --model {' or '.join(sorted(experiments.RETEST_BUDGET_HOURS))}")
    if options.stage == "retest-esmc" and options.model not in experiments.esmc_retest_plans():
        parser.error(f"retest-esmc needs --model {' or '.join(sorted(experiments.esmc_retest_plans()))}")
    if options.stage in ("strength", "wiring") and options.model not in experiments.STRENGTH_SITES:
        parser.error(f"{options.stage} needs --model {' or '.join(sorted(experiments.STRENGTH_SITES))}")
    if options.stage == "inputs":
        if not options.expect_digest:
            parser.error("inputs needs --expect-digest, the split digest the launch record holds")
        stage_inputs(options.root, preregistration, options.expect_digest)
    elif options.stage == "pin-sweep":
        stage_pin_sweep(options.root, preregistration)
    elif options.stage == "parity":
        stage_parity(options.root, preregistration)
    elif options.stage == "pass":
        stage_pass(options.root, preregistration, options.experiment, options.pilot)
    elif options.stage == "score":
        stage_score(options.root, options.experiment)
    elif options.stage == "missing":
        stage_missing(options.root)
    elif options.stage == "dictionary":
        stage_dictionary(options.root, preregistration, options.model, options.pilot)
    elif options.stage == "causal":
        stage_causal(options.root, preregistration, options.pilot)
    elif options.stage == "near-misses":
        stage_near_misses(options.root, preregistration)
    elif options.stage == "exemplars":
        stage_exemplars(options.root, preregistration)
    elif options.stage == "steering":
        stage_steering(options.root, preregistration, options.pilot)
    elif options.stage == "concepts":
        stage_concepts(options.root, preregistration, options.model, options.pilot)
    elif options.stage == "sparse-probes":
        stage_sparse_probes(options.root, preregistration, options.model)
    elif options.stage == "attribution":
        stage_attribution(options.root, preregistration)
    elif options.stage == "payoff":
        stage_payoff(options.root, preregistration, options.model, options.pilot)
    elif options.stage == "taxonomy-panel":
        stage_taxonomy_panel(options.root, preregistration, options.model, options.pilot)
    elif options.stage == "subspaces":
        stage_subspaces(options.root, preregistration, options.model)
    elif options.stage == "retest-inputs":
        stage_retest_inputs(options.root)
    elif options.stage == "retest":
        stage_retest(options.root, preregistration, options.model, options.pilot)
    elif options.stage == "retest-esmc":
        stage_retest_esmc(options.root, preregistration, options.model, options.pilot)
    elif options.stage == "strength":
        stage_strength(options.root, preregistration, options.model, options.pilot)
    elif options.stage == "wiring":
        stage_wiring(options.root, preregistration, options.model)
    elif options.stage == "coalitions":
        stage_coalitions(options.root, preregistration, options.model, options.pilot)
    elif options.stage == "figures":
        stage_figures(options.root, options.figure)
    elif options.stage == "sae-labels":
        stage_sae_labels(options.root)
    elif options.stage == "tables":
        stage_tables(options.root)
    else:
        stage_analyze(options.root)


if __name__ == "__main__":
    main()
