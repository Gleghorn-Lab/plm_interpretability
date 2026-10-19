---

# Projection notice

Appended by `ws publish` from the research foundry workspace. **Do not edit this section.**

This repository is a projection. The canonical source is the workspace entity `tap_probing`
at `projects/tooling/tap_probing/README.md`.

## If you change code here

Changes made in this clone do not reach the workspace on their own. Report what you changed
and at which commit so it can be brought back with `ws import Gleghorn-Lab/tap_probing <sha>`. Do not
assume a later publish preserves your change: publish rebuilds this tree from the workspace
and refuses when GitHub HEAD is not the SHA it last recorded.

## Vendored code

`foundry.secrets_env` is vendored here from the workspace so this clone runs standalone. Do not edit a
vendored directory; edit it in the workspace and republish.

## Results

Log runs to Weights and Biases with the group set to the experiment id. That is how results
reach the workspace, through `ws capture`. Do not copy result files by hand.

---

# Coding standard

Appended by `ws publish` from the workspace's `docs/conventions/python.md`. **Do not edit this
section.** It binds every Python change here. The `ws` commands it names run only in the
workspace; in a clone, hold the same rules by review. Put any other guidance in the sections
above: AGENTS.md is this repository's only instruction file, and a CLAUDE.md is removed on publish.

## Imports

Every plain `import X` goes above every `from X import Y`. After the module docstring:
`from __future__` alone, then the plain block, then the `from` block. Each block runs
standard library, third party, then a blank line and the repository-local names. Two blank
lines after the whole thing.

```python
"""What this module is for."""

from __future__ import annotations

import argparse
import numpy as np
import torch

from pathlib import Path
from torch import Tensor

from .workflow import TrainingWorkflow


MAX_LENGTH = 1024
```

Run `ws imports --fix <your files>`. It rewrites a module into this shape from the syntax tree, so there
is nothing to hand-sort. Only the unbroken run of imports at the top of a module is reordered.
The first statement that is not an import ends the block, so a guarded, optional, registration,
or initialization-sensitive import stays exactly where it is. See
the standard decides import order.

## Types

- Every function gives each parameter and its return a precise type: concrete domain types and
  parameterized containers, `list[Path]` rather than `list`, `dict[str, float]` rather than
  `dict`. `ws lint` checks presence outside test files (`missing-annotation`) and a bare
  container in a signature (`bare-container`).
- A dictionary with fixed keys is a record: type it as a `TypedDict`, or make it a dataclass
  when it never crosses a serialization boundary. A bare `dict` or `list[dict]` for a record
  hides its contract.
- `Any` only at a genuinely dynamic boundary, such as parsed JSON before validation.
- Respect the Python version the code runs on: `requires-python` in the entity's configuration.

## Control flow and errors

- One narrow responsibility per function. Write for the supported inputs, and keep a defensive
  branch only when it protects a real requirement.
- Raise an ordinary exception for invalid external input. Use an assertion for an internal
  invariant.
- No broad exception handling, no silent fallback cascades, no speculative compatibility paths,
  and no wrapper that only renames another call (`ws lint`: `broad-except`, `silent-except`,
  `renaming-wrapper`).
- Extract a one-use helper only when it names a real concept or isolates a separately testable
  phase.
- Separate distinct guards, loops, exception handling, and logical phases with a blank line when
  that makes the function easier to scan.

## Naming

- Domain names over `data`, `result`, `info`, `manager`.
- Descriptive variables are lowercase snake case whatever their rank: `hidden_states`,
  `attention_mask`. Short mathematical symbols follow the numerical section.
- Genuinely constant configuration is descriptive all caps near the top of a runnable module.
- Keep public names, serialized fields, framework-discovered attributes, and established
  external notation, even when a better name exists.

`ws lint`'s `naming` rule checks snake case, CapWords classes, and capitals only for module
constants, and admits short symbols such as `X`, `QKV`, and `d_h`; `vague-name` flags exactly
`data`, `result`, `info`, and `manager`. Whether any other name is precise is reviewed.

## Comments and documentation

- Code, comments, docstrings, error messages, and documentation are as short as their purpose
  permits.
- Comment intent, assumptions, units, biological conventions, non-obvious mechanics, and design
  decisions. Delete a comment that narrates syntax or repeats a precise name and type.
- Write direct sentences in American English with stable terms. No em dashes.
- Base every claim about performance, compatibility, deprecation, tensor shapes, or behavior on
  code, tests, configuration, a benchmark, or current authoritative documentation.

## Numerical symbols and shape traces

Mandatory for NumPy, PyTorch, JAX, and similar array code, as part of the standard above. It
states in full the numerical rules of Logan's standards suite,
`~/.claude/skills/coding-standards/references/numerical-python.md`, with the symbols this
workspace adds. Every array-valued binding carries a shape comment. A stale shape comment is a
correctness defect, not a style issue.

`ws shapes` checks **presence** of the traces a typed signature demands, a shape comment for
every array parameter and every array return, in all the code lint covers; its tier-0 test is
`tests/tier0_smoke/test_shapes.py`. **Accuracy**, and the traces between input and return,
cannot be checked mechanically, so they are a blocking item on review.

### Trace every shape

Annotate every array-valued binding or transformation, so a reader follows the whole path
without running the code in their head:

- function inputs, in a comment at the top of the body or beside each parameter;
- creation and assignment;
- reassignment and in-place mutation;
- layer and method calls, with the unchanged shape after a layer that keeps it;
- reshape, transpose, indexing, slicing, and masking;
- reductions, concatenation, stacking, and broadcasting;
- returned arrays and tensors.

Put the shape on the same line when practical. Annotate the final binding of a chained
right-hand side once, not every subexpression. Use `()` for a scalar array or tensor.

```python
def forward(self, X: Tensor, attention_mask: Tensor) -> Tensor:
    # X: (b, l, d); attention_mask: (b, l)
    Q = self.query(X)  # (b, l, d)
    Q = Q.view(b, l, h, d_h)  # (b, l, h, d_h)
    Q = Q.transpose(1, 2)  # (b, h, l, d_h)
    scores = Q @ K.transpose(-2, -1)  # (b, h, l, l)
    hidden_states = scores @ V  # (b, h, l, d_h)
    X_pooled = hidden_states.mean(dim=(1, 2))  # (b, d_h)
    logits = self.classifier(X_pooled)  # (b, c)

    return logits  # (b, c)
```

Comment a scalar dimension where it is defined when later shapes use it:

```python
self.hidden_size = config.hidden_size  # d
self.num_heads = config.num_heads  # h
```

Derive every shape from the function's contract, the layer configuration, the tests, or the
operations upstream. Never copy a plausible shape without tracing it, and never invent a
dimension or a fixed rank. A dynamic dimension gets a stable symbol, defined near its first
use. When a rank or shape cannot be proven, write a documented symbol or an ellipsis, such as
`(..., d)`, and say so in the report of the change.

### Capitalization

- Scalars and dimensions are lowercase: `b`, `l`, `d`, `n`.
- Vectors are lowercase short symbols: `x`, `q`, `v`.
- Matrices and tensors of rank two or more are uppercase short symbols: `X`, `W`, `Q`, `K`, `V`.
  A batch of vectors is a matrix and takes an uppercase symbol.
- Compound symbols are allowed, such as `d_h`. Define a nonstandard symbol near its first use.
- Descriptive tensor names stay lowercase snake case, such as `hidden_states`. Do not rename a
  clear descriptive name to a single letter to satisfy the convention, and keep established
  domain notation when it is clearer for the reader.

### Symbols

| Symbol | Meaning |
|---|---|
| `b` | batch |
| `l` | sequence length |
| `d` | hidden width |
| `h` | attention heads |
| `d_h` | per-head width |
| `c` | classes |
| `n` | generic count |
| `q` | query proteins |
| `p` | proteome or target proteins |
| `t` | projection tower width |
| `f` | sparse-autoencoder codebook size |
| `k` | active sparse-autoencoder features per position |

The last five recur across this workspace's protein work and are defined once here so every
project uses the same letters. Before wave 0 each repository chose its own, which is a large
part of why two implementations of the PPI model were hard to compare. Dual-tower all-versus-all scoring
therefore reads:

```python
def score(A: Tensor, B: Tensor) -> Tensor:
    # A: (q, t); B: (p, t)
    return A @ B.transpose(-2, -1)  # (q, p)
```

And masked pooling over sparse features:

```python
def pool_sae(Z: Tensor, attention_mask: Tensor) -> Tensor:
    # Z: (b, l, f); attention_mask: (b, l)
    mask = attention_mask.unsqueeze(-1)              # (b, l, 1)
    summed = (Z * mask).sum(dim=1)                   # (b, f)
    return summed / mask.sum(dim=1).clamp(min=1)     # (b, f)
```

## Progress

Long work reports its progress. Logan finds progress output very useful, above all in training
and data processing pipelines, and a silent long job cannot be told apart from a hung one.
Every loop or stage that can run for more than a few seconds shows how far it has come: a
dataset build stage, a download, an embedding pass, a clustering or homology search, a training
or evaluation loop.

- A loop over a known number of items takes a `tqdm` bar, `from tqdm.auto import tqdm`, with a
  `desc` naming the stage, a `total` when the iterable has no length, and a `unit` when the items
  are not generic iterations.
- A stage without a known count prints one line when it starts and one when it ends, with the
  count it produced and the elapsed time, using `flush=True`.
- A training loop shows the step, the loss, and the throughput, on its bar with `set_postfix` or
  on a periodic line, and sends the same numbers to Weights and Biases.
- Progress goes to stderr, where `tqdm` writes, so stdout stays parseable. On Modal and remote
  machines the log is not a terminal: set `mininterval`, such as 30 seconds, so the log holds
  readable lines instead of thousands of redraws.
- A reusable function that may run inside another loop's bar takes a `progress` parameter
  instead of always drawing its own.
- `tqdm` must be in the environment that runs the code: the research lock pins it, and a Modal
  image or a `foundry` module's `requirements.txt` that imports it lists it. Where it is absent,
  print progress lines instead.
- A cleanup keeps the progress output a pipeline already has. Removing it changes behavior.

## Structure

- One primary responsibility per module. Split by ownership and dependency direction, never by
  line count.
- Keep helpers near the feature that owns them. Give an extracted module a small intentional
  public surface.
- Use a class when state and behavior form a meaningful domain object or workflow, and a
  function or typed value for a simple transformation or record.
- Keep entry points thin: parse, construct, invoke a visible sequence, report.
