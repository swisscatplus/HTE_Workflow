#!/usr/bin/env python3
"""
bo_runner.py
Use BoTorch to propose a batch of experiments from an optimizer-spec dict (parsed HCI),
then write a synthesis.json via synthesis_writer.py.

Requires: torch, botorch, gpytorch
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional

import argparse
import math
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition import ExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

from json_handling.library_and_hci_adapter import hci_to_optimizer_dict
from hte_workflow.paths import DATA_DIR, OUT_DIR, ensure_dirs


# ---------------------------
# Utilities
# ---------------------------

def _load_json(p: Union[str, Path]) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))

def _save_json(obj: Any, p: Union[str, Path]) -> None:
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

# ---------------------------
# Encoding / decoding of parameters
# ---------------------------

class SpaceEncoder:
    """
    Build a continuous tensor space for BoTorch:
      - Categorical params -> one-hot blocks
      - Numeric params -> [0, 1] scaled segments

    Provides encode(config_dict) -> X, and decode(X) -> config_dict.
    """
    def __init__(self, parameters: List[Dict[str, Any]]):
        self.params = parameters
        self.blocks: List[Tuple[str, str, Any]] = []  # (name, kind, meta)
        self.dim = 0

        for p in self.params:
            if p["type"] == "categorical":
                k = len(p["choices"])
                self.blocks.append((p["name"], "cat", {"choices": p["choices"], "k": k}))
                self.dim += k
            elif p["type"] == "numerical":
                lo, hi = float(p["low"]), float(p["high"])
                self.blocks.append((p["name"], "num", {"low": lo, "high": hi, "unit": p.get("unit", ""), "step": p.get("step")}))
                self.dim += 1
            else:
                raise ValueError(f"Unsupported param type: {p}")

    def encode_one(self, cfg: Dict[str, Any]) -> torch.Tensor:
        xs: List[float] = []
        for name, kind, meta in self.blocks:
            if kind == "cat":
                # one-hot
                val = cfg[name]
                choices = meta["choices"]
                one = [0.0] * meta["k"]
                idx = choices.index(val)
                one[idx] = 1.0
                xs.extend(one)
            else:
                # numeric scaled to [0, 1]
                lo, hi = meta["low"], meta["high"]
                v = float(cfg[name])
                xs.append((v - lo) / (hi - lo) if hi > lo else 0.0)
        return torch.tensor(xs, dtype=torch.double)

    def decode_one(self, x: torch.Tensor) -> Dict[str, Any]:
        x = x.detach().cpu().double()
        cfg: Dict[str, Any] = {}
        i = 0
        for name, kind, meta in self.blocks:
            if kind == "cat":
                k = meta["k"]
                block = x[i:i+k]
                idx = int(torch.argmax(block).item())
                cfg[name] = meta["choices"][idx]
                i += k
            else:
                lo, hi = meta["low"], meta["high"]
                v01 = float(x[i].clamp(0.0, 1.0).item())
                val = lo + v01 * (hi - lo)
                # snap to step if provided
                step = meta.get("step")
                if step:
                    n = round((val - lo) / step)
                    val = lo + n * step
                cfg[name] = val
                i += 1
        return cfg

    def bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(self.dim, dtype=torch.double), torch.ones(self.dim, dtype=torch.double)

# ---------------------------
# Training data extraction
# ---------------------------

def _default_config(parameters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick the first categorical choice and mid of numeric ranges (used for warm starts)."""
    cfg: Dict[str, Any] = {}
    for p in parameters:
        if p["type"] == "categorical":
            cfg[p["name"]] = p["choices"][0]
        else:
            lo, hi = float(p["low"]), float(p["high"])
            cfg[p["name"]] = (lo + hi) / 2.0
    return cfg

def build_training_tensors(
    encoder: SpaceEncoder,
    results: List[Dict[str, Any]],
    parameters: List[Dict[str, Any]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    results format:
      [
        {"params": {"group:catalyst": "Pd(PPh3)4", "eq:catalyst": 0.02, "global:temperature": 80}, "objective": 0.45},
        ...
      ]
    Returns (X, y) in double tensors or (None, None) if no data.
    """
    if not results:
        return None, None
    Xs, ys = [], []
    for r in results:
        cfg = _default_config(parameters) | r.get("params", {})
        Xs.append(encoder.encode_one(cfg))
        ys.append([float(r["objective"])])
    X = torch.stack(Xs, dim=0)
    y = torch.tensor(ys, dtype=torch.double).view(-1, 1)
    return X, y

# ---------------------------
# Proposal
# ---------------------------

def propose_batch(
    encoder: SpaceEncoder,
    parameters: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    q: int,
    use_noisy_ei: bool = True,
    seed: int = 1234,
) -> torch.Tensor:
    torch.manual_seed(seed)
    bounds = torch.stack(encoder.bounds())

    X_train, y_train = build_training_tensors(encoder, results, parameters)
    if X_train is None:
        # No data yet: use Sobol random batch
        sobol = torch.quasirandom.SobolEngine(dimension=encoder.dim, scramble=True, seed=seed)
        return sobol.draw(q).to(torch.double)

    # Fit GP
    gp = SingleTaskGP(X_train, y_train)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # Acquisition function
    if use_noisy_ei:
        acq = qNoisyExpectedImprovement(model=gp, X_baseline=X_train, sampler=SobolQMCNormalSampler(128))
    else:
        best_f = y_train.max()
        acq = qExpectedImprovement(model=gp, best_f=best_f, sampler=SobolQMCNormalSampler(128))

    # Optimize acqf
    candidates, _ = optimize_acqf(
        acq_function=acq,
        bounds=bounds,
        q=q,
        num_restarts=10,
        raw_samples=256,
        options={"batch_limit": 5, "maxiter": 200},
    )
    return candidates.detach()

# ---------------------------
# Convert decoded configs to per-well selections
# ---------------------------

def selections_from_configs(opt_spec: Dict[str, Any], configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Turn decoded param configs (with names like 'group:catalyst', 'eq:catalyst', 'global:temperature', ...)
    into a compact selection dict for each well that keeps group picks + numericals.
    """
    group_names = [g["name"] for g in opt_spec["groups"]]
    out = []
    for cfg in configs:
        sel = {"groups": {}, "globals": {}}
        for k, v in cfg.items():
            if k.startswith("group:"):
                gname = k.split(":", 1)[1]
                sel["groups"][gname] = {"member": v}
            elif k.startswith("eq:"):
                gname = k.split(":", 1)[1]
                sel["groups"].setdefault(gname, {})["equivalents"] = v
            elif k.startswith("global:"):
                gname = k.split(":", 1)[1]
                sel["globals"][gname] = v
        out.append(sel)
    return out

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Run BoTorch BO on optimizer dict and emit synthesis.json")
    ap.add_argument("--hci_file", required=True, help="HCI File")
    ap.add_argument("--results", help="Past results JSON (list of {params, objective})")
    ap.add_argument("--out", required=True, help="Path to synthesis.json (will be written by synthesis_writer)")
    # Optional chemistry context for quantity computation in writer
    ap.add_argument("--limiting-name", help="Name of limiting chemical (matching a group member's chemicalName)")
    ap.add_argument("--limiting-moles", type=float, help="Moles of limiting reagent per well (e.g., 2e-6)")
    ap.add_argument("--limiting-conc", type=float, help="If given (M), combined with --well-volume-uL to compute moles")
    ap.add_argument("--well-volume-uL", type=float, help="Total volume per well (µL) for concentration-based dosing")

    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--data-dir", default=str(DATA_DIR), help="Directory with data files for layout_parser")

    args = ap.parse_args()
    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    # Load HCI spec
    opt_spec = hci_to_optimizer_dict(str(out_dir/args.hci_file))
    plate_size = opt_spec.get("metadata").get("batch").get("plate_size")


    past = _load_json(args.results) if args.results else []

    parameters = opt_spec["parameters"]
    encoder = SpaceEncoder(parameters)

    # Propose q = plate-size
    Xq = propose_batch(encoder, parameters, past, q=plate_size, use_noisy_ei=True)
    cfgs = [encoder.decode_one(Xq[i]) for i in range(Xq.shape[0])]

    # Convert to selections (groups/globals)
    sels = selections_from_configs(opt_spec, cfgs)

    # Hand off to writer
    from json_handling.synthesis_writer import write_synthesis_json
    write_synthesis_json(
        opt_spec=opt_spec,
        selections=sels,
        out_path=args.out,
        plate_size=plate_size,
        limiting_name=args.limiting_name,
        limiting_moles=args.limiting_moles,
        limiting_conc=args.limiting_conc,
        well_volume_uL=args.well_volume_uL,
    )
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
