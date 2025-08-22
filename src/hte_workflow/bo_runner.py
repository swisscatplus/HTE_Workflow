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
import random
from itertools import product

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
from hte_workflow.descriptor_encoder import DescriptorSpaceEncoderAuto


# ---------------------------
# Utilities
# ---------------------------

def _load_json(p: Union[str, Path]) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))

def _save_json(obj: Any, p: Union[str, Path]) -> None:
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

#----------------------------
# Fix temperature per plate
#----------------------------

def _find_param_index_numeric_auto(encoder: DescriptorSpaceEncoderAuto, param_name: str) -> int:
    """
    Return the column index of the numeric parameter 'param_name' in the encoder's
    concatenated tensor space. Handles DescriptorSpaceEncoderAuto blocks:
      - cat blocks: one-hot (k dims) + optional descriptor dims (len(desc_keys))
      - num blocks: 1 dim each
    """
    i = 0
    for name, kind, meta in encoder.blocks:
        if kind == "cat":
            i += int(meta.get("k", 0))
            # descriptors added after one-hot for this categorical group
            i += len(meta.get("desc_keys", []))
        else:  # numeric
            if name == param_name:
                return i
            i += 1
    raise ValueError(f"Numeric parameter not found in encoder: {param_name}")

def _temperature_range_from_spec(opt_spec: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """Return {'min':..., 'max':..., 'unit':...} for temperature, from optimizer dict or HCI."""
    rng = (opt_spec.get("globals") or {}).get("temperature")
    if rng is None:
        c = opt_spec.get("hasCampaign", {})
        rng = (c.get("hasRanges") or {}).get("temperature")
    if not rng:
        return None
    return {"min": float(rng["min"]), "max": float(rng["max"]), "unit": rng.get("unit", "C")}

def _temperature_candidates(rng: Dict[str, float], how: str = "grid", k: int = 7) -> List[float]:
    lo, hi = float(rng["min"]), float(rng["max"])
    if how == "grid":
        if k < 2:
            return [(lo + hi) / 2.0]
        step = (hi - lo) / (k - 1)
        return [lo + i * step for i in range(k)]
    # random
    random.seed(123)
    return [random.uniform(lo, hi) for _ in range(k)]


def _optimize_batch_with_fixed_numeric(acq_function, encoder, q: int, fixed_index: int, fixed_value_01: float):
    """
    Optimize acquisition with one coordinate fixed in [0,1] across the q-batch.
    """
    lb, ub = encoder.bounds()
    candidates, best_val = optimize_acqf(
        acq_function=acq_function,
        bounds=torch.stack((lb, ub)),
        q=q,
        num_restarts=10,
        raw_samples=256,
        fixed_features={fixed_index: float(fixed_value_01)},
        options={"batch_limit": 5, "maxiter": 200},
    )
    return candidates.detach(), float(best_val.detach().item()) if torch.is_tensor(best_val) else float(best_val)

def propose_plate_with_temperature_BO_auto(
    encoder: DescriptorSpaceEncoderAuto,
    parameters_all: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    q: int,
    temp_range: Dict[str, float],
    temp_param_name: str = "global:temperature",
    use_noisy_ei: bool = True,
):
    """
    Scan candidate temperatures; for each, optimize q-batch with the temperature coordinate fixed.
    Choose the temperature that yields the highest acquisition value.
    Returns (best_T, candidates_tensor).
    """
    # Build training data on *full* space (includes temperature)
    X_train, y_train = build_training_tensors(encoder, results, parameters_all)

    # No data yet: Sobol init, force mid temperature
    if X_train is None:
        sobol = torch.quasirandom.SobolEngine(dimension=encoder.dim, scramble=True, seed=123)
        Xq = sobol.draw(q).to(torch.double)
        t_idx = _find_param_index_numeric_auto(encoder, temp_param_name)
        T01 = 0.5
        Xq[:, t_idx] = T01
        lo, hi = float(temp_range["min"]), float(temp_range["max"])
        T_physical = lo + T01 * (hi - lo)
        return T_physical, Xq

    # Fit GP
    gp = SingleTaskGP(X_train, y_train)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    if use_noisy_ei:
        acq = qNoisyExpectedImprovement(model=gp, X_baseline=X_train, sampler=SobolQMCNormalSampler(128))
    else:
        acq = qExpectedImprovement(model=gp, best_f=y_train.max(), sampler=SobolQMCNormalSampler(128))

    # Fixed coordinate index for temperature
    t_idx = _find_param_index_numeric_auto(encoder, temp_param_name)
    lo, hi = float(temp_range["min"]), float(temp_range["max"])

    best_val = -1e18
    best_T = None
    best_cand = None

    for T in _temperature_candidates(temp_range, how="grid", k=7):  # 7 candidates defined by k
        T01 = (T - lo) / (hi - lo) if hi > lo else 0.0
        cand, val = _optimize_batch_with_fixed_numeric(acq, encoder, q=q, fixed_index=t_idx, fixed_value_01=T01)
        if val > best_val:
            best_val, best_T, best_cand = val, T, cand

    return best_T, best_cand

# ---------------------------
# Fixing other numeric parameters (time)
# ---------------------------

def _range_from_spec(opt_spec: Dict[str, Any], key: str) -> Optional[Dict[str, float]]:
    rng = (opt_spec.get("globals") or {}).get(key)
    if rng is None:
        c = opt_spec.get("hasCampaign", {})
        rng = (c.get("hasRanges") or {}).get(key)
    if not rng:
        return None
    return {"min": float(rng["min"]), "max": float(rng["max"]), "unit": rng.get("unit", "")}

def _candidates_for_range(rng: Dict[str, float], how: str = "grid", k: int = 7) -> list[float]:
    lo, hi = rng["min"], rng["max"]
    if how == "grid":
        if k < 2: return [(lo + hi) / 2.0]
        step = (hi - lo) / (k - 1)
        return [lo + i * step for i in range(k)]
    import random
    random.seed(123)
    return [random.uniform(lo, hi) for _ in range(k)]

def propose_plate_with_plate_globals_BO_auto(
    *,
    encoder,
    parameters_all: list[dict],
    results: list[dict],
    q: int,
    plate_globals: dict[str, dict],   # e.g. {"global:temperature": {"min":..,"max":..}, "global:time": {...}}
    how: str = "grid",
    k_per_dim: int = 7,
    use_noisy_ei: bool = True,
):
    # training data / model
    X_train, y_train = build_training_tensors(encoder, results, parameters_all)
    if X_train is None:
        # Sobol init; fix coords to midpoints
        sobol = torch.quasirandom.SobolEngine(dimension=encoder.dim, scramble=True, seed=123)
        Xq = sobol.draw(q).to(torch.double)
        for pname, rng in plate_globals.items():
            idx = _find_param_index_numeric_auto(encoder, pname)
            lo, hi = rng["min"], rng["max"]
            Xq[:, idx] = 0.5 if hi > lo else 0.0
        # decode physical values for reporting
        chosen = {pname: (rng["min"] + rng["max"]) / 2.0 for pname, rng in plate_globals.items()}
        return chosen, Xq

    # default Kernel is Matérn 5/2 with ARD
    gp = SingleTaskGP(X_train, y_train)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # MC based acquisition function for noise handling
    acq = qNoisyExpectedImprovement(model=gp, X_baseline=X_train, sampler=SobolQMCNormalSampler(128)) \
          if use_noisy_ei else \
          qExpectedImprovement(model=gp, best_f=y_train.max(), sampler=SobolQMCNormalSampler(128))

    # build candidate grids (physical values) and indices
    names = list(plate_globals.keys())
    grids = [_candidates_for_range(plate_globals[n], how=how, k=k_per_dim) for n in names]
    idxs  = {n: _find_param_index_numeric_auto(encoder, n) for n in names}
    mins  = {n: plate_globals[n]["min"] for n in names}
    maxs  = {n: plate_globals[n]["max"] for n in names}

    best_val = -1e18
    best_phys = None
    best_cand = None

    for phys_tuple in product(*grids):
        fixed_features = {}
        for n, phys in zip(names, phys_tuple):
            lo, hi = mins[n], maxs[n]
            fixed_features[idxs[n]] = 0.0 if hi <= lo else (phys - lo) / (hi - lo)

        lb, ub = encoder.bounds()
        cand, val = optimize_acqf(
            acq_function=acq,  # Samples Monte-Carlo q EI for batched experiments
            bounds=torch.stack((lb, ub)),
            q=q,
            num_restarts=10,
            raw_samples=256,
            fixed_features=fixed_features,
            options={"batch_limit": 5, "maxiter": 200},
        )
        val_f = float(val.detach().item()) if torch.is_tensor(val) else float(val)
        if val_f > best_val:
            best_val = val_f
            best_phys = {n: v for n, v in zip(names, phys_tuple)}
            best_cand = cand.detach()

    return best_phys, best_cand

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
    ap.add_argument("--hci-file", required=True, help="HCI File")
    ap.add_argument("--results", help="Past results JSON (list of {params, objective})")
    ap.add_argument("--out", required=True, help="Path to synthesis.json (will be written by synthesis_writer)")
    # Optional chemistry context for quantity computation in writer
    ap.add_argument("--limiting-name", help="Name of limiting chemical (matching a group member's chemicalName)")
    ap.add_argument("--limiting-moles", type=float, help="Moles of limiting reagent per well (e.g., 2e-6)")
    ap.add_argument("--well-volume-uL", type=float, help="Total volume per well (µL) for concentration-based dosing")
    ap.add_argument("--use-descriptors", default = True, help="Use descriptors in encoding (default: True)")
    ap.add_argument("--fix-plate-temperature", default = True)
    ap.add_argument("--fix-plate-time", default = True, help="Fix plate time to a single value (default: False)")

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

    if args.use_descriptors:
        # Use descriptor encoder
        encoder = DescriptorSpaceEncoderAuto(parameters, opt_spec)
    else:
        encoder = SpaceEncoder(parameters)

    temp_rng = _temperature_range_from_spec(opt_spec)
    if temp_rng is None:
        raise SystemExit("No temperature range found (hasRanges.temperature).")

    if args.fix_plate_temperature and not args.fix_plate_time:
        best_T, Xq = propose_plate_with_temperature_BO_auto(
            encoder=encoder,
            parameters_all=opt_spec["parameters"],
            results=past,  # your prior results (can be empty)
            q=plate_size,
            temp_range=temp_rng,
            temp_param_name="global:temperature",
            use_noisy_ei=True,
        )
        chosen_globals = None
    elif args.fix_plate_time and args.fix_plate_temperature:
        plate_globals = {}
        temp_rng = _range_from_spec(opt_spec, "temperature")
        if temp_rng: plate_globals["global:temperature"] = temp_rng
        time_rng = _range_from_spec(opt_spec, "time")
        if time_rng: plate_globals["global:time"] = time_rng

        if not plate_globals:
            raise SystemExit("No plate-global ranges found (expected 'temperature' and/or 'time').")

        chosen_globals, Xq = propose_plate_with_plate_globals_BO_auto(
            encoder=encoder,
            parameters_all=opt_spec["parameters"],
            results=past,
            q=plate_size,
            plate_globals=plate_globals,  # fixes these coords during optimization
            how="grid",
            k_per_dim=5,  # 5x5 grid for (temp,time). Adjust as needed.
            use_noisy_ei=True,
        )
        best_T = None
    else:
        # Propose q = plate-size
        chosen_globals = None
        best_T = None
        Xq = propose_batch(encoder, parameters, past, q=plate_size, use_noisy_ei=True)

    cfgs = [encoder.decode_one(Xq[i]) for i in range(Xq.shape[0])]

    # Convert to selections (groups/globals)
    sels = selections_from_configs(opt_spec, cfgs)

    if best_T is not None:
        for s in sels:
            s.setdefault("globals", {})
            s["globals"]["temperature"] = best_T
    if chosen_globals is not None:
        for s in sels:
            s.setdefault("globals", {})
            for disp_name, phys in chosen_globals.items():
                key = disp_name.split(":", 1)[1]  # "global:temperature" -> "temperature"
                s["globals"][key] = phys

    # Hand off to writer
    from json_handling.synthesis_writer import write_synthesis_json
    write_synthesis_json(
        opt_spec=opt_spec,
        selections=sels,
        out_path=str(out_dir/args.out),
        plate_size=plate_size,
        limiting_name=args.limiting_name,
        limiting_moles=args.limiting_moles,
        well_volume_uL=args.well_volume_uL,
    )
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
