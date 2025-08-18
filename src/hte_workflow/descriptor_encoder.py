from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import torch

class DescriptorSpaceEncoderAuto:
    """
    Auto mode per group:
      - if all members have descriptors with the same keys -> use one-hot + descriptors ("both")
      - else -> use one-hot only
    Descriptors are z-scored per *global* pool (across all groups using descriptors).
    """
    def __init__(self, parameters: List[Dict[str, Any]], opt_spec: Dict[str, Any], impute: float | str = "mean"):
        self.params = parameters
        self.opt_spec = opt_spec
        self.impute = impute  # "mean" or a float

        self.blocks: List[Tuple[str, str, Dict[str, Any]]] = []  # (name, kind, meta)
        self.dim = 0

        # Per-categorical param data
        self._desc_tables: Dict[str, torch.Tensor] = {}   # param_name -> [k, d]
        self._desc_dims: Dict[str, int] = {}              # param_name -> d
        self._desc_means: Optional[torch.Tensor] = None   # global mean over all rows
        self._desc_stds: Optional[torch.Tensor] = None    # global std over all rows

        for p in self.params:
            if p["type"] == "categorical":
                name = p["name"]                 # e.g., "group:catalyst"
                choices = list(p["choices"])     # list[str] of member chemicalName
                gname = name.split(":", 1)[1]

                # Detect common descriptor keys across all members of this group
                desc_keys = self._common_descriptor_keys_for_group(gname, choices)
                use_desc = bool(desc_keys)  # if empty -> no descriptors in this group

                meta = {"choices": choices, "k": len(choices), "group_name": gname, "desc_keys": desc_keys}
                self.blocks.append((name, "cat", meta))

                # one-hot dims always present
                self.dim += meta["k"]

                # if using descriptors, build table now (un-normalized for the moment)
                if use_desc:
                    D = self._build_descriptor_table(gname, choices, desc_keys)  # [k, d]
                    self._desc_tables[name] = D
                    self._desc_dims[name] = D.shape[1]
                    self.dim += D.shape[1]

            elif p["type"] == "numerical":
                lo, hi = float(p["low"]), float(p["high"])
                meta = {"low": lo, "high": hi, "unit": p.get("unit", ""), "step": p.get("step")}
                self.blocks.append((p["name"], "num", meta))
                self.dim += 1
            else:
                raise ValueError(f"Unsupported param type: {p}")

        # Global z-score across ALL descriptor rows from all groups (if any)
        if self._desc_tables:
            allD = torch.cat(list(self._desc_tables.values()), dim=0)  # [sum_k, d_max?]  (each has same d for its group)
            means = allD.mean(dim=0)
            stds  = allD.std(dim=0)
            stds[stds == 0] = 1.0
            self._desc_means, self._desc_stds = means, stds
            # normalize each table
            for key, D in self._desc_tables.items():
                self._desc_tables[key] = (D - means) / stds

    # ---------- helpers ----------

    def _find_member_reference(self, group_name: str, member_name: str) -> Optional[Dict[str, Any]]:
        for g in self.opt_spec.get("groups", []):
            if g["name"] != group_name:
                continue
            for m in g.get("members", []) or []:
                ref = m.get("reference") or {}
                if ref.get("chemicalName") == member_name:
                    return ref
        return None

    def _common_descriptor_keys_for_group(self, group_name: str, choices: List[str]) -> List[str]:
        """Return the set of keys that appear in EVERY member's descriptors (non-empty)."""
        keys_common: Optional[set] = None
        for nm in choices:
            ref = self._find_member_reference(group_name, nm) or {}
            desc = ref.get("descriptors")
            if not isinstance(desc, dict) or not desc:
                return []  # no descriptors for this member -> fallback to one-hot
            keys = set(desc.keys())
            keys_common = keys if keys_common is None else (keys_common & keys)
            if not keys_common:
                return []  # inconsistent keys -> fallback to one-hot (your schema says they should match)
        return sorted(keys_common)

    def _build_descriptor_table(self, group_name: str, choices: List[str], keys: List[str]) -> torch.Tensor:
        rows = []
        for nm in choices:
            ref  = self._find_member_reference(group_name, nm) or {}
            desc = (ref.get("descriptors") or {})
            vec = []
            for k in keys:
                v = desc.get(k, None)
                if v is None:
                    if self.impute == "mean":
                        vec.append(float("nan"))  # fill later
                    else:
                        vec.append(float(self.impute))
                else:
                    vec.append(float(v))
            rows.append(vec)
        D = torch.tensor(rows, dtype=torch.double)
        if self.impute == "mean" and torch.isnan(D).any():
            col_means = torch.nanmean(D, dim=0)
            col_means = torch.nan_to_num(col_means, nan=0.0)
            idx = torch.where(torch.isnan(D))
            D[idx] = col_means[idx[1]]
        return D

    # ---------- API ----------

    def encode_one(self, cfg: Dict[str, Any]) -> torch.Tensor:
        xs: List[float] = []
        for name, kind, meta in self.blocks:
            if kind == "cat":
                # one-hot (always)
                choices = meta["choices"]
                one = [0.0] * meta["k"]
                idx = choices.index(cfg[name])
                one[idx] = 1.0
                xs.extend(one)
                # descriptors if this group has them
                if meta["desc_keys"]:
                    D = self._desc_tables[name]  # already normalized
                    xs.extend(D[idx].tolist())
            else:
                lo, hi = meta["low"], meta["high"]
                v = float(cfg[name])
                xs.append((v - lo) / (hi - lo) if hi > lo else 0.0)
        return torch.tensor(xs, dtype=torch.double)

    def decode_one(self, x: torch.Tensor) -> Dict[str, Any]:
        """Use one-hot portion to decode categories; invert scaling for numericals."""
        x = x.detach().cpu().double()
        cfg: Dict[str, Any] = {}
        i = 0
        for name, kind, meta in self.blocks:
            if kind == "cat":
                k = meta["k"]
                block = x[i:i+k]; i += k
                idx = int(torch.argmax(block).item())
                cfg[name] = meta["choices"][idx]
                # skip descriptor slice (if present)
                if meta["desc_keys"]:
                    i += len(meta["desc_keys"])
            else:
                v01 = float(x[i].clamp(0.0, 1.0).item()); i += 1
                lo, hi = meta["low"], meta["high"]
                val = lo + v01 * (hi - lo)
                step = meta.get("step")
                if step:
                    n = round((val - lo) / step)
                    val = lo + n * step
                cfg[name] = val
        return cfg

    def bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(self.dim, dtype=torch.double), torch.ones(self.dim, dtype=torch.double)
