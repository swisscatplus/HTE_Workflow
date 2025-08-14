"""
chemspace_builder.py
Create an HCI-style chemical-space JSON:
{
  "hasChemicalSpace": {
    "name": "...", "description": "...", "version": "1.0",
    "hasGroups": [ { ... } ],
    "hasRanges": { "temperature": {...}, "concentration": {...}, ... }
  }
}

- Chemicals are referenced by name and enriched from a fixed library (JSON/CSV).
- Groups are arbitrary (e.g., catalyst, solvent, additive).
- Each member has an equivalents range.
- Global ranges (temperature, concentration, etc.) live in hasRanges.

CLI:
  python chemspace_builder.py build --library CHEMS.json --spec SPEC.json --out SPACE.json
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import csv, json, argparse

from hte_workflow.paths import DATA_DIR, OUT_DIR, ensure_dirs


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class Quantity:
    value: float
    unit: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"value": self.value, "unit": self.unit}

@dataclass
class RangeSpec:
    """Generic numeric range with optional step."""
    min: float
    max: float
    unit: str = ""
    step: Optional[float] = None

    def to_json(self) -> Dict[str, Any]:
        out = {"min": self.min, "max": self.max, "unit": self.unit}
        if self.step is not None:
            out["step"] = self.step
        return out

@dataclass
class ChemicalRecord:
    chemicalID: str
    chemicalName: str
    CASNumber: str
    molecularMass: Quantity
    smiles: str
    Inchi: str
    molecularFormula: str
    swissCatNumber: str
    physicalstate: str
    density: Optional[Quantity] = None
    concentration: Optional[Quantity] = None  # optional, e.g., for solutions
    descriptors: Optional[Dict[str, Any]] = None  # optional, e.g., for descriptors
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_reference(self) -> Dict[str, Any]:
        """Minimal reference block to embed in chemical-space members."""
        ref = {
            "chemicalID": self.chemicalID,
            "chemicalName": self.chemicalName,
            "CASNumber": self.CASNumber,
            "molecularMass": self.molecularMass.to_json(),
            "smiles": self.smiles,
            "Inchi": self.Inchi,
            "molecularFormula": self.molecularFormula,
            "swissCatNumber": self.swissCatNumber,
            "physicalstate": self.physicalstate,
        }
        if self.descriptors is not None:
            ref["descriptors"] = self.descriptors
        if self.density:
            ref["density"] = self.density.to_json()
        if self.concentration:
            ref["concentration"] = self.concentration.to_json()
        if self.extras:
            ref["extras"] = self.extras
        return ref

@dataclass
class GroupMember:
    reference: Dict[str, Any]        # From ChemicalRecord.to_reference()
    overrides: Dict[str, Any] = field(default_factory=dict)  # optional free-form

    def to_json(self) -> Dict[str, Any]:
        out = {
            "reference": self.reference,
        }
        if self.overrides:
            out["overrides"] = self.overrides
        return out

@dataclass
class Group:
    groupName: str
    equivalents: RangeSpec  # optional, e.g., for fixed ranges
    fixed: bool = True              # If True, always included at a chosen value
    description: str = ""
    selectionMode: str = "one-of"     # e.g., "one-of", "any", "at-least-one"
    members: List[GroupMember] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "groupName": self.groupName,
            "description": self.description,
            "selectionMode": self.selectionMode,
            "members": [m.to_json() for m in self.members],
            "equivalents": self.equivalents.to_json(),
            "fixed": self.fixed,
        }

@dataclass
class Objective:
    criteria: str
    condition: str
    description: str
    objectiveName: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "criteria": self.criteria,
            "condition": self.condition,
            "description": self.description,
            "objectiveName": self.objectiveName,
        }

@dataclass
class Batch:
    batchID: str
    batchName: str
    reactionType: str
    reactionName: str
    optimizationType: str
    link: str
    platesize: int
    maxIterations: Optional[int] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "batchID": self.batchID,
            "batchName": self.batchName,
            "reactionType": self.reactionType,
            "reactionName": self.reactionName,
            "optimizationType": self.optimizationType,
            "link": self.link,
            "platesize": self.platesize,
            "maxIterations": self.maxIterations,
        }

@dataclass
class Campaign:
    campaignName: str
    description: str
    objective: str
    campaignClass: str
    type: str
    reference: str
    hasBatch: Dict[str, Any] = field(default_factory=dict)
    hasObjective: Dict[str, Any] = field(default_factory=dict)
    hasChemical: List[ChemicalRecord] = field(default_factory=list)
    hasGroups: List[Group] = field(default_factory=list)
    hasRanges: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "hasCampaign": {
                "campaignName": self.campaignName,
                "description": self.description,
                "objective": self.objective,
                "campaignClass": self.campaignClass,
                "type": self.type,
                "reference": self.reference,
                "hasBatch": self.hasBatch,
                "hasObjective": self.hasObjective,
                "hasChemical": [c.to_reference() for c in self.hasChemical],
                "hasGroups": [g.to_json() for g in self.hasGroups],
                "hasRanges": self.hasRanges,
            }
        }


# -----------------------------
# Library loader (by chemical name)
# -----------------------------

class ChemicalLibrary:
    """
    Load a fixed chemicals library from JSON or CSV.
    JSON: list or mapping. CSV: headers below are expected/flexible.
    Required fields (if available): chemicalName (used as key), chemicalID, CASNumber,
    molecularMass_value, molecularMass_unit, smiles, Inchi, molecularFormula, swissCatNumber,
    density_value, density_unit
    """

    def __init__(self, records: Dict[str, ChemicalRecord]):
        self._by_name = {k.lower(): v for k, v in records.items()}

    @staticmethod
    def _q_from_row(prefix: str, row: Dict[str, str]) -> Optional[Quantity]:
        v_key, u_key = f"{prefix}_value", f"{prefix}_unit"
        if v_key in row and row[v_key] not in (None, "", "NA", "na"):
            try:
                v = float(row[v_key])
            except ValueError:
                return None
            return Quantity(v, row.get(u_key, ""))
        return None

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ChemicalLibrary":
        path = Path(path)
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        items = data.values() if isinstance(data, dict) else data
        recs: Dict[str, ChemicalRecord] = {}
        for it in items:
            mm_raw = it.get("molecularMass", {})
            mm = Quantity(float(mm_raw.get("value", 0.0)), mm_raw.get("unit","g/mol"))
            d_raw = it.get("density")
            dens = Quantity(float(d_raw["value"]), d_raw.get("unit","")) if isinstance(d_raw, dict) else None
            name = it.get("chemicalName","")
            physicalstate = it.get("physicalstate")
            conc_raw = it.get("concentration")
            conc = Quantity(float(conc_raw["value"]), conc_raw.get("unit","")) if isinstance(conc_raw, dict) else None
            recs[name] = ChemicalRecord(
                chemicalID=str(it.get("chemicalID", name)),
                chemicalName=name,
                CASNumber=it.get("CASNumber",""),
                molecularMass=mm,
                smiles=it.get("smiles",""),
                Inchi=it.get("Inchi",""),
                molecularFormula=it.get("molecularFormula",""),
                swissCatNumber=it.get("swissCatNumber",""),
                density=dens,
                physicalstate= physicalstate,
                concentration=conc,
                extras={k:v for k,v in it.items() if k not in {
                    "chemicalID","chemicalName","CASNumber","molecularMass","smiles",
                    "Inchi","molecularFormula","swissCatNumber","density", "concentration", "pysicalstate"
                }}
            )
        return cls(recs)

    def get_by_name(self, name: str) -> ChemicalRecord:
        rec = self._by_name.get(name.lower())
        if not rec:
            raise KeyError(f"Chemical not found in library by name: {name}")
        return rec

def load_library(path: Union[str, Path]) -> ChemicalLibrary:
    p = Path(path)
    return ChemicalLibrary.from_json(p)

# -----------------------------
# Build chemical space from a compact spec
# -----------------------------
"""
SPEC.json (example):

{
  "name": "My Catalysis Space",
  "description": "Suzuki scope space",
  "ranges": {
    "temperature": {"min": 25, "max": 120, "unit": "C", "step": 5},
    "concentration": {"min": 0.05, "max": 1.0, "unit": "M", "step": 0.05}
  },
  "groups": [
    {
      "groupName": "catalyst",
      "selectionMode": "one-of",
        "description": "Catalyst for Suzuki reaction",
        "equivalents": {"min": 0.01, "max": 0.1, "unit": "eq", "step": 0.01},
      "members": [
        {"name": "Pd(PPh3)4", "equivalents": {"min": 0.005, "max": 0.05, "unit": "eq", "step": 0.005}},
        {"name": "Pd2(dba)3", "equivalents": {"min": 0.0025, "max": 0.025, "unit": "eq"}}
      ]
    },
    {
      "groupName": "solvent",
      "selectionMode": "one-of",
      "members": [
        {"name": "THF", "equivalents": {"min": 1, "max": 1, "unit": "vol-eq"}, "fixed": true},
        {"name": "MeCN", "equivalents": {"min": 1, "max": 1, "unit": "vol-eq"}, "fixed": true}
      ]
    }
  ]
  "chemicals": [
    {"chemicalName": "chemicalname",
    "descriptors": {"key1": "value1", "key2": "value2"},
    }
    ],
}
"""

def build_chemical_space_from_spec(spec: Dict[str, Any], lib: ChemicalLibrary) -> Campaign:
    campaign = Campaign(
        campaignName=spec.get("campaignName"),
        description=spec.get("description"),
        objective= spec.get("objective"),
        campaignClass=spec.get("campaignClass"),
        type=spec.get("type"),
        reference=spec.get("reference"),
        hasBatch=spec.get("hasBatch"),
        hasObjective=spec.get("hasObjective"),
        )

    # Global ranges
    ranges: Dict[str, Dict[str, Any]] = {}
    for rname, rvals in (spec.get("ranges") or {}).items():
        # Keep as-is; ensure shape {min,max,unit[,step]}
        rng = RangeSpec(
            min=float(rvals["min"]),
            max=float(rvals["max"]),
            unit=rvals.get("unit",""),
            step=float(rvals["step"]) if "step" in rvals else None
        ).to_json()
        ranges[rname] = rng
    campaign.hasRanges = ranges

    # Groups
    groups: List[Group] = []
    for g in (spec.get("groups") or []):
        eq = RangeSpec(
            min=float(g["equivalents"]["min"]),
            max=float(g["equivalents"]["max"]),
            unit=str(g["equivalents"].get("unit","eq")),
            step=float(g["equivalents"]["step"]) if "step" in g else None
        )
        grp = Group(
            groupName=g["groupName"],
            description=g.get("description",""),
            selectionMode=g.get("selectionMode","one-of"),
            members=[],
            equivalents=eq,
            fixed=g.get("fixed",True),
        )
        for m in g.get("members", []):
            rec = lib.get_by_name(m["name"])
            member = GroupMember(
                reference=rec.to_reference(),
                overrides=m.get("overrides", {})
            )
            grp.members.append(member)
        groups.append(grp)
    campaign.hasGroups = groups

    # Chemicals
    chemicals: List[ChemicalRecord] = []
    for chem in (spec.get("chemicals") or []):
        rec = lib.get_by_name(chem["chemicalName"])
        if "descriptors" in chem:
            rec.descriptors = chem["descriptors"]  # Individual chemicals include descriptors
        chemicals.append(rec)
    campaign.hasChemical = chemicals
    return campaign

# -----------------------------
# CLI
# -----------------------------


def write_json(doc: Dict[str, Any], path: Union[str, Path]) -> None:
    Path(path).write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="Build an HCI chemical-space JSON from a library and a space spec.")
    ap.add_argument("cmd", choices=["build","scaffold"], help="build: create JSON; scaffold: write an example spec")
    ap.add_argument("--library", help="Chemical library file (.json or .csv)")
    ap.add_argument("--spec", help="Chemical space spec Dict")
    ap.add_argument("--out", help="Output chemical space JSON")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--data-dir", default=str(DATA_DIR), help="Directory with data files for layout_parser")

    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    if args.cmd == "build":
        if not args.library or not args.spec or not args.out:
            raise SystemExit("For 'build', provide --library, --spec, --out")
        lib = load_library(args.library)
        spec = args.spec
        chemspace = build_chemical_space_from_spec(spec, lib)
        write_json(chemspace.to_json(), str(out_dir / args.out))
        print(f"Wrote chemical space to {args.out}")

if __name__ == "__main__":
    main()
