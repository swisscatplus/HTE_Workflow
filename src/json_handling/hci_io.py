"""
hci_io.py
----------
Bridges workflow.py -> HCI JSON -> hte_calculator.py.

- Loads fixed chemical metadata from a JSON/CSV "library".
- Converts Bayesian engine suggestions into HCI JSON entries.
- Writes an HCI JSON that matches the provided structure.
- Parses HCI JSON into a compact dict for hte_calculator.py.

No third-party dependencies.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Iterable, Union
import csv
import json
from pathlib import Path


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
class ChemicalRecord:
    """Fixed info for a chemical, typically coming from a library file."""
    chemicalID: str
    chemicalName: str
    CASNumber: str
    molecularMass: Quantity
    smiles: str
    swissCatNumber: str
    Inchi: str
    molecularFormula: str
    density: Optional[Quantity] = None
    # You may include arbitrary extra fields in `extras` (e.g., safety codes)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_hci_payload(
        self,
        stoechiometry: Optional[Quantity] = None,
        keywords: Optional[str] = None,
        override_density: Optional[Quantity] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a dict matching the HCI 'hasChemical' entry, allowing minimal overrides."""
        payload = {
            "chemicalID": self.chemicalID,
            "chemicalName": self.chemicalName,
            "CASNumber": self.CASNumber,
            "molecularMass": self.molecularMass.to_json(),
            "smiles": self.smiles,
            "swissCatNumber": self.swissCatNumber,
            "keywords": keywords or "optional only in HCI file",
            "Inchi": self.Inchi,
            "molecularFormula": self.molecularFormula,
            # In the example HCI, stoechiometry is always present
            "stoechiometry": (stoechiometry or Quantity(1, "")).to_json(),
            # In the example HCI, density is present for many entries
            "density": (override_density or self.density or Quantity(1.0, "g/mL")).to_json(),
        }
        if overrides:
            payload.update(overrides)
        return payload


# -----------------------------
# Library loader
# -----------------------------

class ChemicalLibrary:
    """
    Lightweight loader for a chemicals library in JSON or CSV.
    Expected columns/keys (case-sensitive JSON keys; CSV headers can be flexible):
      chemicalID, chemicalName, CASNumber, molecularMass_value, molecularMass_unit,
      smiles, swissCatNumber, Inchi, molecularFormula, density_value, density_unit
    Extra columns are kept in `extras`.
    """

    def __init__(self, records: Dict[str, ChemicalRecord]):
        # index by both name (lower) and ID for convenience
        self._by_id: Dict[str, ChemicalRecord] = {}
        self._by_name: Dict[str, ChemicalRecord] = {}
        for rec in records.values():
            self._by_id[rec.chemicalID] = rec
            self._by_name[rec.chemicalName.lower()] = rec

    @staticmethod
    def _quantity_from_row(prefix: str, row: Dict[str, str]) -> Optional[Quantity]:
        v_key = f"{prefix}_value"
        u_key = f"{prefix}_unit"
        if v_key in row and row[v_key] not in (None, "", "NA", "na"):
            try:
                value = float(row[v_key])
            except ValueError:
                return None
            unit = row.get(u_key, "")
            return Quantity(value=value, unit=unit)
        return None

    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> "ChemicalLibrary":
        path = Path(path)
        records: Dict[str, ChemicalRecord] = {}
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mm = cls._quantity_from_row("molecularMass", row) or Quantity(0.0, "g/mol")
                dens = cls._quantity_from_row("density", row)
                # Extract known fields; keep the rest as extras
                known = {
                    "chemicalID", "chemicalName", "CASNumber",
                    "smiles", "swissCatNumber", "Inchi", "molecularFormula",
                    "molecularMass_value", "molecularMass_unit",
                    "density_value", "density_unit",
                }
                extras = {k: v for k, v in row.items() if k not in known and v not in (None, "")}
                rec = ChemicalRecord(
                    chemicalID=row["chemicalID"],
                    chemicalName=row["chemicalName"],
                    CASNumber=row.get("CASNumber", ""),
                    molecularMass=mm,
                    smiles=row.get("smiles", ""),
                    swissCatNumber=row.get("swissCatNumber", ""),
                    Inchi=row.get("Inchi", ""),
                    molecularFormula=row.get("molecularFormula", ""),
                    density=dens,
                    extras=extras,
                )
                records[rec.chemicalID] = rec
        return cls(records)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ChemicalLibrary":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Accept either list or mapping keyed by ID
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = list(data.values())
        else:
            raise ValueError("Unsupported JSON structure for chemical library")
        records: Dict[str, ChemicalRecord] = {}
        for item in items:
            mm_raw = item.get("molecularMass", {})
            mm = Quantity(float(mm_raw.get("value", 0.0)), mm_raw.get("unit", "g/mol"))
            dens_raw = item.get("density")
            dens = None
            if isinstance(dens_raw, dict):
                dens = Quantity(float(dens_raw.get("value", 0.0)), dens_raw.get("unit", ""))
            rec = ChemicalRecord(
                chemicalID=str(item.get("chemicalID", "")),
                chemicalName=item.get("chemicalName", ""),
                CASNumber=item.get("CASNumber", ""),
                molecularMass=mm,
                smiles=item.get("smiles", ""),
                swissCatNumber=item.get("swissCatNumber", ""),
                Inchi=item.get("Inchi", ""),
                molecularFormula=item.get("molecularFormula", ""),
                density=dens,
                extras={k: v for k, v in item.items() if k not in {
                    "chemicalID", "chemicalName", "CASNumber", "molecularMass",
                    "smiles", "swissCatNumber", "Inchi", "molecularFormula", "density"
                }},
            )
            records[rec.chemicalID] = rec
        return cls(records)

    def get(self, key: str) -> ChemicalRecord:
        """Lookup by chemicalID or case-insensitive chemicalName."""
        if key in self._by_id:
            return self._by_id[key]
        k = key.lower()
        if k in self._by_name:
            return self._by_name[k]
        raise KeyError(f"Chemical '{key}' not found in library.")

    def try_get(self, key: str) -> Optional[ChemicalRecord]:
        try:
            return self.get(key)
        except KeyError:
            return None


# -----------------------------
# HCI JSON builder/parsers
# -----------------------------

def build_hci_document(
    campaign_name: str,
    description: str,
    objective: str,
    campaign_class: str,
    type_: str,
    reference: str,
    batch: Dict[str, Any],
    objective_block: Dict[str, Any],
    chemicals_payload: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build an HCI JSON structure matching the provided example."""
    return {
        "hasCampaign": {
            "campaignName": campaign_name,
            "description": description,
            "objective": objective,
            "campaignClass": campaign_class,
            "type": type_,
            "reference": reference,
            "hasBatch": batch,
            "hasObjective": objective_block,
            "hasChemical": chemicals_payload,
        }
    }


def write_hci_json(doc: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=4, ensure_ascii=False)


def read_hci_json(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Thin adapter: workflow.py -> HCI -> hte_calculator.py
# -----------------------------

def bayes_suggestions_to_hci_chemicals(
    suggestions: Iterable[Dict[str, Any]],
    library: ChemicalLibrary,
    default_keywords: str = "optional only in HCI file",
) -> List[Dict[str, Any]]:
    """
    Convert Bayesian engine 'suggestions' into HCI 'hasChemical' entries.
    Each suggestion item may look like:
      {
        "chemical": "Methanol",          # can be name or ID
        "stoich": 1.0,                   # optional; default 1
        "stoich_unit": "",               # optional
        "density_override": 0.79,        # optional; value only
        "density_unit": "g/mL",          # optional
        "keywords": "solvent",           # optional
        "overrides": { ... }             # any additional fields to inject
      }
    """
    payloads: List[Dict[str, Any]] = []
    for s in suggestions:
        key = str(s.get("chemical", "")).strip()
        rec = library.get(key)
        st = Quantity(float(s.get("stoich", 1.0)), s.get("stoich_unit", ""))
        d_override = None
        if "density_override" in s:
            d_override = Quantity(float(s["density_override"]), s.get("density_unit", "g/mL"))
        payloads.append(
            rec.to_hci_payload(
                stoechiometry=st,
                keywords=s.get("keywords", default_keywords),
                override_density=d_override,
                overrides=s.get("overrides"),
            )
        )
    return payloads


def hci_to_hte_calculator_inputs(hci_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Map HCI JSON into a compact dict for hte_calculator.py."""
    c = hci_doc.get("hasCampaign", {})
    chemicals = []
    for ch in c.get("hasChemical", []) or []:
        chemicals.append({
            "chemicalID": ch.get("chemicalID"),
            "chemicalName": ch.get("chemicalName"),
            "CASNumber": ch.get("CASNumber"),
            "molecularMass": ch.get("molecularMass", {}).get("value"),
            "molecularMassUnit": ch.get("molecularMass", {}).get("unit"),
            "smiles": ch.get("smiles"),
            "Inchi": ch.get("Inchi"),
            "molecularFormula": ch.get("molecularFormula"),
            "density": (ch.get("density") or {}).get("value"),
            "densityUnit": (ch.get("density") or {}).get("unit"),
            "stoich": (ch.get("stoechiometry") or {}).get("value"),
            "stoichUnit": (ch.get("stoechiometry") or {}).get("unit"),
            "swissCatNumber": ch.get("swissCatNumber"),
            "keywords": ch.get("keywords"),
        })
    return {
        "campaign": {
            "campaignName": c.get("campaignName"),
            "description": c.get("description"),
            "objective": c.get("objective"),
            "campaignClass": c.get("campaignClass"),
            "type": c.get("type"),
            "reference": c.get("reference"),
        },
        "batch": c.get("hasBatch", {}),
        "objective": c.get("hasObjective", {}),
        "chemicals": chemicals,
    }


# -----------------------------
# Minimal CLI (optional)
# -----------------------------

def _load_library(path: Union[str, Path]) -> ChemicalLibrary:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return ChemicalLibrary.from_csv(p)
    return ChemicalLibrary.from_json(p)


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build/parse HCI JSON documents to bridge workflow.py and hte_calculator.py"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build an HCI JSON file from a library and a suggestions JSON")
    p_build.add_argument("--library", required=True, help="Path to chemical library (.json or .csv)")
    p_build.add_argument("--suggestions", required=True, help="JSON file with Bayesian suggestions (list of items)")
    p_build.add_argument("--out", required=True, help="Output HCI JSON file")
    p_build.add_argument("--campaign-name", default="Campaign")
    p_build.add_argument("--description", default="DESCRIPTION MISSING")
    p_build.add_argument("--objective", default="")
    p_build.add_argument("--campaign-class", default="Standard Research")
    p_build.add_argument("--type", dest="type_", default="optimization")
    p_build.add_argument("--reference", default="")

    p_parse = sub.add_parser("parse", help="Parse an HCI JSON file to a compact calculator-friendly JSON")
    p_parse.add_argument("--hci", required=True, help="Path to HCI JSON")
    p_parse.add_argument("--out", required=True, help="Where to write the compact JSON")

    args = parser.parse_args(argv)

    if args.cmd == "build":
        lib = _load_library(args.library)
        with open(args.suggestions, "r", encoding="utf-8") as f:
            suggestions = json.load(f)
        chemicals_payload = bayes_suggestions_to_hci_chemicals(suggestions, lib)
        # batch & objective blocks can be customized upstream; provide minimal defaults
        batch = {"batchID": "0", "batchName": "YYYYMMDD", "reactionType": "", "reactionName": "", "optimizationType": "", "link": ""}
        objective_block = {"criteria": "", "condition": "", "description": "", "objectiveName": ""}
        doc = build_hci_document(
            campaign_name=args.campaign_name,
            description=args.description,
            objective=args.objective,
            campaign_class=args.campaign_class,
            type_=args.type_,
            reference=args.reference,
            batch=batch,
            objective_block=objective_block,
            chemicals_payload=chemicals_payload,
        )
        write_hci_json(doc, args.out)

    elif args.cmd == "parse":
        hci = read_hci_json(args.hci)
        compact = hci_to_hte_calculator_inputs(hci)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(compact, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
