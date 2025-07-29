import argparse
import string
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import pandas as pd
import requests

PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{}/property/MolecularWeight/TXT"
)


def fetch_molar_mass(inchikey: str) -> float:
    """Return molecular weight from PubChem given an InChIKey."""
    response = requests.get(PUBCHEM_URL.format(inchikey))
    response.raise_for_status()
    return float(response.text.strip())


@dataclass
class Reagent:
    name: str
    inchikey: str
    rtype: str  # 'solid' or 'liquid'
    equivalents: float
    is_limiting: bool = False
    density: Optional[float] = None  # g/mL
    concentration: Optional[float] = None  # mol/L
    locations: pd.DataFrame = field(default_factory=pd.DataFrame)
    molar_mass: Optional[float] = None

    def ensure_molar_mass(self) -> None:
        if self.molar_mass is None:
            self.molar_mass = fetch_molar_mass(self.inchikey)


@dataclass
class Solvent:
    name: str
    inchikey: str
    locations: pd.DataFrame = field(default_factory=pd.DataFrame)


class Plate:
    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        row_labels = list(string.ascii_uppercase[:rows])
        col_labels = list(range(1, cols + 1))
        self.template = pd.DataFrame(False, index=row_labels, columns=col_labels)

    def parse_location(self, prompt: str) -> pd.DataFrame:
        print(f"\nSpecify location for {prompt}:")
        print("  [a] all wells")
        print("  [c] specific columns")
        print("  [r] specific rows")
        print("  [w] individual wells (e.g., A1,B2)")
        choice = input("Choice: ").strip().lower()
        mat = self.template.copy()
        if choice == 'a':
            mat.loc[:, :] = True
        elif choice == 'c':
            cols = input("Enter columns separated by comma: ").split(',')
            cols = [int(c.strip()) for c in cols if c.strip()]
            mat.loc[:, cols] = True
        elif choice == 'r':
            rows = input("Enter rows separated by comma (A,B,...): ").split(',')
            rows = [r.strip().upper() for r in rows if r.strip()]
            mat.loc[rows, :] = True
        elif choice == 'w':
            wells = input("Enter wells separated by comma (e.g., A1,B2): ").split(',')
            for w in wells:
                w = w.strip()
                if not w:
                    continue
                row = w[0].upper()
                col = int(w[1:])
                mat.loc[row, col] = True
        else:
            print("Invalid choice; none selected")
        return mat

    def input_matrix(self, label: str) -> pd.DataFrame:
        print(f"\nInput method for {label}:")
        print("  [c] constant across all wells")
        print("  [r] per row")
        print("  [C] per column")
        print("  [w] per well")
        choice = input("Choice: ").strip().lower()
        df = pd.DataFrame(0.0, index=self.template.index, columns=self.template.columns)
        if choice == 'c':
            val = float(input(f"Enter {label}: "))
            df.loc[:, :] = val
        elif choice == 'r':
            for row in self.template.index:
                val = float(input(f"{label} for row {row}: "))
                df.loc[row, :] = val
        elif choice == 'c':
            pass
        elif choice == 'C':
            for col in self.template.columns:
                val = float(input(f"{label} for column {col}: "))
                df.loc[:, col] = val
        elif choice == 'w':
            for row in self.template.index:
                for col in self.template.columns:
                    val = float(input(f"{label} for well {row}{col}: "))
                    df.loc[row, col] = val
        else:
            print("Invalid choice; defaulting to 0")
        return df


def main() -> None:
    parser = argparse.ArgumentParser(description="HTE Dispense Calculator")
    parser.add_argument("--rows", type=int, default=8, help="number of plate rows")
    parser.add_argument("--cols", type=int, default=12, help="number of plate columns")
    parser.add_argument("--output", default="hte_output.xlsx", help="Excel output file")
    args = parser.parse_args()

    plate = Plate(args.rows, args.cols)

    reagents: List[Reagent] = []
    solvents: List[Solvent] = []

    limiting_set = False
    while True:
        name = input("Reagent name (blank to finish): ").strip()
        if not name:
            break
        inchikey = input("InChIKey: ").strip()
        rtype = input("Type [solid/liquid/solvent]: ").strip().lower()
        if rtype == 'solvent':
            solv = Solvent(name=name, inchikey=inchikey)
            solv.locations = plate.parse_location(f"solvent {name}")
            solvents.append(solv)
            continue
        eqv = float(input("Equivalents: "))
        is_limiting = False
        if not limiting_set:
            ans = input("Is this the limiting reagent? [y/N]: ").strip().lower()
            is_limiting = ans == 'y'
            limiting_set = is_limiting
        density = None
        concentration = None
        if rtype == 'liquid':
            d = input("Density (g/mL, blank if not known): ").strip()
            if d:
                density = float(d)
            c = input("Concentration (mol/L, blank if not known): ").strip()
            if c:
                concentration = float(c)
        reagent = Reagent(name=name, inchikey=inchikey, rtype=rtype,
                          equivalents=eqv, is_limiting=is_limiting,
                          density=density, concentration=concentration)
        reagent.locations = plate.parse_location(f"reagent {name}")
        reagents.append(reagent)

    if not any(r.is_limiting for r in reagents):
        raise RuntimeError("No limiting reagent specified")

    final_volume = plate.input_matrix("final volume per well (uL)")
    conc_limiting = plate.input_matrix("desired concentration of limiting reagent (mol/L)")

    results: Dict[str, pd.DataFrame] = {}
    totals: Dict[str, float] = {}

    limiting = next(r for r in reagents if r.is_limiting)
    limiting.ensure_molar_mass()
    moles_limiting = conc_limiting * (final_volume / 1_000_000)

    for reagent in reagents:
        reagent.ensure_molar_mass()
        moles = moles_limiting * reagent.equivalents * reagent.locations
        if reagent.rtype == 'solid':
            mass_g = moles * reagent.molar_mass
            results[f"{reagent.name} (mg)"] = mass_g * 1000
            totals[f"{reagent.name} (mg)"] = (mass_g * 1000).sum().sum()
        else:
            mass_g = moles * reagent.molar_mass
            if reagent.concentration:
                volume_l = moles / reagent.concentration
            elif reagent.density:
                volume_l = mass_g / reagent.density / 1000
            else:
                raise ValueError(f"Liquid reagent {reagent.name} requires density or concentration")
            results[f"{reagent.name} (uL)"] = volume_l * 1_000_000
            totals[f"{reagent.name} (uL)"] = (volume_l * 1_000_000).sum().sum()

    volume_used = pd.DataFrame(0.0, index=plate.template.index, columns=plate.template.columns)
    for key, df in results.items():
        if key.endswith('(uL)'):
            volume_used += df
    solvent_vol = final_volume - volume_used
    if solvents:
        each = solvent_vol / len(solvents)
        for solv in solvents:
            vol = each * solv.locations
            results[f"{solv.name} (uL)"] = vol
            totals[f"{solv.name} (uL)"] = vol.sum().sum()
        if (solvent_vol < 0).any().any():
            print("Warning: total reagent volumes exceed final volume in some wells")
        union = sum(s.locations for s in solvents)
        if (union == 0).any().any():
            print("Warning: Some wells have no solvent added.")
    else:
        if (solvent_vol < 0).any().any():
            print("Warning: total reagent volumes exceed final volume in some wells")

    df = pd.concat(results, axis=1)
    df.index.name = 'Row'

    totals_series = pd.Series(totals, name='Total')

    with pd.ExcelWriter(args.output) as writer:
        df.to_excel(writer, sheet_name='per_well')
        totals_series.to_frame().to_excel(writer, sheet_name='totals')
    print(f"Results written to {args.output}")


if __name__ == '__main__':
    main()
