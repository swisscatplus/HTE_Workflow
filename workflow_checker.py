import argparse
from typing import Dict, List, Tuple

import openpyxl
import pandas as pd

import layout_parser as lp


def parse_calculator_excel(path: str) -> Tuple[List[str], List[str], List[str]]:
    """Return lists of solid reagents, liquid reagents and solvents from
    an HTE calculator output file."""
    xl = pd.ExcelFile(path)
    df = pd.read_excel(xl, "per_well", header=None)
    headers = [h for h in df.iloc[0, 1:] if isinstance(h, str)]
    unique = list(dict.fromkeys(headers))

    solids: List[str] = []
    liquids: List[str] = []
    solvents: List[str] = []

    for name in unique:
        low = name.lower()
        if "as stock solution" in low:
            # ignore amounts needed to prepare stock solutions
            continue
        if "(mg)" in low:
            solids.append(name)
            continue
        if " in " in name:
            liquids.append(name)
            continue
        if "(ul)" in low or "microliter" in low:
            if name.count(" ") == 1:
                solvents.append(name)
            else:
                liquids.append(name)
    return solids, liquids, solvents


def workflow_dispense_counts(path: str) -> Tuple[int, int, List[str]]:
    """Return number of solid and liquid dispense steps before the first
    orbital shaker as well as the list of workflow steps."""
    df, _ = lp.read_experiment_definition(path)
    steps = lp.extract_workflow_steps(df)

    orbital_idx = df.index[df["WF NODE"].str.contains("Orbital Shaker", na=False)]
    first_orbital = orbital_idx.min() if not orbital_idx.empty else len(df)

    pre = df.loc[: first_orbital - 1]
    solids = pre[(pre["TYPE"] == "Product") & pre["UNIT"].str.contains("gram|milligram", case=False, na=False)]
    liquids = pre[(pre["TYPE"] == "Product") & pre["UNIT"].str.contains("microliter", case=False, na=False)]

    return len(solids), len(liquids), steps


def per_well_to_experiments(
    per_well: pd.DataFrame, plate_layout: pd.DataFrame
) -> pd.DataFrame:
    """Return per-experiment reagent amounts from calculator output."""
    exp_index = [label for label in plate_layout.stack()]
    result = pd.DataFrame(index=exp_index)
    for reagent in per_well.columns.levels[0]:
        sub = per_well[reagent]
        values: Dict[str, float] = {}
        for row in sub.index:
            for col in sub.columns:
                exp = plate_layout.loc[row, col]
                values[exp] = float(sub.loc[row, col])
        result[reagent] = pd.Series(values)
    # ensure experiments are sorted numerically
    result.index = pd.CategoricalIndex(
        result.index, ordered=True, categories=sorted(result.index, key=lambda x: int(x.split()[1]))
    )
    result.sort_index(inplace=True)
    return result


NAME_MAP: Dict[str, str] = {
    "N_boc_pyrrole_2_boronic_acid_ester_mida": "N-Boc-Pyrrole-2-boronic acid MIDA ester (mg)",
    "K2CO3": "Potassium Carbonate (mg)",
    "KOAc": "Potassium Acetate (mg)",
    "K3PO4": "Tripotassium Phosphate (mg)",
    "Lutidine_in_MeOH_0_66M": "1,6-Lutidine in Methanol (uL)",
    "Lutidine_in_THF_0_66M": "1,6-Lutidine in Tetrahydrofuran (uL)",
    "Lutidine_in_MeCN_0_66M": "1,6-Lutidine in Acetonitrile (uL)",
    "Lutidine_in_dioxane_0_66M": "1,6-Lutidine in 1,4-Dioxane (uL)",
    "Lutidine_in_toluene_0_66M": "1,6-Lutidine in Toluene (uL)",
    "Et3N_in_MeOH_0_66M": "Triethylamine in Methanol (uL)",
    "Et3N_in_THF_0_66M": "Triethylamine in Tetrahydrofuran (uL)",
    "Et3N_in_dioxane_0_66M": "Triethylamine in 1,4-Dioxane (uL)",
    "Et3N_in_toluene_0_66M": "Triethylamine in Toluene (uL)",
    "TBAH_in_MeOH_0_66M": "Tetrabutylammonium Hydroxide in Methanol (uL)",
    "TBAH_in_THF_0_66M": "Tetrabutylammonium Hydroxide in Tetrahydrofuran (uL)",
    "TBAH_in_MeCN_0_66M": "Tetrabutylammonium Hydroxide in Acetonitrile (uL)",
    "TBAH_in_dioxane_0_66M": "Tetrabutylammonium Hydroxide in 1,4-Dioxane (uL)",
    "TBAH_in_toluene_0_66M": "Tetrabutylammonium Hydroxide in Toluene (uL)",
    "18C6_in_MeOH_0_033M": "18-Crown-6 in Methanol (uL)",
    "18C6_in_THF_0_033M": "18-Crown-6 in Tetrahydrofuran (uL)",
    "18C6_in_MeCN_0_033M": "18-Crown-6 in Acetonitrile (uL)",
    "18C6_in_dioxane_0_033M": "18-Crown-6 in 1,4-Dioxane (uL)",
    "18C6_in_toluene_0_033M": "18-Crown-6 in Toluene (uL)",
    "MeOH_pure": "Methanol (uL)",
    "THF_pure": "Tetrahydrofuran (uL)",
    "MeCN_pure": "Acetonitrile (uL)",
    "Dioxane_pure": "1,4-Dioxane (uL)",
    "Toluene_pure": "Toluene (uL)",
}


def fill_workflow(calculator: str, workflow: str) -> None:
    """Fill an empty workflow file with amounts from the calculator export.

    The workflow file is modified in place."""
    wf_df, mapping = lp.read_experiment_definition(workflow)
    num_exp = len(mapping)

    per = pd.read_excel(calculator, "per_well", header=[0, 1], index_col=0)
    plate_layout, _ = lp.map_experiments_to_wells(num_exp)
    per_exp = per_well_to_experiments(per, plate_layout)

    wb = openpyxl.load_workbook(workflow)
    ws = wb["Experiment Definition"]

    rev_map = {v: k + 1 for k, v in mapping.items()}

    orbital_idx = wf_df.index[wf_df["WF NODE"].str.contains("Orbital Shaker", na=False)]
    first_orbital = orbital_idx.min() if not orbital_idx.empty else len(wf_df)

    header_offset = 8  # number of header rows before pandas data

    for idx, row in wf_df.loc[: first_orbital - 1].iterrows():
        label = row.get("LABEL")
        if not isinstance(label, str):
            continue
        calc_name = NAME_MAP.get(label)
        if not calc_name:
            continue
        if calc_name not in per_exp.columns:
            continue
        excel_row = idx + header_offset + 1
        for exp_label, col_idx in rev_map.items():
            val = per_exp.at[exp_label, calc_name]
            ws.cell(row=excel_row, column=col_idx, value=val)

    wb.save(workflow)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check workflow against calculator output")
    parser.add_argument("calculator", help="Excel file from hte_calculator")
    parser.add_argument("workflow", help="Workflow Excel file (empty)")
    parser.add_argument("--visualize", action="store_true", help="Run layout_parser to generate images")
    parser.add_argument(
        "--fill",
        action="store_true",
        help="Write dispense amounts directly into the workflow file",
    )
    args = parser.parse_args()

    calc_solids, calc_liquids, calc_solvents = parse_calculator_excel(args.calculator)
    wf_solids, wf_liquids, steps = workflow_dispense_counts(args.workflow)

    print(f"Calculator reagents: {len(calc_solids)} solids, {len(calc_liquids)} liquids, {len(calc_solvents)} solvents")
    print(f"Workflow before first orbital shaker has {wf_solids} solid dispenses and {wf_liquids} liquid dispenses")
    if len(calc_solids) != wf_solids:
        print("Warning: mismatch in number of solid dispenses")
    if len(calc_liquids) + len(calc_solvents) != wf_liquids:
        print("Warning: mismatch in number of liquid/solvent dispenses")

    if args.fill:
        fill_workflow(args.calculator, args.workflow)
        print(f"Workflow {args.workflow} updated")

    if args.visualize:
        lp.main()


if __name__ == "__main__":
    main()
