import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
from visualization import generate_pie_plots, generate_heatmaps


def load_calibration_data(folder: str) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Return calibration info mapping compound to signal and fit data.

    The folder is searched for files named ``calibration_*.xlsx``. For each
    calibration file the ``compound_name`` is read from sheet ``data`` and the
    ``fit`` sheet is used to extract slope and intercept. The user is prompted
    whether the calibration should be used.
    """

    calibrations: Dict[str, Dict[str, float]] = {}
    signal_names: List[str] = []
    for fname in os.listdir(folder):
        if not fname.lower().startswith("calibration") or not fname.lower().endswith(".xlsx"):
            continue
        path = os.path.join(folder, fname)
        try:
            data = pd.read_excel(path, sheet_name="data")
            compound = str(data["compound_name"].iloc[0])
        except Exception:
            continue
        resp = input(f"Use calibration for {compound}? [y/N]: ").strip().lower()
        if resp != "y":
            continue
        signal = str(data["signal"].iloc[0])
        fit = pd.read_excel(path, sheet_name="fit")
        slope = float(fit.get("slope", [1.0])[0])
        intercept = float(fit.get("intercept", [0.0])[0])
        is_conc = float(fit.get("internal_standard_concentration", [1.0])[0])
        calibrations[compound] = {
            "signal": signal,
            "slope": slope,
            "intercept": intercept,
            "internal_standard_concentration": is_conc,
        }
        signal_names.append(signal)
    return calibrations, signal_names

def find_analysis_files(folder: str) -> List[str]:
    """Return list of csv analysis files ignoring calibration and blanc."""
    files: List[str] = []
    for fname in os.listdir(folder):
        low = fname.lower()
        if low.endswith(".csv") and "calibration" not in low and "blanc" not in low:
            files.append(os.path.join(folder, fname))
    return sorted(files)


def parse_experiment_number(path: str) -> int:
    """Return experiment number based on plate position in filename."""
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    parts = name.split("_")
    analysis_pos = parts[-1]
    plate_id, plate_pos = analysis_pos.split("-")  # Implement multiple plate IDs if needed
    match = re.fullmatch(r"([A-Za-z])(\d+)", plate_pos)
    if not match:
        raise ValueError(f"Cannot parse plate position from {base}")
    row = ord(match.group(1).upper()) - ord("A")
    col = int(match.group(2)) - 1
    return row * 9 + col + 1


def process_file(
    path: str,
    signal_names: List[str],
    start_time: Optional[float],
    name_map: Dict[str, str],
    calibrations: Dict[str, Dict[str, float]],
    internal_std: Optional[str],
) -> Tuple[Dict[str, float], Dict[str, str], Optional[str]]:
    """Extract areas per compound from a single analysis file."""
    df = pd.read_csv(path)
    sig_col = next((c for c in df.columns if "signal description" in c.lower()), None)
    name_col = next((c for c in df.columns if "name" in c.lower()), None)
    area_col = next((c for c in df.columns if "area" in c.lower()), None)
    rt_col = next((c for c in df.columns if c.lower() in ("rt", "retention time", "retention_time", "rt (min)")), None)
    if not sig_col or not name_col or not area_col:
        return {}, name_map, internal_std
    filtered = df[df[sig_col].astype(str).isin(signal_names)]
    if start_time is not None and rt_col:
        filtered = filtered[filtered[rt_col] >= start_time]
    areas: Dict[str, float] = {}
    for _, row in filtered.iterrows():
        pname = str(row[name_col]).strip()
        if not pname or pname.lower() == "nan":
            comp = str(row[rt_col]).strip()
            area_val = row[area_col]
            areas[comp] = float(area_val) if not pd.isna(area_val) else 0.0
            continue
        if pname not in name_map:
            print(f"Calibration compounds: {', '.join(calibrations.keys())}")
            resp = input(
                f"Classify '{pname}': [i]nternal standard/[c]alibration/[n]ew? "
            ).strip().lower()
            if resp == "i":
                if not internal_std:
                    internal_std = "Internal Standard"
                name_map[pname] = internal_std
            elif resp == "c":
                choice = input(
                    f"Enter calibration compound name ({', '.join(calibrations.keys())}): "
                ).strip()
                name_map[pname] = choice if choice else pname
            else:
                name_map[pname] = input(f"Enter name for '{pname}': ").strip() or pname
        comp = name_map[pname]
        area_val = row[area_col]
        if pd.isna(area_val):
            continue
        areas[comp] = float(area_val)
    return areas, name_map, internal_std


def load_actual_layout(path: str) -> pd.DataFrame:
    """Parse ``actual.xlsx`` produced by ``hte_calculator`` and return layout."""
    per = pd.read_excel(path, sheet_name="per_well", header=[0, 1])
    rows = per.iloc[1:, 0].astype(str).tolist()
    first_reagent = per.columns[1][0]
    cols = [int(c[1]) for c in per.columns if c[0] == first_reagent]
    layout = pd.DataFrame(index=rows, columns=cols)
    exp = 1
    for r in rows:
        for c in cols:
            layout.loc[r, c] = exp
            exp += 1
    return layout


def build_matrix(
    data: Dict[str, Dict[int, float]],
    layout: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Arrange compound data into plate layout or simple experiment order."""
    if layout is not None:
        exp_to_rc: Dict[int, Tuple[str, int]] = {}
        for r in layout.index:
            for c in layout.columns:
                exp = int(layout.loc[r, c])
                exp_to_rc[exp] = (r, c)
        per_comp: Dict[str, pd.DataFrame] = {}
        for comp, exp_map in data.items():
            mat = pd.DataFrame(float("nan"), index=layout.index, columns=layout.columns)
            for exp, val in exp_map.items():
                if exp in exp_to_rc:
                    r, c = exp_to_rc[exp]
                    mat.loc[r, c] = val
            per_comp[comp] = mat
        return pd.concat(per_comp, axis=1)
    # fallback: simple experiment number ordering
    all_exps = sorted({e for mapping in data.values() for e in mapping})
    df = pd.DataFrame(index=all_exps)
    for comp, exp_map in data.items():
        df[comp] = [exp_map.get(exp, float("nan")) for exp in all_exps]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze HTE data files")
    parser.add_argument("folder", help="Folder with analysis CSV files")
    parser.add_argument("--output", default="analysis_output.xlsx", help="Output Excel file")
    parser.add_argument("--layout", help="Path to actual.xlsx for layout ordering", default=None)
    args = parser.parse_args()

    calibrations, signal_names = load_calibration_data(args.folder)
    if not signal_names:
        print("No calibrations selected; aborting")
        return
    files = find_analysis_files(args.folder)
    if not files:
        print("No analysis files found")
        return
    start = input("Analysis start time in min (blank for none): ").strip()
    start_time = float(start) if start else None
    name_map: Dict[str, str] = {}
    internal_std: Optional[str] = None
    area_data: Dict[str, Dict[int, float]] = {}
    ratio_total: Dict[str, Dict[int, float]] = {}
    ratio_is: Dict[str, Dict[int, float]] = {}
    for path in files:
        areas, name_map, internal_std = process_file(
            path, signal_names, start_time, name_map, calibrations, internal_std
        )
        if not areas:
            continue
        exp = parse_experiment_number(path)
        total_non_is = sum(v for k, v in areas.items() if k != internal_std)
        for comp, area in areas.items():
            area_data.setdefault(comp, {})[exp] = area
            if comp != internal_std and total_non_is:
                ratio_total.setdefault(comp, {})[exp] = area / total_non_is
            if comp in calibrations and internal_std and internal_std in areas and areas[internal_std]:
                ratio_is.setdefault(comp, {})[exp] = area / areas[internal_std]
    layout_path = args.layout or input("Path to actual.xlsx for ordering (blank for none): ").strip()
    layout_df: Optional[pd.DataFrame] = None
    if layout_path:
        try:
            layout_df = load_actual_layout(layout_path)
        except Exception as exc:
            print(f"Could not parse layout: {exc}; using experiment number order")
            layout_df = None
    area_df = build_matrix(area_data, layout_df)
    total_df = build_matrix(ratio_total, layout_df)
    is_df = build_matrix(ratio_is, layout_df)

    scale_in = input("Reaction scale in mmol: ").strip()
    scale = float(scale_in) if scale_in else 1.0
    is_amount_in = input("Internal standard added in mmol: ").strip()
    is_amount = float(is_amount_in) if is_amount_in else 1.0

    yield_data: Dict[str, Dict[int, float]] = {}
    for comp, exp_map in ratio_is.items():
        calib = calibrations.get(comp)
        if not calib:
            continue
        slope = calib.get("slope", 1.0)
        intercept = calib.get("intercept", 0.0)
        is_conc = calib.get("internal_standard_concentration", 1.0)
        for exp, ratio_val in exp_map.items():
            conc = (ratio_val - intercept) / slope
            n_prod = conc * (is_amount / is_conc)
            yield_percent = n_prod / scale * 100.0
            yield_data.setdefault(comp, {})[exp] = yield_percent

    yield_df = build_matrix(yield_data, layout_df)

    with pd.ExcelWriter(args.output) as writer:
        area_df.to_excel(writer, sheet_name="area")
        total_df.to_excel(writer, sheet_name="area_fraction")
        is_df.to_excel(writer, sheet_name="area_internal")
        yield_df.to_excel(writer, sheet_name="yield")
        pd.DataFrame(
            {
                "reaction_scale_mmol": [scale],
                "internal_standard_mmol": [is_amount],
            }
        ).to_excel(writer, sheet_name="info", index=False)
    print(f"Saved analysis to {args.output}")

    generate_pie_plots(area_data, yield_data, calibrations, internal_std, layout_df)
    generate_heatmaps(yield_df)


if __name__ == "__main__":
    main()
