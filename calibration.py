import argparse
import os
import re
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.stats import linregress
except Exception:  # scipy might not be installed
    linregress = None


def parse_filename(path: str) -> Tuple[str, float]:
    """Extract product name and concentration (mM) from calibration file name.

    Filenames follow ``date_reaction_calibration_product_a_bM.csv`` where the
    concentration ``a_bM`` represents ``a.b`` mM.
    """

    base = os.path.basename(path)
    if not base.endswith(".csv"):
        raise ValueError(f"{base} is not a CSV file")
    match = re.fullmatch(
        r".+_calibration_(.+)_([0-9]+)_([0-9]+)M\.csv",
        base,
        re.IGNORECASE,
    )
    if not match:
        raise ValueError(f"Filename {base} does not match expected pattern")
    product, a, b = match.groups()
    concentration = float(f"{a}.{b}")
    return product, concentration


def select_signal(df: pd.DataFrame, signal) -> Tuple[pd.DataFrame, str]:
    """Prompt user to choose a signal description and filter rows."""
    sig_cols = [c for c in df.columns if "signal" in c.lower()]
    if not sig_cols:
        raise ValueError("No signal description column found in file")
    sig_col = sig_cols[0]
    if signal == None:
        print("Available signals:")
        for s in df[sig_col].dropna().unique():
            print(f"  {s}")
        user_sig = input("Enter signal description to use: ").strip()
    else:
        user_sig = signal
    filtered = df[df[sig_col].astype(str).str.contains(user_sig, regex=False)]
    if filtered.empty:
        raise ValueError(f"No rows matched signal description '{user_sig}'")
    return filtered, user_sig


def handle_rows(
    df: pd.DataFrame,
    product: str,
    internal_std_name: Optional[str],
    internal_std_conc: Optional[float],
    actual_product_names: dict,
) -> Tuple[dict, Optional[str], Optional[float]]:
    """Process filtered rows, identify product and internal standard, compute ratios."""
    name_cols = [c for c in df.columns if "name" in c.lower()]
    area_cols = [c for c in df.columns if "area" in c.lower()]
    rt_cols = [c for c in df.columns if c.lower() in ("rt", "retention time", "retention_time", "rt (min)")]
    if not name_cols or not area_cols:
        raise ValueError("Required columns 'name' and 'area' not found")
    name_col = name_cols[0]
    area_col = area_cols[0]
    rt_col = rt_cols[0] if rt_cols else None

    comp_name = None
    product_area = None
    is_area = None
    other_area = 0.0

    for _, row in df.iterrows():
        name = str(row.get(name_col, "")).strip()
        if not name or name.lower() == "nan":
            area_val = row.get(area_col)
            if pd.isna(area_val):
                continue
            area_val = float(area_val)
            other_area += area_val
            continue
        if rt_col and pd.isna(row.get(rt_col)):
            continue
        area_val = row.get(area_col)
        if pd.isna(area_val):
            continue
        area_val = float(area_val)

        # Detect internal standard already defined
        if internal_std_name and name == internal_std_name:
            if is_area is not None:
                raise ValueError("Multiple internal standard entries found")
            is_area = area_val
            continue

        is_is = False
        if internal_std_name is None:
            resp = input(f"Is '{name}' the internal standard? [y/N]: ").strip().lower()
            if resp == "y":
                is_is = True
        if is_is:
            internal_std_name = name
            internal_std_conc = float(input("Enter known concentration of internal standard: "))
            is_area = area_val
            continue

        # Non internal standard entries
        if name in actual_product_names:
            comp_name = actual_product_names[name]
        else:
            comp_name = input(f"Enter compound name for '{name}': ").strip() or product
            actual_product_names[name] = comp_name
        if product_area is not None:
            raise ValueError("Multiple product entries found")
        product_area = area_val

    if product_area is None or is_area is None:
        raise ValueError("Internal standard or product peak missing")

    ratio = product_area / is_area
    combined_ratio = (product_area + is_area) / (other_area + product_area + is_area) if other_area else float(1.0)
    return {
        "product": product,
        "compound_name": comp_name,
        "area_product": product_area,
        "area_internal_standard": is_area,
        "ratio": ratio,
        "combined_ratio": combined_ratio,
        "internal_standard_concentration": internal_std_conc,
    }, internal_std_name, internal_std_conc


def analyze(data: pd.DataFrame, folder) -> None:
    """Group data by product, plot calibration curves, and save to Excel."""
    for product, group in data.groupby("product"):
        signals = group["signal"].unique()
        comp_name = group["compound_name"].unique()
        if len(comp_name) != 1:
            raise ValueError(f"Inconsistent compound names for product '{product}'")
        if len(signals) != 1:
            raise ValueError(f"Inconsistent signal descriptions for product '{product}'")
        concentrations = group["concentration"].astype(float).to_numpy()
        ratios = group["ratio"].astype(float).to_numpy()
        if linregress:
            slope, intercept, r_value, _, _ = linregress(concentrations, ratios)
        else:
            slope, intercept = np.polyfit(concentrations, ratios, 1)
            r_value = np.corrcoef(concentrations, ratios)[0, 1]
        r_squared = r_value ** 2
        mean_combined = group["combined_ratio"].mean()
        print(f"Analyzing {comp_name[0]} ({product}):")
        print(f"  Slope: {slope:.4f}, Intercept: {intercept:.4f}, R^2: {r_squared:.4f}, Ratio to noise: {mean_combined:.4f}")

        # Plot
        x_vals = np.linspace(concentrations.min(), concentrations.max(), 100)
        y_vals = intercept + slope * x_vals
        plt.figure()
        plt.scatter(concentrations, ratios, label="data")
        plt.plot(x_vals, y_vals, color="red", label="fit")
        plt.xlabel("Concentration (mM)")
        plt.ylabel("Area_product / Area_IS")
        plt.title(f"Calibration for {comp_name[0]}")
        plt.legend()
        out_plot = f"calibration_{comp_name[0]}.png"
        plt.tight_layout()
        plt.savefig(out_plot)
        plt.close()
        print(f"Saved plot to {out_plot}")

        # Save to Excel
        out_file = f"{folder}/calibration_{comp_name[0]}.xlsx"
        with pd.ExcelWriter(out_file) as writer:
            group.to_excel(writer, index=False, sheet_name="data")
            pd.DataFrame({
                "slope": [slope],
                "intercept": [intercept],
                "r_squared": [r_squared],
                "ratio_to_noise": [mean_combined],
                "internal_standard_concentration": [group["internal_standard_concentration"].iloc[0]],
            }).to_excel(writer, index=False, sheet_name="fit")
        print(f"Saved calibration data to {out_file}")


def get_calibration_files(folder: str) -> List[str]:
    """Return list of calibration csv files in folder."""
    files: List[str] = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(".csv") and "calibration" in fname.lower():
            files.append(os.path.join(folder, fname))
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Process calibration CSV files.")
    parser.add_argument("folder", nargs="?", help="Folder containing calibration CSV files.")
    args = parser.parse_args()

    folder = args.folder
    if not folder:
        folder = input("Enter folder path containing calibration csv files [example_calibration]: ").strip() or "example_calibration"
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder '{folder}' not found")

    files = get_calibration_files(folder)
    if not files:
        raise ValueError("No calibration csv files found in the provided folder")

    results = []
    actual_product_names = {}
    internal_std_name: Optional[str] = None
    internal_std_conc: Optional[float] = None
    signal: Optional[str] = None
    for path in files:
        product, concentration = parse_filename(path)
        df = pd.read_csv(path)
        filtered, signal = select_signal(df, signal)
        record, internal_std_name, internal_std_conc = handle_rows(
            filtered, product, internal_std_name, internal_std_conc, actual_product_names
        )
        record.update({
            "file": os.path.basename(path),
            "signal": signal,
            "concentration": concentration,
        })
        results.append(record)

    data = pd.DataFrame(results)
    analyze(data, folder)


if __name__ == "__main__":
    main()
