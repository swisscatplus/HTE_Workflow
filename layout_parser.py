import argparse
import pandas as pd
import string
from typing import Dict, Tuple


EXPERIMENT_COL_START = 17  # column index in Excel where experiment columns begin


def load_experiment_mapping(xl: pd.ExcelFile) -> Dict[int, str]:
    """Return mapping of column index to experiment label."""
    header_rows = pd.read_excel(xl, 'Experiment Definition', header=None, nrows=2)
    mapping = {}
    for idx, val in header_rows.iloc[1].items():
        if isinstance(val, str) and val.startswith('Experiment'):
            mapping[idx] = val
    return mapping


def read_experiment_definition(path: str) -> Tuple[pd.DataFrame, Dict[int, str]]:
    xl = pd.ExcelFile(path)
    mapping = load_experiment_mapping(xl)
    df = pd.read_excel(xl, 'Experiment Definition', header=7)
    df.rename(columns={df.columns[idx]: label for idx, label in mapping.items()}, inplace=True)
    return df, mapping


def map_experiments_to_wells(num_experiments: int) -> Tuple[pd.DataFrame, Dict[str, Tuple[str, int]]]:
    layouts = {24: (4, 6), 48: (6, 8), 96: (8, 12)}
    if num_experiments not in layouts:
        raise ValueError(f"Unsupported number of experiments: {num_experiments}")
    rows, cols = layouts[num_experiments]
    row_labels = list(string.ascii_uppercase[:rows])
    col_labels = list(range(1, cols + 1))
    layout = pd.DataFrame(index=row_labels, columns=col_labels)
    mapping: Dict[str, Tuple[str, int]] = {}
    exp = 1
    for r in row_labels:
        for c in col_labels:
            layout.loc[r, c] = f"Experiment {exp}"
            mapping[f"Experiment {exp}"] = (r, c)
            exp += 1
            if exp > num_experiments:
                break
        if exp > num_experiments:
            break
    return layout, mapping


def build_layout(df: pd.DataFrame, mapping: Dict[int, str]) -> Tuple[pd.DataFrame, pd.Series]:
    experiments = list(mapping.values())
    num_experiments = len(experiments)
    plate_layout, well_map = map_experiments_to_wells(num_experiments)

    results: Dict[str, pd.DataFrame] = {}
    totals: Dict[str, float] = {}

    reagent_rows = df[df['TYPE'] == 'Product']
    for _, row in reagent_rows.iterrows():
        reagent = row['LABEL']
        unit = row['UNIT'] if isinstance(row['UNIT'], str) else ''
        data = pd.DataFrame(0.0, index=plate_layout.index, columns=plate_layout.columns)
        for exp_label in experiments:
            val = row.get(exp_label)
            if pd.isna(val):
                continue
            r, c = well_map[exp_label]
            data.loc[r, c] = val
        col_name = f"{reagent} ({unit})".strip()
        results[col_name] = data
        totals[col_name] = data.sum().sum()

    layout_df = pd.concat(results, axis=1)
    layout_df.index.name = 'Row'
    totals_series = pd.Series(totals, name='Total')
    return layout_df, totals_series


def main() -> None:
    parser = argparse.ArgumentParser(description='Parse HTE dispense file')
    parser.add_argument('excel', help='Input Excel file from dispense machine')
    parser.add_argument('--output', default='parsed_layout.xlsx', help='Output Excel file')
    args = parser.parse_args()

    df, mapping = read_experiment_definition(args.excel)
    layout_df, totals = build_layout(df, mapping)

    with pd.ExcelWriter(args.output) as writer:
        layout_df.to_excel(writer, sheet_name='per_well')
        totals.to_frame().to_excel(writer, sheet_name='totals')
    print(f'Wrote layout to {args.output}')


if __name__ == '__main__':
    main()
