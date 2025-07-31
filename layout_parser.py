import argparse
import math
import string
from typing import Dict, Tuple, List, Mapping

import pandas as pd
import matplotlib.pyplot as plt


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


def build_layout(df: pd.DataFrame, mapping: Mapping[int, str]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Return layout dataframe, totals, final volumes and plate layout."""
    experiments: List[str] = list(mapping.values())
    num_experiments = len(experiments)
    plate_layout, well_map = map_experiments_to_wells(num_experiments)

    results: Dict[str, pd.DataFrame] = {}
    totals: Dict[str, float] = {}
    final_volume = pd.DataFrame(0.0, index=plate_layout.index, columns=plate_layout.columns)

    reagent_rows = df[df['TYPE'] == 'Product']
    for _, row in reagent_rows.iterrows():
        reagent = row['LABEL']
        unit = row['UNIT'] if isinstance(row['UNIT'], str) else ''
        data = pd.DataFrame(0.0, index=plate_layout.index, columns=plate_layout.columns)
        for exp_label in experiments:
            val = row.get(exp_label)
            if pd.isna(val):
                val = row.get('DEFAULT')
            if pd.isna(val):
                val = 0.0
            r, c = well_map[exp_label]
            data.loc[r, c] = float(val)
        col_name = f"{reagent} ({unit})".strip()
        results[col_name] = data
        totals[col_name] = data.sum().sum()

        if isinstance(unit, str) and 'microliter' in unit.lower():
            final_volume += data

    layout_df = pd.concat(results, axis=1)
    layout_df.index.name = 'Row'
    totals_series = pd.Series(totals, name='Total')
    return layout_df, totals_series, final_volume, plate_layout, results


def visualize_reagents(per_reagent: Dict[str, pd.DataFrame], final_volume: pd.DataFrame, output: str) -> None:
    """Save heatmaps of reagent concentrations."""
    if not per_reagent:
        return

    n = len(per_reagent)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, (name, df) in zip(axes, per_reagent.items()):
        if 'microliter' in name.lower():
            conc = df / final_volume.replace(0, float('nan'))
            im = ax.imshow(conc.values.astype(float), cmap='viridis', vmin=0, vmax=conc.max().max())
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('vol frac')
        else:
            im = ax.imshow(df.values.astype(float), cmap='Blues')
        ax.set_xticks(range(df.shape[1]))
        ax.set_xticklabels(df.columns)
        ax.set_yticks(range(df.shape[0]))
        ax.set_yticklabels(df.index)
        ax.set_title(name)
    for ax in axes[n:]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(output)
    plt.close(fig)


def plot_experiment_map(plate_layout: pd.DataFrame, output: str) -> None:
    """Visualize which well corresponds to which experiment."""
    fig, ax = plt.subplots(figsize=(plate_layout.shape[1], plate_layout.shape[0]))
    ax.imshow([[0] * plate_layout.shape[1]] * plate_layout.shape[0], cmap='Greys', vmin=0, vmax=1)
    for i, row in enumerate(plate_layout.index):
        for j, col in enumerate(plate_layout.columns):
            label = plate_layout.loc[row, col]
            ax.text(j, i, label.split()[-1], ha='center', va='center', color='black')
    ax.set_xticks(range(len(plate_layout.columns)))
    ax.set_xticklabels(plate_layout.columns)
    ax.set_yticks(range(len(plate_layout.index)))
    ax.set_yticklabels(plate_layout.index)
    plt.tight_layout()
    plt.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Parse HTE dispense file')
    parser.add_argument('excel', help='Input Excel file from dispense machine')
    parser.add_argument('--output', default='parsed_layout.xlsx', help='Output Excel file')
    parser.add_argument('--image', default='layout.png', help='Output image for reagent distribution')
    parser.add_argument('--map', default='experiment_map.png', help='Output image showing experiment numbers')
    args = parser.parse_args()

    df, mapping = read_experiment_definition(args.excel)
    layout_df, totals, final_vol, plate_layout, per_reagent = build_layout(df, mapping)

    with pd.ExcelWriter(args.output) as writer:
        layout_df.to_excel(writer, sheet_name='per_well')
        totals.to_frame().to_excel(writer, sheet_name='totals')
        final_vol.to_excel(writer, sheet_name='final_volume')
        plate_layout.to_excel(writer, sheet_name='experiment_map')
    print(f'Wrote layout to {args.output}')

    visualize_reagents(per_reagent, final_vol, args.image)
    plot_experiment_map(plate_layout, args.map)
    print(f'Images saved to {args.image} and {args.map}')


if __name__ == '__main__':
    main()
