import argparse
import math
import string
from typing import Dict, Tuple, List, Mapping

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from hte_workflow.paths import DATA_DIR, OUT_DIR, ensure_dirs


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
    """Return a plate layout mapping experiments column-wise."""
    layouts = {24: (4, 6), 48: (6, 8), 96: (8, 12)}
    if num_experiments not in layouts:
        raise ValueError(f"Unsupported number of experiments: {num_experiments}")

    rows, cols = layouts[num_experiments]
    row_labels = list(string.ascii_uppercase[:rows])
    col_labels = list(range(1, cols + 1))

    layout = pd.DataFrame(index=row_labels, columns=col_labels)
    mapping: Dict[str, Tuple[str, int]] = {}

    exp = 1
    for c in col_labels:
        for r in row_labels:
            if exp > num_experiments:
                break
            layout.loc[r, c] = f"Experiment {exp}"
            mapping[f"Experiment {exp}"] = (r, c)
            exp += 1
        if exp > num_experiments:
            break

    return layout, mapping


def build_layout(
    df: pd.DataFrame, mapping: Mapping[int, str]
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], Tuple[List[str], List[str]]]:
    """Return layout dataframe, totals, final volumes and plate layout.

    Also categorises reagents into solids and liquids for workflow visualisation.
    """
    experiments: List[str] = list(mapping.values())
    num_experiments = len(experiments)
    plate_layout, well_map = map_experiments_to_wells(num_experiments)

    results: Dict[str, pd.DataFrame] = {}
    totals: Dict[str, float] = {}
    final_volume = pd.DataFrame(0.0, index=plate_layout.index, columns=plate_layout.columns)

    solids: List[str] = []
    liquids: List[str] = []

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
            liquids.append(reagent)
        else:
            solids.append(reagent)

    layout_df = pd.concat(results, axis=1)
    layout_df.index.name = 'Row'
    totals_series = pd.Series(totals, name='Total')
    return layout_df, totals_series, final_volume, plate_layout, results, (sorted(set(solids)), sorted(set(liquids)))


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
            mat = conc.values.astype(float)
            im = ax.imshow(mat, cmap='viridis', origin='upper', vmin=0, vmax=conc.max().max())
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('vol frac')
        else:
            im = ax.imshow(df.values.astype(float), cmap='Blues', origin='upper')
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


def extract_workflow_steps(df: pd.DataFrame) -> List[str]:
    """Return ordered workflow steps deduplicated by node id.

    The Excel file may list multiple parameters for a single workflow node. We
    use column ``ID.1`` which contains the node identifier to group those rows
    together. For orbital shaker nodes, the heating/cooling value and rotation
    speed are appended in parentheses.  For dispense nodes the compound name and
    the amount dispensed (or range if it varies across wells) are appended.
    """

    steps: List[str] = []
    seen_ids = set()
    if 'ID.1' not in df.columns:
        return steps

    for node_id in df['ID.1']:
        if pd.isna(node_id) or node_id in seen_ids:
            continue
        group = df[df['ID.1'] == node_id]
        name = str(group['WF NODE'].iloc[0])

        product_rows = group[group['TYPE'] == 'Product'] if 'TYPE' in group.columns else pd.DataFrame()
        if not product_rows.empty:
            prod = product_rows.iloc[0]
            compound = str(prod.get('LABEL', '')).strip()
            unit = str(prod.get('UNIT', '')).strip()
            # collect amounts per experiment similar to build_layout
            exp_cols = [c for c in df.columns if isinstance(c, str) and c.startswith('Experiment')]
            values: List[float] = []
            for col in exp_cols:
                val = prod.get(col)
                if pd.isna(val):
                    val = prod.get('DEFAULT')
                if pd.isna(val):
                    val = 0.0
                values.append(float(val))
            if values:
                min_val = min(values)
                max_val = max(values)
                def fmt(v: float) -> str:
                    return (
                        f"{v:.2f}".rstrip('0').rstrip('.')
                        if not float(v).is_integer()
                        else str(int(v))
                    )
                qty = fmt(min_val) if math.isclose(min_val, max_val) else f"{fmt(min_val)}-{fmt(max_val)}"
                unit_map = {'milligram': 'mg', 'microliter': 'µL'}
                unit_short = unit_map.get(unit.lower(), unit)
                if unit_short:
                    qty += f" {unit_short}"
                name += f": {compound} {qty}".rstrip()
        elif 'Orbital Shaker' in name:
            heating = group.loc[group['PARAMETER'] == 'Heating/Cooling', 'DEFAULT']
            speed = group.loc[group['PARAMETER'] == 'Rotation Speed', 'DEFAULT']
            details: List[str] = []
            if not heating.empty and pd.notna(heating.iloc[0]):
                details.append(f"{heating.iloc[0]} °C")
            if not speed.empty and pd.notna(speed.iloc[0]):
                details.append(f"{speed.iloc[0]} rpm")
            if details:
                name += f" ({', '.join(details)})"

        steps.append(name)
        seen_ids.add(node_id)

    return steps


def plot_workflow(steps: List[str], output: str) -> None:
    """Plot the workflow based on extracted steps."""
    if not steps:
        return

    height = max(2, 0.5 * len(steps))
    fig, ax = plt.subplots(figsize=(8, height))
    ax.axis('off')

    box_height = 0.6
    for i, step in enumerate(steps):
        y = len(steps) - i - 1
        ax.add_patch(
            plt.Rectangle(
                (0.1, y - box_height / 2),
                0.8,
                box_height,
                edgecolor="black",
                facecolor="lightgrey",
            )
        )
        ax.text(0.5, y, step, ha="center", va="center", wrap=True, fontsize=8)
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(0.5, y - box_height / 2),
                xytext=(0.5, y - 1 + box_height / 2),
                arrowprops=dict(arrowstyle="->"),
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(steps) - 0.5)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Parse HTE dispense file')
    parser.add_argument('excel', help='Input Excel file from dispense machine')
    parser.add_argument('--output', default='parsed_layout.xlsx', help='Output Excel file')
    parser.add_argument('--image', default='layout.png', help='Output image for reagent distribution')
    parser.add_argument('--map', default='experiment_map.png', help='Output image showing experiment numbers')
    parser.add_argument('--workflow', default='workflow.png', help='Output workflow diagram')
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir).resolve()

    excel_path = data_dir / args.excel
    output_excel = out_dir / args.output
    output_image = out_dir / args.image
    output_map = out_dir / args.map
    output_workflow = out_dir / args.workflow

    df, mapping = read_experiment_definition(excel_path)
    layout_df, totals, final_vol, plate_layout, per_reagent, (solids, liquids) = build_layout(df, mapping)
    steps = extract_workflow_steps(df)

    with pd.ExcelWriter(output_excel) as writer:
        layout_df.to_excel(writer, sheet_name='per_well')
        totals.to_frame().to_excel(writer, sheet_name='totals')
        final_vol.to_excel(writer, sheet_name='final_volume')
        plate_layout.to_excel(writer, sheet_name='experiment_map')
    print(f'Wrote layout to {args.output}')

    visualize_reagents(per_reagent, final_vol, output_image)
    plot_experiment_map(plate_layout, output_map)
    plot_workflow(steps, output_workflow)
    print(f'Images saved to {args.image}, {args.map} and {args.workflow}')


if __name__ == '__main__':
    main()
