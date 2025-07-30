# HTE Dispense Calculator

This repository contains a Python script for calculating reagent and solvent amounts for high-throughput experimentation (HTE) reactions. The script guides the user through entering reagents, specifying dispensing locations on a plate, and calculating masses or volumes required in each well.

## Usage

```
python hte_calculator.py [--preload my_reagents.py]
```

The script will interactively prompt for reagent information and plate setup. Results are saved as an Excel file summarizing the amounts for each well and total consumption. If `--preload` is supplied, the given Python file must define `PRELOADED_REAGENTS`, a list of dictionaries describing reagents to load before prompting.

During execution you will be asked for a reaction name (used for the Excel/figure filenames) and the desired plate layout (24, 48 or 96 wells). After entering reagents, the script displays the current list and lets you add more if needed. Warnings are shown if any well lacks solvent or the limiting reagent. A colour-coded layout image is also created for quick visual inspection of reagent distribution.
Solvent amounts are applied directly according to each solvent's location mask rather than being averaged over all solvents.

## Requirements
- Python 3.8+
- pandas
- requests
- openpyxl
- matplotlib

Install dependencies with:

```
pip install -r requirements.txt
```

