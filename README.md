# HTE Dispense Calculator

This repository contains a Python script for calculating reagent and solvent amounts for high-throughput experimentation (HTE) reactions. The script guides the user through entering reagents, specifying dispensing locations on a plate, and calculating masses or volumes required in each well.

## Usage

```
python hte_calculator.py
```

The script will interactively prompt for reagent information and plate setup. Results are saved as an Excel file summarizing the amounts for each well and total consumption.

During execution you will be asked for a reaction name (used for the Excel/figure filenames) and the desired plate layout (24, 48 or 96 wells). After entering reagents, the script displays the current list and lets you add more if needed. Warnings are shown if any well lacks solvent or the limiting reagent. A colour-coded layout image is also created for quick visual inspection of reagent distribution.

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

