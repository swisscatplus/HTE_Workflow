# HTE Dispense Calculator

This repository contains a Python script for calculating reagent and solvent amounts for high-throughput experimentation (HTE) reactions. The script guides the user through entering reagents, specifying dispensing locations on a plate, and calculating masses or volumes required in each well.

## Usage

```
python hte_calculator.py
```

The script will interactively prompt for reagent information and plate setup. Results are saved as an Excel file summarizing the amounts for each well and total consumption.

## Requirements
- Python 3.8+
- pandas
- requests
- openpyxl

Install dependencies with:

```
pip install -r requirements.txt
```

