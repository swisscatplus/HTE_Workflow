import json
from pathlib import Path
import argparse
from typing import Any, Dict, List

from hte_workflow.paths import DATA_DIR, OUT_DIR, ensure_dirs


def load_hplc_json(path: str | Path) -> Dict[str, Any]:
    """Load raw HPLC JSON data into a Python dict."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw_file = json.load(f)
        preprocessed_file = raw_file["liquid chromatography aggregate document"]["liquid chromatography document"]\
        [0]["measurement aggregate document"]["measurement document"]
        """
            -----------------File structure-----------------
            measurement_data
                @index
                processed data identifier
                data processing time
                data processing document
                peak list
                    peak
                    -- list of peaks, structure:
                    {'@index': 1, 'peak area': {'value': 497.21370042329727, 'unit': 'mAU.s'},
                    'retention time': {'value': 9.499999999999996, 'unit': 's'}, 'identifier': 'af0a64a6-9df7-46a6-92e6-4c09ef12d69a',
                    'peak end': {'value': 9.832232762464256, 'unit': 's'}, 'relative peak height': {'value': 3.0204194880991024, 'unit': '%'},
                    'peak height': {'value': 99.0104185601515, 'unit': 'mAU'}, 'peak start': {'value': 0.3, 'unit': 's'},
                    'relative peak area': {'value': 3.1529362924203683, 'unit': '%'},
                    'peak value at start': {'value': 155.4746627918148, 'unit': 'mAU'},
                    'peak value at end': {'value': 164.46385570693673, 'unit': 'mAU'}}
            """
        return preprocessed_file


def extract_measurement_identifiers(hplc_data: Dict) -> Dict[str, int]:
    """Extract measurement identifiers from HPLC data."""
    measurement_identifiers = {}
    for index in range(len(hplc_data)):
        measurement_identifiers[index] = hplc_data[index]["measurement identifier"]
    return measurement_identifiers


def main():
    parser = argparse.ArgumentParser(description="Parser for HPLC JSON data")
    parser.add_argument("--analysis_file", default = "export_Leander/250805_NP_Cat_01_MIDAHydrolysis_1_D1B-A1.dx.JSON", help="Path (in data) to HPLC JSON file")
    parser.add_argument("--measurement_name", help="Name of measurement chosen")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Directory with data files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    data_dir = Path(args.data_dir).resolve()

    analysis_file_path = data_dir / args.analysis_file

    raw_data = load_hplc_json(analysis_file_path)
    measurement_identifiers = extract_measurement_identifiers(raw_data)
    if args.measurement_name:
        for index in measurement_identifiers:
            if measurement_identifiers[index] == args.measurement_name:
                chosen_measurement = index
                break
        else:
            raise ValueError(f"Measurement '{args.measurement_name}' not found in the data.")

    measurement_data = raw_data[chosen_measurement]["processed data aggregate document"]["processed data document"][0]
    data_file_name = measurement_data[chosen_measurement]["sample document"]["written name"]
    peak_list = measurement_data["peak list"]["peak"]

    if args.measurement_name:
        return {
            "data_file_name": data_file_name,
            "measurement_identifiers": measurement_identifiers,
            "peak_list": peak_list
        }
    else:
        return {"measurement_identifiers": measurement_identifiers}



if __name__ == "__main__":
    main()
