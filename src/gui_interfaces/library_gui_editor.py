#!/usr/bin/env python3
"""
library_gui_editor.py

Tkinter-based GUI to inspect and edit a chemical library JSON
compatible with `library_and_hci_adapter.py`.

Features
--------
- Shows library metadata (libraryName, dateCreated, createdBy, LibraryIndex).
- Lists all chemicals in the library (ignores the meta header row).
- Search by name / ID / CAS / formula / SwissCat.
- Shows detailed info + optional RDKit structure for the selected chemical.
- Add / update / delete chemicals:
    * All core fields mandatory:
      chemicalName, CASNumber (use "None" if not available), MW, SMILES, Inchi, formula, physicalstate
    * physicalstate = solid | liquid | solution
    * density required only for liquids, but if present must always be float.
- Automatically assigns:
    * chemicalID = max(existing) + 1
    * swissCatNumber = max(existing) + 1
- Bulk import from another library:
    * Uses LibraryIndex: lower index library is the "base".
    * Deduplicate by InChI (same InChI → only one entry).
    * Name conflicts:
        - same InChI, different names → ask user for canonical name.
        - different InChI, same name → ask user for a new unique name for imported compound.
    * chemicalID / swissCatNumber from base library are preserved.
      IDs for imported-from-higher-index library are adjusted if needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re


import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog

from hte_workflow.paths import DATA_DIR
from json_handling.library_and_hci_adapter import (
    _lib_load,
    _lib_save,
    Chemical,
    Quantity,
)

# Optional RDKit / Pillow for structure rendering
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from PIL import ImageTk

    _HAS_RDKIT = True
except Exception:
    Chem = None
    Draw = None
    ImageTk = None
    _HAS_RDKIT = False

try:
    import pubchempy as pcp
    _HAS_PUBCHEM = True
except Exception:
    _HAS_PUBCHEM = False



class LibraryEditorGUI(tk.Tk):
    """
    GUI editor for a single chemical library JSON file.
    """

    def __init__(self, library_path: Path) -> None:
        super().__init__()

        self.title(f"Library editor – {library_path.name}")
        self.geometry("1150x700")

        self.library_path = library_path

        # Internal state
        self.meta: Dict[str, Any] = {}
        self.chemicals: List[Dict[str, Any]] = []  # list of chemical dicts
        self._chem_view: List[int] = []  # mapping listbox index -> chemicals index
        self._selected_index: Optional[int] = None

        # RDKit image holder
        self._chem_img = None

        # Tk variables for search
        self.var_search = tk.StringVar()

        # Form variables
        self.var_id = tk.StringVar()
        self.var_swiss = tk.StringVar()
        self.var_name = tk.StringVar()
        self.var_cas = tk.StringVar()
        self.var_mw_value = tk.StringVar()
        self.var_mw_unit = tk.StringVar(value="g/mol")
        self.var_smiles = tk.StringVar()
        self.var_inchi = tk.StringVar()
        self.var_formula = tk.StringVar()
        self.var_state = tk.StringVar(value="solid")
        self.var_density_value = tk.StringVar()
        self.var_density_unit = tk.StringVar(value="g/mL")
        self.var_conc_value = tk.StringVar()
        self.var_conc_unit = tk.StringVar(value="M")

        # Build UI and load data
        self._build_ui()
        self._load_library()

    # ------------------------------------------------------------------ #
    # Loading / saving
    # ------------------------------------------------------------------ #

    def _load_library(self) -> None:
        items = _lib_load(self.library_path)
        if not items:
            messagebox.showerror("Library", f"Library file is empty: {self.library_path}")
            self.meta = {}
            self.chemicals = []
            return

        # First element is meta, if it has libraryName
        first = items[0]
        if isinstance(first, dict) and "libraryName" in first:
            self.meta = first
            self.chemicals = items[1:]
        else:
            # no meta block, treat all as chemicals
            self.meta = {}
            self.chemicals = items

        self._chem_view = list(range(len(self.chemicals)))
        self._refresh_meta_view()
        self._refresh_chem_list(preserve_selection=False)

    def _save_library(self) -> None:
        to_save: List[Dict[str, Any]] = []
        if self.meta:
            to_save.append(self.meta)
        to_save.extend(self.chemicals)
        _lib_save(self.library_path, to_save)
        print(f"[info] Library saved to {self.library_path}")

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        main = ttk.Frame(self, padding=8)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(1, weight=1)

        # --- Meta info at top ---
        meta_frame = ttk.LabelFrame(main, text="Library metadata", padding=6)
        meta_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        meta_frame.columnconfigure(0, weight=1)
        meta_frame.columnconfigure(1, weight=1)
        meta_frame.columnconfigure(2, weight=1)
        meta_frame.columnconfigure(3, weight=1)

        self.lbl_libname = ttk.Label(meta_frame, text="libraryName: –")
        self.lbl_libname.grid(row=0, column=0, sticky="w")

        self.lbl_date = ttk.Label(meta_frame, text="dateCreated: –")
        self.lbl_date.grid(row=0, column=1, sticky="w")

        self.lbl_creator = ttk.Label(meta_frame, text="createdBy: –")
        self.lbl_creator.grid(row=0, column=2, sticky="w")

        self.lbl_index = ttk.Label(meta_frame, text="LibraryIndex: –")
        self.lbl_index.grid(row=0, column=3, sticky="w")

        # --- Left panel: search + list ---
        left = ttk.Frame(main, padding=(0, 6, 6, 0))
        left.grid(row=1, column=0, sticky="nsw")
        left.rowconfigure(2, weight=1)
        left.columnconfigure(0, weight=1)

        ttk.Label(left, text="Search (name / ID / CAS / formula / SwissCat)").grid(
            row=0, column=0, sticky="w"
        )
        search_entry = ttk.Entry(left, textvariable=self.var_search)
        search_entry.grid(row=1, column=0, sticky="ew", pady=2)

        btn_search_frame = ttk.Frame(left)
        btn_search_frame.grid(row=1, column=1, sticky="w")
        ttk.Button(btn_search_frame, text="Search", command=self._on_search).grid(row=0, column=0, padx=2)
        ttk.Button(btn_search_frame, text="Clear", command=self._on_clear_search).grid(row=0, column=1, padx=2)

        self.listbox = tk.Listbox(left, height=20, exportselection=False)
        self.listbox.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(4, 0))
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)

        btn_list_frame = ttk.Frame(left)
        btn_list_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(btn_list_frame, text="Reload from disk", command=self._load_library).grid(
            row=0, column=0, padx=2, sticky="ew"
        )
        ttk.Button(btn_list_frame, text="Bulk import...", command=self._bulk_import_library).grid(
            row=0, column=1, padx=2, sticky="ew"
        )

        # --- Right panel: top = details, bottom = edit form ---
        right = ttk.Frame(main, padding=(0, 6, 0, 0))
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=0)
        right.rowconfigure(1, weight=1)

        # Details frame
        details = ttk.LabelFrame(right, text="Chemical details", padding=6)
        details.grid(row=0, column=0, sticky="ew")
        details.columnconfigure(0, weight=1)
        details.columnconfigure(1, weight=0)

        self.chem_info_label = ttk.Label(details, text="", justify="left")
        self.chem_info_label.grid(row=0, column=0, sticky="w")

        self.chem_img_label = ttk.Label(details)
        self.chem_img_label.grid(row=0, column=1, sticky="e", padx=10)

        # Edit frame
        edit = ttk.LabelFrame(right, text="Edit / Add chemical", padding=6)
        edit.grid(row=1, column=0, sticky="nsew", pady=(6, 0))
        edit.columnconfigure(1, weight=1)
        edit.columnconfigure(3, weight=1)

        r = 0
        ttk.Label(edit, text="chemicalID").grid(row=r, column=0, sticky="w")
        ttk.Entry(edit, textvariable=self.var_id, state="readonly", width=10).grid(
            row=r, column=1, sticky="w", pady=2
        )
        ttk.Label(edit, text="SwissCatNumber").grid(row=r, column=2, sticky="w")
        ttk.Entry(edit, textvariable=self.var_swiss, state="readonly", width=10).grid(
            row=r, column=3, sticky="w", pady=2
        )

        r += 1
        ttk.Label(edit, text="Name*").grid(row=r, column=0, sticky="w")
        ttk.Entry(edit, textvariable=self.var_name).grid(row=r, column=1, columnspan=3, sticky="ew", pady=2)

        r += 1
        ttk.Label(edit, text="CAS* (use 'None' if not available)").grid(row=r, column=0, sticky="w")
        ttk.Entry(edit, textvariable=self.var_cas).grid(row=r, column=1, sticky="ew", pady=2)
        ttk.Label(edit, text="Formula*").grid(row=r, column=2, sticky="w")
        ttk.Entry(edit, textvariable=self.var_formula).grid(row=r, column=3, sticky="ew", pady=2)

        r += 1
        ttk.Label(edit, text="MW value*").grid(row=r, column=0, sticky="w")
        ttk.Entry(edit, textvariable=self.var_mw_value, width=12).grid(row=r, column=1, sticky="w", pady=2)
        ttk.Label(edit, text="MW unit").grid(row=r, column=2, sticky="w")
        ttk.Entry(edit, textvariable=self.var_mw_unit, width=10).grid(row=r, column=3, sticky="w", pady=2)

        r += 1
        ttk.Label(edit, text="SMILES*").grid(row=r, column=0, sticky="w")
        ttk.Entry(edit, textvariable=self.var_smiles).grid(row=r, column=1, columnspan=3, sticky="ew", pady=2)

        r += 1
        ttk.Label(edit, text="InChI*").grid(row=r, column=0, sticky="w")
        ttk.Entry(edit, textvariable=self.var_inchi).grid(row=r, column=1, columnspan=3, sticky="ew", pady=2)

        r += 1
        ttk.Label(edit, text="Physical state*").grid(row=r, column=0, sticky="w")
        cb_state = ttk.Combobox(
            edit,
            textvariable=self.var_state,
            values=["solid", "liquid", "solution"],
            state="readonly",
            width=10,
        )
        cb_state.grid(row=r, column=1, sticky="w", pady=2)
        cb_state.bind("<<ComboboxSelected>>", self._on_state_changed)

        r += 1
        ttk.Label(edit, text="Density (liquid only)").grid(row=r, column=0, sticky="w")
        ttk.Entry(edit, textvariable=self.var_density_value, width=10).grid(
            row=r, column=1, sticky="w", pady=2
        )
        ttk.Entry(edit, textvariable=self.var_density_unit, width=10).grid(
            row=r, column=2, sticky="w", pady=2
        )

        r += 1
        ttk.Label(edit, text="Concentration (for solutions, optional)").grid(row=r, column=0, sticky="w")
        ttk.Entry(edit, textvariable=self.var_conc_value, width=10).grid(
            row=r, column=1, sticky="w", pady=2
        )
        ttk.Entry(edit, textvariable=self.var_conc_unit, width=10).grid(
            row=r, column=2, sticky="w", pady=2
        )

        # Buttons at bottom of edit
        r += 1
        btn_frame = ttk.Frame(edit)
        btn_frame.grid(row=r, column=0, columnspan=5, sticky="ew", pady=(8, 0))
        for col in range(4):
            btn_frame.columnconfigure(col, weight=1)

        ttk.Button(btn_frame, text="New chemical", command=self._on_new_chemical).grid(
            row=0, column=0, sticky="ew", padx=2
        )
        ttk.Button(btn_frame, text="Save / Update", command=self._on_save_chemical).grid(
            row=0, column=1, sticky="ew", padx=2
        )
        ttk.Button(btn_frame, text="Delete", command=self._on_delete_chemical).grid(
            row=0, column=2, sticky="ew", padx=2
        )
        ttk.Button(btn_frame, text="Close", command=self.destroy).grid(
            row=0, column=3, sticky="ew", padx=2
        )
        ttk.Button(btn_frame, text="Fetch from PubChem", command=self._on_fetch_pubchem).grid(
            row=0, column=4, sticky="ew", padx=2
        )
        btn_frame.columnconfigure(4, weight=1)

        self._on_state_changed()  # ensure correct state on first draw

    # ------------------------------------------------------------------ #
    # Meta + list refresh
    # ------------------------------------------------------------------ #

    def _refresh_meta_view(self) -> None:
        m = self.meta or {}
        self.lbl_libname.config(text=f"libraryName: {m.get('libraryName', '–')}")
        self.lbl_date.config(text=f"dateCreated: {m.get('dateCreated', '–')}")
        self.lbl_creator.config(text=f"createdBy: {m.get('createdBy', '–')}")
        self.lbl_index.config(text=f"LibraryIndex: {m.get('LibraryIndex', '–')}")

    def _refresh_chem_list(self, preserve_selection: bool = True) -> None:
        """
        Refresh the listbox from self.chemicals using current _chem_view.
        If _chem_view is empty, create it as range(len(chemicals)).
        """
        if not self.chemicals:
            self.listbox.delete(0, tk.END)
            self._chem_view = []
            self._selected_index = None
            self._set_chem_details(None)
            return

        if not self._chem_view:
            self._chem_view = list(range(len(self.chemicals)))

        current_sel = self.listbox.curselection()
        sel_index = current_sel[0] if current_sel else None
        if not preserve_selection:
            sel_index = None

        self.listbox.delete(0, tk.END)
        for idx in self._chem_view:
            chem = self.chemicals[idx]
            name = chem.get("chemicalName", "")
            cid = chem.get("chemicalID", "")
            swiss = chem.get("swissCatNumber", "")
            self.listbox.insert(tk.END, f"{cid}: {name} (SwissCat: {swiss})")

        # restore selection if we can
        if sel_index is not None and 0 <= sel_index < len(self._chem_view):
            self.listbox.select_set(sel_index)
            self.listbox.event_generate("<<ListboxSelect>>")

    # ------------------------------------------------------------------ #
    # List / search handlers
    # ------------------------------------------------------------------ #

    def _on_search(self) -> None:
        query = self.var_search.get().strip().lower()
        if not query:
            self._chem_view = list(range(len(self.chemicals)))
            self._refresh_chem_list(preserve_selection=False)
            return

        new_view: List[int] = []
        for i, chem in enumerate(self.chemicals):
            name = str(chem.get("chemicalName", "")).lower()
            cid = str(chem.get("chemicalID", "")).lower()
            cas = str(chem.get("CASNumber", "")).lower()
            formula = str(chem.get("molecularFormula", "")).lower()
            swiss = str(chem.get("swissCatNumber", "")).lower()
            if (
                query in name
                or query in cid
                or query in cas
                or query in formula
                or query in swiss
            ):
                new_view.append(i)

        self._chem_view = new_view
        self._refresh_chem_list(preserve_selection=False)

    def _on_clear_search(self) -> None:
        self.var_search.set("")
        self._chem_view = list(range(len(self.chemicals)))
        self._refresh_chem_list(preserve_selection=False)

    def _on_list_select(self, event=None) -> None:
        sel = self.listbox.curselection()
        if not sel or not self._chem_view:
            self._selected_index = None
            self._set_chem_details(None)
            self._clear_form()
            return

        row = sel[0]
        if row < 0 or row >= len(self._chem_view):
            return

        idx = self._chem_view[row]
        self._selected_index = idx
        chem = self.chemicals[idx]
        self._fill_form(chem)
        self._set_chem_details(chem)

    # ------------------------------------------------------------------ #
    # Details panel + RDKit
    # ------------------------------------------------------------------ #

    def _set_chem_details(self, chem: Optional[Dict[str, Any]]) -> None:
        if chem is None:
            self.chem_info_label.config(text="")
            self.chem_img_label.config(image="")
            self._chem_img = None
            return

        name = chem.get("chemicalName", "")
        cid = chem.get("chemicalID", "")
        cas = chem.get("CASNumber", "")
        formula = chem.get("molecularFormula", "")
        swiss = chem.get("swissCatNumber", "")
        state = chem.get("physicalstate", "")

        mm = chem.get("molecularMass")
        mass_val = None
        mass_unit = ""
        if isinstance(mm, dict):
            mass_val = mm.get("value")
            mass_unit = mm.get("unit", "")

        dens = chem.get("density")
        dens_val = None
        dens_unit = ""
        if isinstance(dens, dict):
            dens_val = dens.get("value")
            dens_unit = dens.get("unit", "")

        conc = chem.get("concentration")
        conc_val = None
        conc_unit = ""
        if isinstance(conc, dict):
            conc_val = conc.get("value")
            conc_unit = conc.get("unit", "")

        lines = []
        if name:
            lines.append(f"Name: {name}")
        if cid:
            lines.append(f"ID: {cid}")
        if swiss:
            lines.append(f"SwissCat: {swiss}")
        if cas:
            lines.append(f"CAS: {cas}")
        if formula:
            lines.append(f"Formula: {formula}")
        if mass_val is not None:
            lines.append(f"MW: {mass_val} {mass_unit}".strip())
        if state:
            lines.append(f"Physical state: {state}")
        if dens_val is not None:
            lines.append(f"Density: {dens_val} {dens_unit}".strip())
        if conc_val is not None:
            lines.append(f"Concentration: {conc_val} {conc_unit}".strip())

        self.chem_info_label.config(text="\n".join(lines))

        # RDKit structure from SMILES
        if _HAS_RDKIT:
            smiles = chem.get("smiles", "")
            if not smiles:
                self.chem_img_label.config(image="")
                self._chem_img = None
                return
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    self.chem_img_label.config(image="")
                    self._chem_img = None
                    return
                img = Draw.MolToImage(mol, size=(220, 220))
                self._chem_img = ImageTk.PhotoImage(img)
                self.chem_img_label.config(image=self._chem_img)
            except Exception:
                self.chem_img_label.config(image="")
                self._chem_img = None
        else:
            self.chem_img_label.config(image="")
            self._chem_img = None

    # ------------------------------------------------------------------ #
    # Form handling
    # ------------------------------------------------------------------ #

    def _clear_form(self) -> None:
        self.var_id.set("")
        self.var_swiss.set("")
        self.var_name.set("")
        self.var_cas.set("")
        self.var_mw_value.set("")
        self.var_mw_unit.set("g/mol")
        self.var_smiles.set("")
        self.var_inchi.set("")
        self.var_formula.set("")
        self.var_state.set("solid")
        self.var_density_value.set("")
        self.var_density_unit.set("g/mL")
        self.var_conc_value.set("")
        self.var_conc_unit.set("M")
        self._on_state_changed()

    def _fill_form(self, chem: Dict[str, Any]) -> None:
        self.var_id.set(str(chem.get("chemicalID", "")))
        self.var_swiss.set(str(chem.get("swissCatNumber", "")))
        self.var_name.set(chem.get("chemicalName", ""))
        self.var_cas.set(chem.get("CASNumber", ""))
        self.var_formula.set(chem.get("molecularFormula", ""))

        mm = chem.get("molecularMass", {})
        if isinstance(mm, dict):
            self.var_mw_value.set(str(mm.get("value", "")))
            self.var_mw_unit.set(str(mm.get("unit", "g/mol")))
        else:
            self.var_mw_value.set("")
            self.var_mw_unit.set("g/mol")

        self.var_smiles.set(chem.get("smiles", ""))
        self.var_inchi.set(chem.get("Inchi", ""))

        self.var_state.set(chem.get("physicalstate", "solid"))

        dens = chem.get("density", {})
        if isinstance(dens, dict):
            self.var_density_value.set(str(dens.get("value", "")))
            self.var_density_unit.set(str(dens.get("unit", "g/mL")))
        else:
            self.var_density_value.set("")
            self.var_density_unit.set("g/mL")

        conc = chem.get("concentration", {})
        if isinstance(conc, dict):
            self.var_conc_value.set(str(conc.get("value", "")))
            self.var_conc_unit.set(str(conc.get("unit", "M")))
        else:
            self.var_conc_value.set("")
            self.var_conc_unit.set("M")

        self._on_state_changed()

    def _on_state_changed(self, event=None) -> None:
        """
        Hook for physical state changes.

        We keep fields enabled in all cases, but validation enforces that:
        - MW is always a valid float.
        - If density is provided, it must be a valid float.
        - If state == 'liquid', density must be provided and numeric.
        """
        # Currently no widget disabling to keep flexibility, only validation rules.
        return

    # ------------------------------------------------------------------ #
    # New / save / delete
    # ------------------------------------------------------------------ #

    def _next_ids(self) -> Tuple[str, str]:
        """
        Compute next chemicalID and swissCatNumber from existing chemicals.
        If none exist, start at "1" and "1".
        """
        max_cid = 0
        max_swiss = 0
        for chem in self.chemicals:
            try:
                cid = int(str(chem.get("chemicalID", "0")))
                if cid > max_cid:
                    max_cid = cid
            except ValueError:
                pass
            try:
                sc = int(str(chem.get("swissCatNumber", "0")))
                if sc > max_swiss:
                    max_swiss = sc
            except ValueError:
                pass
        return str(max_cid + 1), str(max_swiss + 1)

    def _on_new_chemical(self) -> None:
        """
        Prepare form for a new chemical, assigning next ID / SwissCatNumber.
        """
        self._selected_index = None
        self._clear_form()
        new_id, new_swiss = self._next_ids()
        self.var_id.set(new_id)
        self.var_swiss.set(new_swiss)

    def _validate_form(self) -> Optional[str]:
        """
        Validate fields; return None if OK, or error string if invalid.
        """
        name = self.var_name.get().strip()
        cas = self.var_cas.get().strip()
        formula = self.var_formula.get().strip()
        mw_val = self.var_mw_value.get().strip()
        mw_unit = self.var_mw_unit.get().strip() or "g/mol"
        smiles = self.var_smiles.get().strip()
        inchi = self.var_inchi.get().strip()
        state = self.var_state.get().strip().lower()

        if not name:
            return "chemicalName is required."
        if not cas:
            return "CASNumber is required (use 'None' if not available)."
        if not formula:
            return "molecularFormula is required."
        if not mw_val:
            return "Molecular mass value is required."
        if not smiles:
            return "SMILES is required."
        if not inchi:
            return "InChI is required."
        if state not in {"solid", "liquid", "solution"}:
            return "Physical state must be solid, liquid, or solution."

        # MW must be numeric
        try:
            float(mw_val)
        except ValueError:
            return "Molecular mass value must be numeric (float)."

        dens_val = self.var_density_value.get().strip()
        if state == "liquid":
            if not dens_val:
                return "Density is required for liquids."
        # If density is present (any state), it must be numeric
        if dens_val:
            try:
                float(dens_val)
            except ValueError:
                return "Density value must be numeric (float)."

        # If concentration present, it must be numeric
        conc_val = self.var_conc_value.get().strip()
        if conc_val:
            try:
                float(conc_val)
            except ValueError:
                return "Concentration value must be numeric (float)."

        # If all checks passed
        return None

    def _collect_form_chemical(self) -> Chemical:
        """
        Build a Chemical dataclass from the form (without writing it yet).
        Automatically uses current var_id and var_swiss.
        """
        chemicalID = self.var_id.get().strip()
        if not chemicalID:
            # fallback if user somehow clears it
            chemicalID, _ = self._next_ids()
            self.var_id.set(chemicalID)

        swissCat = self.var_swiss.get().strip()
        if not swissCat:
            _, swissCat = self._next_ids()
            self.var_swiss.set(swissCat)

        name = self.var_name.get().strip()
        cas = self.var_cas.get().strip()
        formula = self.var_formula.get().strip()
        mw_val = float(self.var_mw_value.get().strip())
        mw_unit = self.var_mw_unit.get().strip() or "g/mol"
        smiles = self.var_smiles.get().strip()
        inchi = self.var_inchi.get().strip()
        state = self.var_state.get().strip().lower()

        # density (liquid only mandatory, but can exist otherwise)
        dens_q = None
        dens_val_txt = self.var_density_value.get().strip()
        if dens_val_txt:
            dens_q = Quantity(float(dens_val_txt), self.var_density_unit.get().strip() or "g/mL")

        # concentration (for solutions, optional)
        conc_q = None
        conc_val_txt = self.var_conc_value.get().strip()
        if conc_val_txt:
            conc_q = Quantity(float(conc_val_txt), self.var_conc_unit.get().strip() or "M")

        chem = Chemical(
            chemicalID=chemicalID,
            chemicalName=name,
            CASNumber=cas,
            molecularMass=Quantity(mw_val, mw_unit),
            smiles=smiles,
            Inchi=inchi,
            molecularFormula=formula,
            swissCatNumber=swissCat,
            physicalstate=state,
            density=dens_q,
            concentration=conc_q,
        )
        return chem

    def _on_save_chemical(self) -> None:
        """
        Add or update a chemical using the form contents.

        Extra checks:
        - Prevent manual duplicates by InChI:
            * If another entry has the same normalized InChI → block save and show error.
        - Resolve name conflicts:
            * If another entry has the same name but a different InChI → ask for a new unique name.
        """
        err = self._validate_form()
        if err is not None:
            messagebox.showerror("Validation error", err)
            return

        chem = self._collect_form_chemical()
        payload = chem.to_json()

        new_cid = str(payload.get("chemicalID", "")).strip()
        new_name = str(payload.get("chemicalName", "")).strip()
        new_inchi = self._get_inchi(payload)

        # ------------------------------------------------------------------
        # 1) Check for duplicate structure by InChI
        #    → only allow once in the library.
        # ------------------------------------------------------------------
        if new_inchi:
            for i, existing in enumerate(self.chemicals):
                # Same record by ID: this is an update, allow it through here.
                if str(existing.get("chemicalID", "")).strip() == new_cid:
                    continue

                existing_inchi = self._get_inchi(existing)
                if existing_inchi and existing_inchi == new_inchi:
                    # Same structure already in library → block duplicate
                    existing_name = str(existing.get("chemicalName", "")).strip()
                    existing_id = str(existing.get("chemicalID", "")).strip()
                    messagebox.showerror(
                        "Duplicate structure",
                        "A chemical with the same InChI already exists in the library:\n\n"
                        f"Name: {existing_name}\n"
                        f"ID:   {existing_id}\n\n"
                        "The library is meant to contain each structure only once.\n"
                        "Please edit the existing entry instead or change the InChI."
                    )
                    return

        # ------------------------------------------------------------------
        # 2) Check for name conflicts with different structures
        #    → same name, different InChI → ask user to rename.
        # ------------------------------------------------------------------
        for i, existing in enumerate(self.chemicals):
            # Same record by ID: skip (we are updating this one)
            if str(existing.get("chemicalID", "")).strip() == new_cid:
                continue

            existing_name = str(existing.get("chemicalName", "")).strip()
            if not existing_name or not new_name:
                continue

            if existing_name == new_name:
                existing_inchi = self._get_inchi(existing)
                # If both have InChI and they differ → name conflict
                if existing_inchi and new_inchi and existing_inchi != new_inchi:
                    msg = (
                        f"The name '{new_name}' is already used for a different compound "
                        "in this library.\n\n"
                        "Please enter a new unique name for this compound:"
                    )
                    from tkinter import simpledialog  # already imported at top, but safe

                    new_name_input = simpledialog.askstring(
                        "Name conflict", msg, initialvalue=new_name + "_2"
                    )
                    if not new_name_input:
                        # User cancelled → abort save
                        return
                    new_name = new_name_input.strip()
                    payload["chemicalName"] = new_name
                    self.var_name.set(new_name)
                    # Only need to resolve one such conflict
                    break

        # ------------------------------------------------------------------
        # 3) Proceed with add / update using chemicalID as before
        # ------------------------------------------------------------------
        cid = payload["chemicalID"]
        existing_idx = None
        for i, c in enumerate(self.chemicals):
            if str(c.get("chemicalID")) == str(cid):
                existing_idx = i
                break

        if existing_idx is None:
            # New chemical
            self.chemicals.append(payload)
            self._chem_view = list(range(len(self.chemicals)))
            self._save_library()
            self._refresh_chem_list(preserve_selection=False)
            messagebox.showinfo("Saved", f"Added new chemical '{payload['chemicalName']}' (ID={cid}).")
        else:
            # Update existing
            self.chemicals[existing_idx] = payload
            self._save_library()
            self._refresh_chem_list(preserve_selection=True)
            messagebox.showinfo("Saved", f"Updated chemical '{payload['chemicalName']}' (ID={cid}).")


    def _on_delete_chemical(self) -> None:
        """
        Delete the currently selected chemical.
        """
        if self._selected_index is None:
            messagebox.showinfo("Delete", "No chemical selected.")
            return

        chem = self.chemicals[self._selected_index]
        name = chem.get("chemicalName", "")
        cid = chem.get("chemicalID", "")
        if not messagebox.askyesno(
            "Confirm delete",
            f"Delete chemical '{name}' (ID={cid})?",
        ):
            return

        del self.chemicals[self._selected_index]
        self._selected_index = None
        self._chem_view = list(range(len(self.chemicals)))
        self._save_library()
        self._refresh_chem_list(preserve_selection=False)
        self._clear_form()
        self._set_chem_details(None)

    # -----------PubChem requests-------------------------------- #
    def _on_fetch_pubchem(self) -> None:
        """
        Uses pubchempy to autofill selected fields:
        - MW
        - SMILES (if empty)
        - InChI (if empty, strip leading 'InChI=')
        - Formula
        - CAS (if empty or 'None')
        - Density (if available)

        Never overwrites the chemical name.
        """

        if not _HAS_PUBCHEM:
            messagebox.showerror(
                "PubChem unavailable",
                "pubchempy is not installed, cannot query PubChem.\n\n"
                "Install with: pip install pubchempy"
            )
            return

        # Prepare possible keys
        cas = self.var_cas.get().strip()
        inchi = self.var_inchi.get().strip()
        smiles = self.var_smiles.get().strip()
        name = self.var_name.get().strip()

        # Determine best lookup key
        query = None
        query_type = None

        if cas and cas.lower() != "none":
            query = cas
            query_type = "name"  # PubChem treats CAS as a kind of "name"
        elif inchi:
            query = inchi
            query_type = "inchi"
        elif smiles:
            query = smiles
            query_type = "smiles"
        elif name:
            query = name
            query_type = "name"

        if not query:
            messagebox.showerror("PubChem", "No usable identifier or name to query PubChem.")
            return

        try:
            if query_type == "inchi":
                res = pcp.get_compounds(query, "inchi")
            elif query_type == "smiles":
                res = pcp.get_compounds(query, "smiles")
            else:
                # CAS or name → PubChem treats them both as 'name' searches
                res = pcp.get_compounds(query, "name")
        except Exception as e:
            messagebox.showerror("PubChem error", f"Error querying PubChem:\n{e}")
            return

        if not res:
            messagebox.showerror("PubChem", f"No data found for '{query}'.")
            return

        c = res[0]  # Take first hit

        # Track what was filled
        filled: list[str] = []
        failed: list[str] = []

        # --- Molecular weight ---
        try:
            if c.molecular_weight:
                self.var_mw_value.set(str(c.molecular_weight))
                filled.append("molecular weight")
            else:
                failed.append("molecular weight")
        except Exception:
            failed.append("molecular weight")

        # --- Molecular formula ---
        try:
            if c.molecular_formula:
                self.var_formula.set(c.molecular_formula)
                filled.append("formula")
            else:
                failed.append("formula")
        except Exception:
            failed.append("formula")

        # --- CAS (if current field is empty or 'None') ---
        try:
            current_cas = self.var_cas.get().strip()
            if not current_cas or current_cas.lower() == "none":
                cas_val = None

                # 1) Try xrefs if available
                xrefs = getattr(c, "xrefs", None)
                if xrefs and isinstance(xrefs, dict):
                    cas_list = xrefs.get("CAS") or xrefs.get("cas")
                    if cas_list:
                        cas_val = cas_list[0]

                # 2) Fallback: look in synonyms for a CAS-like pattern
                if cas_val is None:
                    syns = getattr(c, "synonyms", []) or []
                    cas_pattern = re.compile(r"^\d{2,7}-\d{2}-\d$")
                    for s in syns:
                        s = str(s).strip()
                        if cas_pattern.match(s):
                            cas_val = s
                            break

                if cas_val:
                    self.var_cas.set(str(cas_val))
                    filled.append("CAS")
                else:
                    failed.append("CAS")
        except Exception:
            failed.append("CAS")

        # --- SMILES (fill only if empty) ---
        try:
            if not self.var_smiles.get().strip() and c.canonical_smiles:
                self.var_smiles.set(c.canonical_smiles)
                filled.append("SMILES (empty → filled)")
            elif not c.canonical_smiles:
                failed.append("SMILES")
        except Exception:
            failed.append("SMILES")

        # --- InChI (fill only if empty; strip leading 'InChI=') ---
        try:
            inchi_raw = c.inchi
            if inchi_raw:
                inchi_val = inchi_raw.strip()
                if inchi_val.lower().startswith("inchi="):
                    inchi_val = inchi_val.split("=", 1)[1].strip()

                if not self.var_inchi.get().strip():
                    self.var_inchi.set(inchi_val)
                    filled.append("InChI (empty → filled)")
            else:
                failed.append("InChI")
        except Exception:
            failed.append("InChI")

        # --- Density (rare in PubChem) ---
        try:
            dens_list = [
                p for p in c.to_dict().get("props", [])
                if p.get("urn", {}).get("label") == "Density"
            ]
            if dens_list:
                dens_val = dens_list[0].get("value", {}).get("sval")
                dens_unit = dens_list[0].get("urn", {}).get("unit", "g/mL")
                if dens_val:
                    self.var_density_value.set(str(dens_val))
                    self.var_density_unit.set(str(dens_unit))
                    filled.append("density")
                else:
                    failed.append("density")
            else:
                failed.append("density")
        except Exception:
            failed.append("density")

        # Summary dialog
        msg = []
        if filled:
            msg.append("Filled:\n  - " + "\n  - ".join(filled) + "\n")
        if failed:
            msg.append("Not available:\n  - " + "\n  - ".join(failed))

        messagebox.showinfo("PubChem autofill", "\n".join(msg) if msg else "No properties retrieved.")

    # ------------------------------------------------------------------ #
    # Bulk import
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_library_index(meta: Dict[str, Any]) -> int:
        try:
            return int(meta.get("LibraryIndex", 0))
        except Exception:
            return 0

    @staticmethod
    def _get_inchi(chem: Dict[str, Any]) -> str:
        val = str(chem.get("Inchi", "") or chem.get("inchi", "")).strip()
        # Normalize away leading 'InChI=' if present
        if val.lower().startswith("inchi="):
            val = val.split("=", 1)[1].strip()
        return val

    def _bulk_import_library(self) -> None:
        """
        Bulk import from another library JSON into this one.

        - Determine base library as the one with lowest LibraryIndex.
        - Deduplicate by InChI: if same structure, keep base library record.
        - Resolve name conflicts with dialogs.
        - Ensure unique chemicalID and swissCatNumber, preserving base IDs.
        """
        fname = filedialog.askopenfilename(
            title="Select library JSON to import",
            initialdir=str(self.library_path.parent),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not fname:
            return

        other_path = Path(fname)
        if not other_path.exists():
            messagebox.showerror("Bulk import", f"File does not exist: {other_path}")
            return

        items = _lib_load(other_path)
        if not items:
            messagebox.showerror("Bulk import", f"Library is empty: {other_path}")
            return

        # split meta / chemicals
        first = items[0]
        if isinstance(first, dict) and "libraryName" in first:
            meta2 = first
            chems2 = items[1:]
        else:
            meta2 = {}
            chems2 = items

        # Decide which library is base (lower LibraryIndex)
        idx_current = self._get_library_index(self.meta)
        idx_import = self._get_library_index(meta2)
        if idx_current <= idx_import:
            base_meta = self.meta
            base_chems = self.chemicals
            incoming = chems2
            base_kind = "current"
        else:
            base_meta = meta2
            base_chems = chems2
            incoming = self.chemicals
            base_kind = "imported"

        # Prepare merge containers
        final_meta = dict(base_meta) if base_meta else {}
        final_chems: List[Dict[str, Any]] = [c.copy() for c in base_chems]

        used_ids = {str(c.get("chemicalID", "")).strip() for c in final_chems if c.get("chemicalID") is not None}
        used_ids.discard("")  # remove empty

        used_swiss = {
            str(c.get("swissCatNumber", "")).strip() for c in final_chems if c.get("swissCatNumber") is not None
        }
        used_swiss.discard("")

        # inchi -> idx & name->inchi for conflict detection
        inchi_to_idx: Dict[str, int] = {}
        name_to_inchi: Dict[str, str] = {}
        for i, c in enumerate(final_chems):
            inchi = self._get_inchi(c)
            nm = str(c.get("chemicalName", "")).strip()
            if inchi:
                inchi_to_idx[inchi] = i
            if nm and inchi:
                # if there is already an entry, we keep the first – base library precedence
                name_to_inchi.setdefault(nm, inchi)

        added = 0
        skipped_dup = 0

        for c_orig in incoming:
            c = c_orig.copy()
            inchi = self._get_inchi(c)
            name = str(c.get("chemicalName", "")).strip()

            # Same structure (InChI) already present → deduplicate
            if inchi and inchi in inchi_to_idx:
                existing = final_chems[inchi_to_idx[inchi]]
                existing_name = str(existing.get("chemicalName", "")).strip()
                if existing_name and name and existing_name != name:
                    # Name conflict, same structure → ask for canonical name
                    msg = (
                        "The same InChI appears with two different names:\n\n"
                        f"Base library: {existing_name}\n"
                        f"Imported library: {name}\n\n"
                        "Enter the preferred name for this compound:"
                    )
                    initial = existing_name if base_kind == "imported" else existing_name
                    # (base_kind mainly for documentation; we default to existing name)
                    new_name = simpledialog.askstring(
                        "Name conflict (same structure)", msg, initialvalue=initial
                    )
                    if new_name:
                        existing["chemicalName"] = new_name.strip()
                skipped_dup += 1
                continue

            # Different structure; check name conflicts
            if name:
                if name in name_to_inchi and name_to_inchi[name] != inchi and inchi:
                    # Same name, different structure → need a new name
                    msg = (
                        f"The name '{name}' is already used for a different compound.\n"
                        "Please enter a new unique name for the imported compound:"
                    )
                    new_name = simpledialog.askstring(
                        "Name conflict (different structure)", msg, initialvalue=name + "_2"
                    )
                    if new_name:
                        name = new_name.strip()
                        c["chemicalName"] = name
                if inchi:
                    name_to_inchi[name] = inchi

            # Ensure unique IDs; base library IDs are kept as-is
            cid = str(c.get("chemicalID", "")).strip()
            if not cid or cid in used_ids:
                next_id = 1
                while str(next_id) in used_ids:
                    next_id += 1
                cid = str(next_id)
                c["chemicalID"] = cid
            used_ids.add(cid)

            sc = str(c.get("swissCatNumber", "")).strip()
            if not sc or sc in used_swiss:
                next_sc = 1
                while str(next_sc) in used_swiss:
                    next_sc += 1
                sc = str(next_sc)
                c["swissCatNumber"] = sc
            used_swiss.add(sc)

            final_chems.append(c)
            if inchi:
                inchi_to_idx[inchi] = len(final_chems) - 1
            added += 1

        # Update this library to be the merged one (base + incoming unique)
        self.meta = final_meta
        self.chemicals = final_chems
        self._chem_view = list(range(len(self.chemicals)))
        self._save_library()
        self._refresh_meta_view()
        self._refresh_chem_list(preserve_selection=False)

        messagebox.showinfo(
            "Bulk import",
            (
                f"Imported from: {other_path.name}\n"
                f"Base library: {base_kind} (lower LibraryIndex)\n\n"
                f"New compounds added: {added}\n"
                f"Duplicates by InChI skipped: {skipped_dup}"
            ),
        )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GUI editor for chemical library JSON.")
    parser.add_argument(
        "--lib",
        required=True,
        help="Library JSON file name (relative to data-dir).",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Data directory where the library JSON resides.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    library_path = data_dir / args.lib
    if not library_path.exists():
        raise SystemExit(f"Library file does not exist: {library_path}")

    app = LibraryEditorGUI(library_path=library_path)
    app.mainloop()


if __name__ == "__main__":
    main()
