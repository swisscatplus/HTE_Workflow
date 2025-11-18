#!/usr/bin/env python3
"""
hci_gui_builder.py

Tkinter-based GUI to create an HCI (chemical campaign) JSON file,
using the same spec & builder as `interactive_create_hci` in workflow.py.

- Loads a chemical library (mylene_screening_01.json, etc.)
- Lets you define:
    * Campaign metadata
    * Batch info (plate size, reaction type, etc.)
    * Objective block
    * Global ranges (temperature, concentration, time, plus custom)
    * Groups (name, description, selectionMode, fixed, equivalents)
    * Members per group, searched from the library by name / ID / SwissCat
- Builds the `spec` dict and calls `build_chemical_space_from_spec`.
- Writes the resulting Campaign HCI JSON to a file you choose.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from hte_workflow.paths import DATA_DIR, OUT_DIR, ensure_dirs
from json_handling.hci_file_creator import build_chemical_space_from_spec, load_library

# Optional RDKit / Pillow support for structure rendering
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



# ---------------------------------------------------------------------------
# GUI class
# ---------------------------------------------------------------------------

class HciBuilderGUI(tk.Tk):
    def __init__(self, library_path: Path, out_dir: Path) -> None:
        super().__init__()

        self.title("HCI Campaign Builder")
        self.geometry("1100x700")

        self.library_path = library_path
        self.out_dir = out_dir

        # Load library via existing helper
        self.lib = load_library(library_path)

        # Build a search index from library
        self._lib_records: List[Any] = []
        self._lib_display_rows: List[Tuple[str, str]] = []  # [(display_str, chemicalName), ...]
        self._lib_by_name: Dict[str, Any] = {}  # chemicalName -> record
        self._build_library_index()

        # Holder for currently displayed structure image (to avoid GC)
        self._chem_img = None

        # State for spec being constructed
        self.ranges: Dict[str, Dict[str, Any]] = {}
        self.groups: List[Dict[str, Any]] = []  # each: {groupName, description, selectionMode, fixed, equivalents, members}
        self.current_group_index: int | None = None

        # chemicals not associated with any group (e.g. primary substrate)
        self.global_chemicals: list[str] = []

        # Tk variables for meta
        self.var_campaign_name = tk.StringVar()
        self.var_description = tk.StringVar()
        self.var_objective_txt = tk.StringVar()
        self.var_campaign_class = tk.StringVar(value="Standard Research")
        self.var_type = tk.StringVar(value="optimization")
        self.var_reference = tk.StringVar()

        # Batch
        today = datetime.date.today().strftime("%Y%m%d")
        self.var_batch_id = tk.StringVar(value="0")
        self.var_batch_name = tk.StringVar(value=today)
        self.var_reaction_type = tk.StringVar()
        self.var_reaction_name = tk.StringVar()
        self.var_optimization_type = tk.StringVar()
        self.var_link = tk.StringVar()
        self.var_plate_size = tk.IntVar(value=48)

        # Objective block
        self.var_obj_criteria = tk.StringVar()
        self.var_obj_condition = tk.StringVar()
        self.var_obj_description = tk.StringVar()
        self.var_obj_name = tk.StringVar()

        # Range editor vars
        self.var_range_name = tk.StringVar()
        self.var_range_min = tk.StringVar()
        self.var_range_max = tk.StringVar()
        self.var_range_unit = tk.StringVar()
        self.var_range_step = tk.StringVar()

        # Group editor vars
        self.var_group_name = tk.StringVar()
        self.var_group_desc = tk.StringVar()
        self.var_selection_mode = tk.StringVar(value="one-of")
        self.var_group_fixed = tk.BooleanVar(value=True)
        self.var_eq_min = tk.StringVar(value="1.0")
        self.var_eq_max = tk.StringVar(value="1.0")
        self.var_eq_unit = tk.StringVar(value="eq")
        self.var_eq_step = tk.StringVar()

        # Search vars
        self.var_search = tk.StringVar()

        self._build_ui()

    # ------------------------------------------------------------------ #
    # Library index
    # ------------------------------------------------------------------ #

    def _build_library_index(self) -> None:
        """
        Build a simple search index from the ChemicalLibrary object.
        We assume the loader used in workflow.py:
          lib._by_name: dict[name_lower -> Chemical]
        where Chemical has attributes:
          chemicalName, chemicalID, swissCatNumber, molecularMass, physicalstate, smiles, ...
        """
        try:
            by_name = getattr(self.lib, "_by_name", {})
        except Exception:
            by_name = {}

        self._lib_records = list(by_name.values())
        self._lib_display_rows.clear()
        self._lib_by_name.clear()

        for rec in self._lib_records:
            name = getattr(rec, "chemicalName", "")
            chem_id = getattr(rec, "chemicalID", "")
            swiss = getattr(rec, "swissCatNumber", "")
            label_parts = [name]
            if chem_id:
                label_parts.append(f"[ID: {chem_id}]")
            if swiss:
                label_parts.append(f"(SwissCat: {swiss})")
            display = " ".join(label_parts)
            self._lib_display_rows.append((display, name))
            if name:
                self._lib_by_name[name] = rec

    # ------------------------------------------------------------------ #
    # UI building
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        # The main window itself is a single grid cell
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # --- Scrollable container ---
        container = ttk.Frame(self)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        canvas = tk.Canvas(container, highlightthickness=0)
        vscroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=1, sticky="ns")

        # Inner frame that will contain ALL UI (including the Notebook)
        self.inner = ttk.Frame(canvas)
        self.inner.columnconfigure(0, weight=1)

        # put inner frame into canvas
        window_id = canvas.create_window((0, 0), window=self.inner, anchor="nw")

        # Update scroll region when inner frame grows/shrinks
        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        self.inner.bind("<Configure>", _on_frame_configure)

        # Make inner frame always as wide as the canvas
        def _on_canvas_configure(event):
            canvas.itemconfigure(window_id, width=event.width)

        canvas.bind("<Configure>", _on_canvas_configure)

        # Mouse wheel scrolling (Windows / Mac / Linux)
        def _on_mousewheel(event):
            # Windows / MacOS: event.delta is ±120
            delta = event.delta
            if delta != 0:
                canvas.yview_scroll(int(-delta / 120), "units")

        # Linux uses Button-4/5 events
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
        canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

        # --- Notebook lives inside the scrollable inner frame ---
        nb = ttk.Notebook(self.inner)
        nb.grid(row=0, column=0, sticky="nsew")

        # Tabs
        frame_meta = ttk.Frame(nb, padding=10)
        frame_ranges = ttk.Frame(nb, padding=10)
        frame_groups = ttk.Frame(nb, padding=10)
        frame_preview = ttk.Frame(nb, padding=10)

        nb.add(frame_meta, text="Meta & Batch")
        nb.add(frame_ranges, text="Global ranges")
        nb.add(frame_groups, text="Groups & Members")
        nb.add(frame_preview, text="Preview & Save")

        self._build_tab_meta(frame_meta)
        self._build_tab_ranges(frame_ranges)
        self._build_tab_groups(frame_groups)
        self._build_tab_preview(frame_preview)

    # --------------------- Meta & Batch tab --------------------- #

    def _build_tab_meta(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

        # Campaign meta
        meta_frame = ttk.LabelFrame(parent, text="Campaign metadata", padding=10)
        meta_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))

        def lbl_entry(row: int, text: str, var: tk.StringVar, width: int = 40) -> None:
            ttk.Label(meta_frame, text=text).grid(row=row, column=0, sticky="w")
            ttk.Entry(meta_frame, textvariable=var, width=width).grid(row=row, column=1, sticky="we", pady=2)

        lbl_entry(0, "Campaign name*", self.var_campaign_name)
        lbl_entry(1, "Description", self.var_description)
        lbl_entry(2, "Objective (free text)", self.var_objective_txt)
        lbl_entry(3, "Campaign class", self.var_campaign_class)
        lbl_entry(4, "Type", self.var_type)
        lbl_entry(5, "Reference", self.var_reference)

        # Batch + Objective
        batch_frame = ttk.LabelFrame(parent, text="Batch info & Objective", padding=10)
        batch_frame.grid(row=0, column=1, sticky="nsew", pady=(0, 8))

        batch_frame.columnconfigure(0, weight=1)
        batch_frame.columnconfigure(1, weight=1)

        # Batch info
        ttk.Label(batch_frame, text="Batch ID").grid(row=0, column=0, sticky="w")
        ttk.Entry(batch_frame, textvariable=self.var_batch_id, width=10).grid(row=0, column=1, sticky="w")

        ttk.Label(batch_frame, text="Batch name").grid(row=1, column=0, sticky="w")
        ttk.Entry(batch_frame, textvariable=self.var_batch_name, width=16).grid(row=1, column=1, sticky="w")

        ttk.Label(batch_frame, text="Reaction type").grid(row=2, column=0, sticky="w")
        ttk.Entry(batch_frame, textvariable=self.var_reaction_type).grid(row=2, column=1, sticky="we")

        ttk.Label(batch_frame, text="Reaction name").grid(row=3, column=0, sticky="w")
        ttk.Entry(batch_frame, textvariable=self.var_reaction_name).grid(row=3, column=1, sticky="we")

        ttk.Label(batch_frame, text="Optimization type").grid(row=4, column=0, sticky="w")
        ttk.Entry(batch_frame, textvariable=self.var_optimization_type).grid(row=4, column=1, sticky="we")

        ttk.Label(batch_frame, text="Link").grid(row=5, column=0, sticky="w")
        ttk.Entry(batch_frame, textvariable=self.var_link).grid(row=5, column=1, sticky="we")

        ttk.Label(batch_frame, text="Plate size").grid(row=6, column=0, sticky="w")
        spin = tk.Spinbox(
            batch_frame,
            from_=1,
            to=384,
            textvariable=self.var_plate_size,
            width=6,
        )
        spin.grid(row=6, column=1, sticky="w")

        # Objective block
        sep = ttk.Separator(batch_frame, orient="horizontal")
        sep.grid(row=7, column=0, columnspan=2, sticky="ew", pady=6)

        ttk.Label(batch_frame, text="Objective criteria").grid(row=8, column=0, sticky="w")
        ttk.Entry(batch_frame, textvariable=self.var_obj_criteria).grid(row=8, column=1, sticky="we")

        ttk.Label(batch_frame, text="Objective condition").grid(row=9, column=0, sticky="w")
        ttk.Entry(batch_frame, textvariable=self.var_obj_condition).grid(row=9, column=1, sticky="we")

        ttk.Label(batch_frame, text="Objective description").grid(row=10, column=0, sticky="w")
        ttk.Entry(batch_frame, textvariable=self.var_obj_description).grid(row=10, column=1, sticky="we")

        ttk.Label(batch_frame, text="Objective name").grid(row=11, column=0, sticky="w")
        ttk.Entry(batch_frame, textvariable=self.var_obj_name).grid(row=11, column=1, sticky="we")

    # --------------------- Ranges tab --------------------------- #

    def _build_tab_ranges(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        left = ttk.Frame(parent)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        # Table of ranges
        columns = ("name", "min", "max", "unit", "step")
        self.range_tree = ttk.Treeview(left, columns=columns, show="headings", height=10)
        for col in columns:
            self.range_tree.heading(col, text=col)
            self.range_tree.column(col, width=80, anchor="center")
        self.range_tree.grid(row=0, column=0, sticky="nsew")

        self.range_tree.bind("<<TreeviewSelect>>", self._on_range_selected)

        # Buttons under table
        btn_frame = ttk.Frame(left)
        btn_frame.grid(row=1, column=0, sticky="ew", pady=4)
        ttk.Button(btn_frame, text="Add / Update", command=self._on_range_add_update).grid(row=0, column=0, padx=2)
        ttk.Button(btn_frame, text="Delete", command=self._on_range_delete).grid(row=0, column=1, padx=2)

        # Editor on right
        right = ttk.LabelFrame(parent, text="Edit range", padding=10)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.columnconfigure(1, weight=1)

        def row(label: str, var: tk.StringVar, r: int) -> None:
            ttk.Label(right, text=label).grid(row=r, column=0, sticky="w")
            ttk.Entry(right, textvariable=var).grid(row=r, column=1, sticky="we", pady=2)

        row("Name", self.var_range_name, 0)
        row("min", self.var_range_min, 1)
        row("max", self.var_range_max, 2)
        row("unit", self.var_range_unit, 3)
        row("step (optional)", self.var_range_step, 4)

        ttk.Label(
            right,
            text="Note: 'temperature' and 'time' are fixed per plate.\n"
                 "Other globals (e.g. 'concentration') may vary per well.",
            foreground="gray",
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # Pre-populate with the common ones, similar to interactive_create_hci
        for nm, unit in (("temperature", "C"), ("concentration", "M"), ("time", "min")):
            if nm not in self.ranges:
                self.ranges[nm] = {"min": 0.0, "max": 0.0, "unit": unit}
        self._refresh_range_table()

    def _refresh_range_table(self) -> None:
        self.range_tree.delete(*self.range_tree.get_children())
        for name, r in self.ranges.items():
            self.range_tree.insert(
                "",
                "end",
                iid=name,
                values=(
                    name,
                    r.get("min", ""),
                    r.get("max", ""),
                    r.get("unit", ""),
                    r.get("step", ""),
                ),
            )

    def _on_range_selected(self, event=None) -> None:
        sel = self.range_tree.selection()
        if not sel:
            return
        name = sel[0]
        r = self.ranges.get(name, {})
        self.var_range_name.set(name)
        self.var_range_min.set(str(r.get("min", "")))
        self.var_range_max.set(str(r.get("max", "")))
        self.var_range_unit.set(str(r.get("unit", "")))
        self.var_range_step.set(str(r.get("step", "")))

    def _on_range_add_update(self) -> None:
        nm = self.var_range_name.get().strip()
        if not nm:
            messagebox.showerror("Range", "Name is required.")
            return
        try:
            vmin = float(self.var_range_min.get()) if self.var_range_min.get() else 0.0
            vmax = float(self.var_range_max.get()) if self.var_range_max.get() else 0.0
        except ValueError:
            messagebox.showerror("Range", "min/max must be numeric.")
            return

        r: Dict[str, Any] = {
            "min": vmin,
            "max": vmax,
            "unit": self.var_range_unit.get().strip(),
        }
        step_txt = self.var_range_step.get().strip()
        if step_txt:
            try:
                step_val = float(step_txt)
                r["step"] = step_val
            except ValueError:
                messagebox.showerror("Range", "step must be numeric.")
                return

        self.ranges[nm] = r
        self._refresh_range_table()

    def _on_range_delete(self) -> None:
        sel = self.range_tree.selection()
        if not sel:
            return
        for iid in sel:
            self.ranges.pop(iid, None)
        self._refresh_range_table()

    # --------------------- Groups tab --------------------------- #

    def _build_tab_groups(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(2, weight=1)
        parent.rowconfigure(0, weight=1)

        # Left: group list + buttons
        left = ttk.Frame(parent)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 8))
        left.rowconfigure(1, weight=1)

        ttk.Label(left, text="Groups").grid(row=0, column=0, sticky="w")
        self.group_listbox = tk.Listbox(left, height=12)
        self.group_listbox.grid(row=1, column=0, sticky="nsew")

        self.group_listbox.bind("<<ListboxSelect>>", self._on_group_selected)

        btn_frame = ttk.Frame(left)
        btn_frame.grid(row=2, column=0, sticky="ew", pady=4)
        ttk.Button(btn_frame, text="Add group", command=self._on_group_add).grid(row=0, column=0, padx=2)
        ttk.Button(btn_frame, text="Delete group", command=self._on_group_delete).grid(row=0, column=1, padx=2)

        # Middle: group metadata
        mid = ttk.LabelFrame(parent, text="Group metadata", padding=8)
        mid.grid(row=0, column=1, sticky="nsew")
        mid.columnconfigure(1, weight=1)

        ttk.Label(mid, text="Group name").grid(row=0, column=0, sticky="w")
        ttk.Entry(mid, textvariable=self.var_group_name).grid(row=0, column=1, sticky="we", pady=2)

        ttk.Label(mid, text="Description").grid(row=1, column=0, sticky="w")
        ttk.Entry(mid, textvariable=self.var_group_desc).grid(row=1, column=1, sticky="we", pady=2)

        ttk.Label(mid, text="Selection mode").grid(row=2, column=0, sticky="w")
        sel_cb = ttk.Combobox(mid, textvariable=self.var_selection_mode, values=["one-of", "any", "at-least-one"])
        sel_cb.grid(row=2, column=1, sticky="we", pady=2)

        ttk.Checkbutton(mid, text="Fixed (always included)", variable=self.var_group_fixed).grid(
            row=3, column=0, columnspan=2, sticky="w"
        )

        ttk.Label(mid, text="Eq min").grid(row=4, column=0, sticky="w")
        ttk.Entry(mid, textvariable=self.var_eq_min, width=8).grid(row=4, column=1, sticky="w")

        ttk.Label(mid, text="Eq max").grid(row=5, column=0, sticky="w")
        ttk.Entry(mid, textvariable=self.var_eq_max, width=8).grid(row=5, column=1, sticky="w")

        ttk.Label(mid, text="Eq unit").grid(row=6, column=0, sticky="w")
        ttk.Entry(mid, textvariable=self.var_eq_unit, width=8).grid(row=6, column=1, sticky="w")

        ttk.Label(mid, text="Eq step (optional)").grid(row=7, column=0, sticky="w")
        ttk.Entry(mid, textvariable=self.var_eq_step, width=8).grid(row=7, column=1, sticky="w")

        ttk.Button(mid, text="Apply meta to group", command=self._on_group_apply_meta).grid(
            row=8, column=0, columnspan=2, pady=6
        )

        # Right: members + library search
        right = ttk.LabelFrame(parent, text="Group members", padding=8)
        right.grid(row=0, column=2, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(4, weight=1)

        # Show library name
        ttk.Label(
            right,
            text=f"Library: {self.library_path.name}",
            foreground="gray",
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(right, text="Search library (name / ID / SwissCat)").grid(row=1, column=0, sticky="w")
        search_entry = ttk.Entry(right, textvariable=self.var_search)
        search_entry.grid(row=2, column=0, sticky="ew", pady=2)
        ttk.Button(right, text="Search", command=self._on_search_library).grid(row=2, column=1, padx=4)

        ttk.Label(right, text="Search results").grid(row=3, column=0, sticky="w")
        self.search_listbox = tk.Listbox(right, height=8)
        self.search_listbox.grid(row=4, column=0, sticky="nsew")

        # Update details when a search result is selected
        self.search_listbox.bind("<<ListboxSelect>>", self._on_search_select)

        # Details panel: textual info + optional structure
        details_frame = ttk.LabelFrame(right, text="Chemical details", padding=5)
        details_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        details_frame.columnconfigure(0, weight=1)

        self.chem_info_label = ttk.Label(details_frame, text="", justify="left")
        self.chem_info_label.grid(row=0, column=0, sticky="w")

        # Image label for RDKit drawing (if available)
        self.chem_img_label = ttk.Label(details_frame)
        self.chem_img_label.grid(row=0, column=1, sticky="e", padx=8)


        # Members
        ttk.Label(right, text="Members in group").grid(row=6, column=0, sticky="w", pady=(6, 0))
        self.members_listbox = tk.Listbox(right, height=8)
        self.members_listbox.grid(row=7, column=0, sticky="nsew")
        self.members_listbox.bind("<<ListboxSelect>>", self._on_member_select)


        btn_mframe = ttk.Frame(right)
        btn_mframe.grid(row=8, column=0, sticky="ew", pady=4)
        ttk.Button(btn_mframe, text="Add from search", command=self._on_member_add_from_search).grid(
            row=0, column=0, padx=2
        )
        ttk.Button(btn_mframe, text="Remove selected", command=self._on_member_remove).grid(
            row=0, column=1, padx=2
        )

        # Ungrouped / global chemicals
        ttk.Label(right, text="Ungrouped chemicals").grid(row=9, column=0, sticky="w", pady=(6, 0))
        self.global_listbox = tk.Listbox(right, height=6)
        self.global_listbox.grid(row=10, column=0, sticky="nsew")
        self.global_listbox.bind("<<ListboxSelect>>", self._on_global_select)


        btn_gframe = ttk.Frame(right)
        btn_gframe.grid(row=11, column=0, sticky="ew", pady=4)
        ttk.Button(btn_gframe, text="Add as global chem", command=self._on_global_add_from_search).grid(
            row=0, column=0, padx=2
        )
        ttk.Button(btn_gframe, text="Remove global chem", command=self._on_global_remove).grid(
            row=0, column=1, padx=2
        )

    def _refresh_group_list(self) -> None:
        self.group_listbox.delete(0, tk.END)
        for g in self.groups:
            self.group_listbox.insert(tk.END, g.get("groupName", "<unnamed>"))

    def _on_group_add(self) -> None:
        # Create empty group and select it
        new = {
            "groupName": f"group_{len(self.groups)+1}",
            "description": "",
            "selectionMode": "one-of",
            "equivalents": {"min": 1.0, "max": 1.0, "unit": "eq"},
            "fixed": True,
            "members": [],
        }
        self.groups.append(new)
        self._refresh_group_list()
        idx = len(self.groups) - 1
        self.group_listbox.select_clear(0, tk.END)
        self.group_listbox.select_set(idx)
        self.group_listbox.event_generate("<<ListboxSelect>>")

    def _on_group_delete(self) -> None:
        sel = self.group_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self.groups):
            del self.groups[idx]
        self.current_group_index = None
        self._refresh_group_list()
        self._clear_group_editor()

    def _on_group_selected(self, event=None) -> None:
        sel = self.group_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if not (0 <= idx < len(self.groups)):
            return
        self.current_group_index = idx
        g = self.groups[idx]
        self.var_group_name.set(g.get("groupName", ""))
        self.var_group_desc.set(g.get("description", ""))
        self.var_selection_mode.set(g.get("selectionMode", "one-of"))
        self.var_group_fixed.set(bool(g.get("fixed", True)))

        eq = g.get("equivalents", {})
        self.var_eq_min.set(str(eq.get("min", "")))
        self.var_eq_max.set(str(eq.get("max", "")))
        self.var_eq_unit.set(str(eq.get("unit", "")))
        self.var_eq_step.set(str(eq.get("step", "")))

        # members
        self.members_listbox.delete(0, tk.END)
        for m in g.get("members", []):
            nm = m.get("name", "")
            if nm:
                self.members_listbox.insert(tk.END, nm)

    def _clear_group_editor(self) -> None:
        self.var_group_name.set("")
        self.var_group_desc.set("")
        self.var_selection_mode.set("one-of")
        self.var_group_fixed.set(True)
        self.var_eq_min.set("1.0")
        self.var_eq_max.set("1.0")
        self.var_eq_unit.set("eq")
        self.var_eq_step.set("")
        self.members_listbox.delete(0, tk.END)

    def _on_group_apply_meta(self) -> None:
        if self.current_group_index is None or not (0 <= self.current_group_index < len(self.groups)):
            messagebox.showerror("Group", "No group selected.")
            return
        try:
            eq_min = float(self.var_eq_min.get()) if self.var_eq_min.get() else 1.0
            eq_max = float(self.var_eq_max.get()) if self.var_eq_max.get() else eq_min
        except ValueError:
            messagebox.showerror("Group", "Eq min/max must be numeric.")
            return

        eq: Dict[str, Any] = {
            "min": eq_min,
            "max": eq_max,
            "unit": self.var_eq_unit.get().strip() or "eq",
        }
        if self.var_eq_step.get().strip():
            try:
                eq["step"] = float(self.var_eq_step.get().strip())
            except ValueError:
                messagebox.showerror("Group", "Eq step must be numeric.")
                return

        g = self.groups[self.current_group_index]
        g["groupName"] = self.var_group_name.get().strip() or g["groupName"]
        g["description"] = self.var_group_desc.get().strip()
        g["selectionMode"] = self.var_selection_mode.get()
        g["fixed"] = bool(self.var_group_fixed.get())
        g["equivalents"] = eq

        self._refresh_group_list()

    def _on_search_library(self) -> None:
        query = self.var_search.get().strip().lower()
        self.search_listbox.delete(0, tk.END)
        if not query:
            # show first N
            for disp, _name in self._lib_display_rows[:50]:
                self.search_listbox.insert(tk.END, disp)
            return

        for disp, name in self._lib_display_rows:
            if query in disp.lower():
                self.search_listbox.insert(tk.END, disp)

    def _on_member_add_from_search(self) -> None:
        if self.current_group_index is None or not (0 <= self.current_group_index < len(self.groups)):
            messagebox.showerror("Group members", "No group selected.")
            return
        sel = self.search_listbox.curselection()
        if not sel:
            messagebox.showerror("Group members", "No search result selected.")
            return

        disp_row = self.search_listbox.get(sel[0])
        # Find canonical chemicalName for this display row
        selected_name = None
        for disp, nm in self._lib_display_rows:
            if disp == disp_row:
                selected_name = nm
                break
        if not selected_name:
            messagebox.showerror("Group members", "Could not resolve chemical name.")
            return

        g = self.groups[self.current_group_index]
        members = g.get("members", [])
        # avoid duplicates
        if any(m.get("name") == selected_name for m in members):
            return
        members.append({"name": selected_name})
        g["members"] = members

        self.members_listbox.insert(tk.END, selected_name)

    def _on_search_select(self, event=None) -> None:
        """
        When a search result is selected, show detailed info + structure.
        """
        sel = self.search_listbox.curselection()
        if not sel:
            self._set_chem_details(None)
            return

        disp_row = self.search_listbox.get(sel[0])
        # Resolve display string -> chemicalName
        selected_name = None
        for disp, nm in self._lib_display_rows:
            if disp == disp_row:
                selected_name = nm
                break

        if not selected_name:
            self._set_chem_details(None)
            return

        rec = self._lib_by_name.get(selected_name)
        self._set_chem_details(rec)

    def _on_member_select(self, event=None) -> None:
        """
        When a group member is clicked, show its chemical details.
        """
        if not hasattr(self, "members_listbox"):
            return
        sel = self.members_listbox.curselection()
        if not sel:
            self._set_chem_details(None)
            return

        name = self.members_listbox.get(sel[0]).strip()
        rec = self._lib_by_name.get(name)
        self._set_chem_details(rec)

    def _on_global_select(self, event=None) -> None:
        """
        When a global (ungrouped) chemical is clicked, show its chemical details.
        """
        if not hasattr(self, "global_listbox"):
            return
        sel = self.global_listbox.curselection()
        if not sel:
            self._set_chem_details(None)
            return

        name = self.global_listbox.get(sel[0]).strip()
        rec = self._lib_by_name.get(name)
        self._set_chem_details(rec)


    def _set_chem_details(self, rec: Any | None) -> None:
        """
        Update the details pane with metadata + optional RDKit structure.
        """
        if rec is None:
            # clear
            if hasattr(self, "chem_info_label"):
                self.chem_info_label.config(text="")
            if hasattr(self, "chem_img_label"):
                self.chem_img_label.config(image="")
            self._chem_img = None
            return

        name = getattr(rec, "chemicalName", "")
        chem_id = getattr(rec, "chemicalID", "")
        swiss = getattr(rec, "swissCatNumber", "")
        state = getattr(rec, "physicalstate", "")

        # molecularMass may be a Quantity dataclass or dict
        mm = getattr(rec, "molecularMass", None)
        mass_val = None
        mass_unit = ""
        if mm is not None:
            # dataclass-like
            mass_val = getattr(mm, "value", None)
            mass_unit = getattr(mm, "unit", "") or ""
            # or JSON-dict-like
            if mass_val is None and isinstance(mm, dict):
                mass_val = mm.get("value")
                mass_unit = mm.get("unit", "")

        lines = []
        if name:
            lines.append(f"Name: {name}")
        if chem_id:
            lines.append(f"ID: {chem_id}")
        if swiss:
            lines.append(f"SwissCat: {swiss}")
        if mass_val is not None:
            lines.append(f"MW: {mass_val} {mass_unit}".strip())
        if state:
            lines.append(f"Physical state: {state}")

        info_txt = "\n".join(lines) if lines else "No metadata available."
        if hasattr(self, "chem_info_label"):
            self.chem_info_label.config(text=info_txt)

        # Draw structure if RDKit is available and SMILES present
        if _HAS_RDKIT and hasattr(self, "chem_img_label"):
            smiles = getattr(rec, "smiles", "")
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
                img = Draw.MolToImage(mol, size=(200, 200))
                # Convert to Tk image and keep a reference
                self._chem_img = ImageTk.PhotoImage(img)
                self.chem_img_label.config(image=self._chem_img)
            except Exception:
                # Fail silently, keep text info
                self.chem_img_label.config(image="")
                self._chem_img = None
        else:
            # No RDKit/Pillow -> no image
            if hasattr(self, "chem_img_label"):
                self.chem_img_label.config(image="")
            self._chem_img = None


    def _on_member_remove(self) -> None:
        if self.current_group_index is None or not (0 <= self.current_group_index < len(self.groups)):
            return
        sel = self.members_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        g = self.groups[self.current_group_index]
        members = g.get("members", [])
        if 0 <= idx < len(members):
            del members[idx]
        g["members"] = members
        self.members_listbox.delete(idx)

    def _on_global_add_from_search(self) -> None:
        """
        Add selected library hit as an ungrouped / global chemical.
        """
        sel = self.search_listbox.curselection()
        if not sel:
            messagebox.showerror("Global chemicals", "No search result selected.")
            return

        disp_row = self.search_listbox.get(sel[0])
        selected_name = None
        for disp, nm in self._lib_display_rows:
            if disp == disp_row:
                selected_name = nm
                break
        if not selected_name:
            messagebox.showerror("Global chemicals", "Could not resolve chemical name.")
            return

        if selected_name in self.global_chemicals:
            return

        self.global_chemicals.append(selected_name)
        self.global_listbox.insert(tk.END, selected_name)

    def _on_global_remove(self) -> None:
        """
        Remove selected ungrouped / global chemical.
        """
        sel = self.global_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self.global_chemicals):
            del self.global_chemicals[idx]
        self.global_listbox.delete(idx)


    # --------------------- Preview & Save tab ------------------- #

    def _build_tab_preview(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        self.preview_text = tk.Text(parent, wrap="none")
        self.preview_text.grid(row=0, column=0, sticky="nsew")

        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=1, column=0, sticky="ew", pady=4)
        ttk.Button(btn_frame, text="Refresh preview", command=self._on_refresh_preview).grid(row=0, column=0, padx=2)
        ttk.Button(btn_frame, text="Save HCI...", command=self._on_save_hci).grid(row=0, column=1, padx=2)

    def _build_spec_dict(self) -> Dict[str, Any]:
        # campaign meta
        campaignName = self.var_campaign_name.get().strip()
        if not campaignName:
            raise ValueError("Campaign name is required.")

        description = self.var_description.get().strip()
        objective_txt = self.var_objective_txt.get().strip()
        campaignClass = self.var_campaign_class.get().strip() or "Standard Research"
        type_txt = self.var_type.get().strip() or "optimization"
        reference = self.var_reference.get().strip()

        # REQUIRED fields
        reaction_type = self.var_reaction_type.get().strip()
        if not reaction_type:
            raise ValueError("Reaction type is required.")

        reaction_name = self.var_reaction_name.get().strip()
        if not reaction_name:
            raise ValueError("Reaction name is required.")

        objective_name = self.var_obj_name.get().strip()
        if not objective_name:
            raise ValueError("Objective name is required.")

        hasBatch = {
            "batchID": self.var_batch_id.get().strip() or "0",
            "batchName": self.var_batch_name.get().strip() or datetime.date.today().strftime("%Y%m%d"),
            "reactionType": reaction_type,
            "reactionName": reaction_name,
            "optimizationType": self.var_optimization_type.get().strip(),
            "link": self.var_link.get().strip(),
            "plate_size": int(self.var_plate_size.get() or 48),
        }

        hasObjective = {
            "criteria": self.var_obj_criteria.get().strip(),
            "condition": self.var_obj_condition.get().strip(),
            "description": self.var_obj_description.get().strip(),
            "objectiveName": objective_name,
        }

        # ranges already in self.ranges
        ranges = self.ranges

        # groups: ensure structure: { .., "members":[{"name":..},..] }
        groups = self.groups

        # chemicals: deduplicate all member names + global chemicals
        chem_names: set[str] = set()
        for g in groups:
            for m in g.get("members", []):
                nm = str(m.get("name", "")).strip()
                if nm:
                    chem_names.add(nm)

        # add ungrouped / global chemicals (will be defined below)
        for nm in self.global_chemicals:
            nm = nm.strip()
            if nm:
                chem_names.add(nm)

        chemicals: List[Dict[str, Any]] = [{"chemicalName": nm} for nm in sorted(chem_names)]

        spec: Dict[str, Any] = {
            "campaignName": campaignName,
            "description": description,
            "objective": objective_txt,
            "campaignClass": campaignClass,
            "type": type_txt,
            "reference": reference,
            "hasBatch": hasBatch,
            "hasObjective": hasObjective,
            "ranges": ranges,
            "groups": groups,
            "chemicals": chemicals,
        }
        return spec

    def _on_refresh_preview(self) -> None:
        try:
            spec = self._build_spec_dict()
        except Exception as e:
            messagebox.showerror("Preview", f"Error building spec: {e}")
            return
        self.preview_text.delete("1.0", tk.END)
        self.preview_text.insert("1.0", json.dumps(spec, indent=2, ensure_ascii=False))

    def _on_save_hci(self) -> None:
        try:
            spec = self._build_spec_dict()
        except Exception as e:
            messagebox.showerror("Save HCI", f"Error building spec: {e}")
            return

        # Validate via builder
        try:
            campaign = build_chemical_space_from_spec(spec, self.lib)
            hci_json = campaign.to_json()
        except Exception as e:
            messagebox.showerror(
                "Save HCI",
                f"Error building Campaign with build_chemical_space_from_spec:\n{e}",
            )
            return

        campaignName = spec["campaignName"]
        batchID = spec["hasBatch"]["batchID"]
        default_name = f"{campaignName}_{batchID}_hci.json"

        self.out_dir.mkdir(parents=True, exist_ok=True)
        initial = str(self.out_dir / default_name)

        fname = filedialog.asksaveasfilename(
            title="Save HCI JSON",
            defaultextension=".json",
            initialfile=default_name,
            initialdir=str(self.out_dir),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not fname:
            return

        out_path = Path(fname)
        out_path.write_text(json.dumps(hci_json, indent=2, ensure_ascii=False), encoding="utf-8")

        messagebox.showinfo("Save HCI", f"HCI written to:\n{out_path}")
        # For CLI / workflow integration:
        # - a human-friendly message
        # - and a machine-readable line that your workflow can parse
        abs_path = out_path.resolve()
        print(f"Wrote HCI file to: {abs_path}")
        print(f"HCI_FILE_PATH={abs_path}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GUI HCI file creator")
    parser.add_argument(
        "--library-path",
        required=True,
        help="Path to the chemical library JSON (relative to data-dir by default)",
    )
    parser.add_argument(
        "--data-dir",
        default=str(DATA_DIR),
        help="Directory for data files (default: DATA_DIR)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Directory for output files (default: OUT_DIR)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    library_path = data_dir / args.library_path

    if not library_path.exists():
        raise SystemExit(f"Library file {library_path} does not exist.")

    app = HciBuilderGUI(library_path=library_path, out_dir=out_dir)
    app.mainloop()


if __name__ == "__main__":
    main()
