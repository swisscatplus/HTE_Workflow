#!/usr/bin/env python3
"""
manual_plate_gui.py

Graphical manual plate builder:

- Load HCI
- Show a plate layout (buttons for Experiment 1..N)
- Show global parameters (from hasRanges)
- For the selected experiment, let the user pick:
    - member per group (solvent, base, substrate, ...)
    - equivalents (within min/max range from HCI)
- Build BO-style `selections` and write synthesis.json via synthesis_writer.

Usage
-----

python -m json_handling.manual_plate_gui \
    --hci-file Amidation_0_hci.json \
    --out 20251113_Selective_Amidation_0_synthesis.json \
    --limiting-name "OMe-Lys(NPC)-NPC" \
    --well-volume-uL 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import tkinter as tk
from tkinter import ttk, messagebox

from hte_workflow.manual_plate_builder import (
    load_hci,
    hci_plate_size,
    hci_globals_from_ranges,
    collect_group_catalog,
    seq_labels,
)
from json_handling.library_and_hci_adapter import hci_to_optimizer_dict
from json_handling.synthesis_writer import write_synthesis_json
from hte_workflow.paths import DATA_DIR, OUT_DIR


# ---- small helpers ----

def plate_dims_from_size(plate_size: int) -> Tuple[int, int]:
    """
    Guess a reasonably compact grid for plate_size wells.
    96 -> 8x12, 48 -> 6x8, 24 -> 4x6, 18 -> 3x6, etc.
    """
    if plate_size == 96:
        return 8, 12
    if plate_size == 48:
        return 6, 8
    if plate_size == 24:
        return 4, 6
    if plate_size == 18:
        return 3, 6
    if plate_size == 12:
        return 3, 4
    if plate_size == 6:
        return 2, 3

    # fallback: square-ish
    import math
    side = int(math.ceil(math.sqrt(plate_size)))
    rows = side
    cols = side
    # we might have some empty trailing buttons, that's fine
    return rows, cols


class ManualPlateGUI(tk.Tk):
    def __init__(
        self,
        hci: Dict[str, Any],
        hci_path: Path,
        plate_size: int,
        base_globals: Dict[str, float],
        variable_defs: Dict[str, Tuple[float, float, str]],
        out_path: Path,
        limiting_name: str | None,
        well_volume_uL: float,
    ) -> None:
        super().__init__()

        self.title("Manual Plate Builder")
        self.hci = hci
        self.hci_path = hci_path
        self.plate_size = plate_size
        self.base_globals = base_globals
        self.variable_defs = variable_defs  # {name: (min, max, unit)}
        self.out_path = out_path
        self.limiting_name = limiting_name
        self.well_volume_uL = well_volume_uL

        # group catalog
        self.group_names, self.members_by_group, self.eq_ranges = collect_group_catalog(hci)

        # color mode: "completion" or a group name
        self.color_mode_var = tk.StringVar(value="completion")

        # experiments: "1", "2", ...
        self.exp_labels = seq_labels(plate_size)  # ["1", "2", ...]
        self.current_exp_index = 0  # index in exp_labels list

        # selections per experiment: list[dict[groupName -> {"member": str, "equivalents": float}]]
        self.exp_groups: List[Dict[str, Dict[str, Any]]] = [
            {} for _ in range(self.plate_size)
        ]

        # variable global entries (stringvars)
        self.var_global_vars: Dict[str, tk.StringVar] = {}

        # UI containers
        self.plate_buttons: List[tk.Button] = []

        self._build_ui()

    # ---------- UI construction ----------

    def _build_ui(self) -> None:
        # overall layout: left plate, right config
        main_frame = ttk.Frame(self, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # configure columns
        main_frame.columnconfigure(0, weight=0)  # plate
        main_frame.columnconfigure(1, weight=1)  # config

        # Plate frame
        plate_frame = ttk.LabelFrame(main_frame, text="Plate Layout", padding=5)
        plate_frame.grid(row=0, column=0, sticky="nsw", padx=(0, 8))

        # Config frame (globals + groups)
        config_frame = ttk.Frame(main_frame)
        config_frame.grid(row=0, column=1, sticky="nsew")
        config_frame.columnconfigure(0, weight=1)
        config_frame.rowconfigure(1, weight=1)

        # Globals frame
        globals_frame = ttk.LabelFrame(config_frame, text="Global Parameters", padding=5)
        globals_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        # Groups frame
        groups_frame = ttk.LabelFrame(config_frame, text="Groups for current experiment", padding=5)
        groups_frame.grid(row=1, column=0, sticky="nsew")
        groups_frame.columnconfigure(1, weight=1)

        # Batch edit frame (experiment multi-selection)
        batch_frame = ttk.LabelFrame(config_frame, text="Batch edit", padding=5)
        batch_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        batch_frame.columnconfigure(0, weight=1)

        self.batch_listbox = tk.Listbox(
            batch_frame,
            selectmode="extended",
            exportselection=False,
            height=6,
        )
        for lbl in self.exp_labels:
            self.batch_listbox.insert("end", lbl)
        self.batch_listbox.grid(row=0, column=0, sticky="ew")
        self.batch_listbox.bind("<Control-a>", self._batch_select_all)
        self.batch_listbox.bind("<Command-a>", self._batch_select_all)


        # Bottom buttons
        bottom_frame = ttk.Frame(config_frame)
        bottom_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.columnconfigure(1, weight=1)

        # Color mode selector
        color_frame = ttk.LabelFrame(main_frame, text="Color mode", padding=5)
        color_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(color_frame, text="Color wells by:").grid(row=0, column=0, sticky="w")

        mode_cb = ttk.Combobox(
            color_frame,
            textvariable=self.color_mode_var,
            values=["completion"] + self.group_names,
            state="readonly",
            width=20,
        )
        mode_cb.grid(row=0, column=1, sticky="w", padx=(4, 0))
        mode_cb.bind("<<ComboboxSelected>>", lambda e: self._update_plate_colors())


        # Build plate grid
        self._build_plate(plate_frame)

        # Build globals panel
        self._build_globals(globals_frame)

        # Build groups panel
        self._build_groups_panel(groups_frame)

        # bottom buttons
        save_btn = ttk.Button(bottom_frame, text="Save synthesis.json", command=self.on_save)
        save_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        quit_btn = ttk.Button(bottom_frame, text="Quit without saving", command=self.on_quit)
        quit_btn.grid(row=0, column=1, sticky="ew", padx=(4, 0))
        copy_btn = ttk.Button(bottom_frame, text="Copy pattern…", command=self.open_copy_dialog)
        copy_btn.grid(row=1, column=0, columnspan=1, sticky="ew", pady=(6, 0))

        apply_batch_btn = ttk.Button(
            bottom_frame,
            text="Apply current settings to selected",
            command=self.apply_current_to_selected,
        )
        apply_batch_btn.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(4, 0))

        # ensure first experiment is selected
        self.select_experiment(0)
        self._update_plate_colors()

    def _batch_select_all(self, event=None):
        self.batch_listbox.selection_set(0, "end")
        return "break"

    def _build_plate(self, parent: ttk.Frame) -> None:
        rows, cols = plate_dims_from_size(self.plate_size)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= self.plate_size:
                    # empty cell
                    spacer = ttk.Label(parent, text=" ", width=4)
                    spacer.grid(row=r, column=c, padx=2, pady=2)
                    continue
                exp_label = self.exp_labels[idx]
                btn = tk.Button(
                    parent,
                    text=exp_label,
                    width=4,
                    relief="raised",
                    command=lambda i=idx: self.select_experiment(i),
                )
                btn.grid(row=r, column=c, padx=2, pady=2)
                self.plate_buttons.append(btn)
                idx += 1

    def _build_globals(self, parent: ttk.LabelFrame) -> None:
        """
        Show all open-range globals (where min != max) from hasRanges,
        and show constants as labels.
        """
        row = 0
        # constants
        if self.base_globals:
            ttk.Label(parent, text="Fixed (from HCI):").grid(row=row, column=0, sticky="w")
            row += 1
            for name, val in self.base_globals.items():
                ttk.Label(parent, text=f"  {name} = {val}").grid(row=row, column=0, columnspan=2, sticky="w")
                row += 1

        # variable globals
        if self.variable_defs:
            ttk.Label(parent, text="Editable:").grid(row=row, column=0, sticky="w", pady=(4, 0))
            row += 1
            for name, (vmin, vmax, unit) in self.variable_defs.items():
                lbl = ttk.Label(parent, text=f"{name} [{vmin}..{vmax} {unit}]:")
                lbl.grid(row=row, column=0, sticky="w")
                var = tk.StringVar(value=str(vmin))  # default to min
                ent = ttk.Entry(parent, textvariable=var, width=12)
                ent.grid(row=row, column=1, sticky="w", padx=(4, 0))
                self.var_global_vars[name] = var
                row += 1

    def _build_groups_panel(self, parent: ttk.LabelFrame) -> None:
        """
        Create the widgets for group selection. We'll re-bind their values
        when the experiment changes.
        """
        self.group_widgets: Dict[str, Dict[str, Any]] = {}  # groupName -> {'member': Combobox, 'eq': Entry, 'eq_label': Label}

        # header row: labels
        ttk.Label(parent, text="Group").grid(row=0, column=0, sticky="w")
        ttk.Label(parent, text="Member").grid(row=0, column=1, sticky="w")
        ttk.Label(parent, text="Equivalents").grid(row=0, column=2, sticky="w")
        ttk.Label(parent, text="Batch").grid(row=0, column=3, sticky="w")

        for i, gname in enumerate(self.group_names, start=1):
            ttk.Label(parent, text=gname).grid(row=i, column=0, sticky="w")

            # member combobox
            members = self.members_by_group.get(gname, [])
            member_var = tk.StringVar()
            cb = ttk.Combobox(parent, textvariable=member_var, values=members, state="readonly")
            cb.grid(row=i, column=1, sticky="ew", padx=(4, 4))

            # callback: when user picks a member, store it
            cb.bind("<<ComboboxSelected>>", lambda event, group=gname, var=member_var: self.on_member_changed(group, var))

            # equivalents
            eq_info = self.eq_ranges.get(gname)
            eq_var = tk.StringVar()
            if eq_info is None:
                # no equivalents defined -> show "n/a"
                eq_lbl = ttk.Label(parent, text="n/a")
                eq_lbl.grid(row=i, column=2, sticky="w")
                eq_entry = None
            else:
                eq_min, eq_max, unit = eq_info
                if eq_min == eq_max:
                    # fixed -> label only
                    eq_lbl = ttk.Label(parent, text=f"{eq_min} {unit}")
                    eq_lbl.grid(row=i, column=2, sticky="w")
                    eq_entry = None
                else:
                    # editable
                    eq_entry = ttk.Entry(parent, textvariable=eq_var, width=8)
                    eq_entry.grid(row=i, column=2, sticky="w")
                    # propagate changes on Enter or focus-out
                    eq_entry.bind("<FocusOut>", lambda e, g=gname: self.on_eq_changed(g))
                    eq_entry.bind("<Return>", lambda e, g=gname: self.on_eq_changed(g))
                    # store default as empty; we will validate later

            apply_var = tk.BooleanVar(value=True)
            apply_chk = ttk.Checkbutton(parent, variable=apply_var)
            apply_chk.grid(row=i, column=3, sticky="w")

            self.group_widgets[gname] = {
                "member_var": member_var,
                "combobox": cb,
                "eq_var": eq_var,
                "eq_entry": eq_entry,
                "apply_var": apply_var,
            }

    # ---------- state sync ----------

    def select_experiment(self, index: int) -> None:
        """
        Switch focus to experiment `index`, update group widget values,
        and refresh plate colors. (Single-selection version.)
        """
        if index < 0 or index >= self.plate_size:
            return

        # sync current UI into state before switching away
        self._sync_widgets_to_current_state()

        self.current_exp_index = index

        self._load_current_experiment_into_widgets()
        self._update_plate_colors()

    def _load_current_experiment_into_widgets(self) -> None:
        """
        Load self.exp_groups[self.current_exp_index] into the group widgets.
        """
        cur = self.exp_groups[self.current_exp_index]
        for gname in self.group_names:
            w = self.group_widgets.get(gname, {})
            member_var: tk.StringVar = w.get("member_var")  # type: ignore
            eq_var: tk.StringVar = w.get("eq_var")          # type: ignore
            eq_entry = w.get("eq_entry")

            entry = cur.get(gname, {})
            member = entry.get("member")
            eq = entry.get("equivalents")

            if member is not None:
                member_var.set(member)
            else:
                member_var.set("")

            if eq_entry is not None:
                eq_var.set("" if eq is None else str(eq))
            # if no eq_entry (fixed or n/a), nothing to set

    def on_eq_changed(self, group: str) -> None:
        """
        Called when the equivalents entry for a group changes.
        Update ALL batch-target experiments for this group.
        """
        w = self.group_widgets[group]
        raw = w["eq_var"].get().strip()
        if not raw:
            return
        try:
            eq_val = float(raw)
        except ValueError:
            # you could show a messagebox here if you want strict checking
            return

        targets = self._get_batch_targets()
        for idx in targets:
            cur = self.exp_groups[idx]
            entry = cur.setdefault(group, {})
            entry["equivalents"] = eq_val

        self._update_plate_colors()


    def _get_batch_targets(self) -> list[int]:
        """
        Experiments that should receive changes when editing:
        - if some experiments are selected in the batch listbox, use those
        - otherwise, just the current experiment
        """
        sel = list(self.batch_listbox.curselection())
        if sel:
            return sel
        return [self.current_exp_index]

    def on_member_changed(self, group: str, var: tk.StringVar) -> None:
        """
        Called when the user picks a member in a combobox.
        Update ALL batch-target experiments (plate selection).
        """
        member = var.get().strip()
        if not member:
            return

        targets = self._get_batch_targets()
        for idx in targets:
            cur = self.exp_groups[idx]
            cur.setdefault(group, {})["member"] = member

        # widgets already show the new member for the current experiment
        self._update_plate_colors()



    #---------- plate coloring ----------

    def _sync_widgets_to_current_state(self) -> None:
        """
        Read the group widgets (member + equivalents) and store them
        into self.exp_groups[self.current_exp_index].
        """
        idx = self.current_exp_index
        cur = self.exp_groups[idx]

        for gname in self.group_names:
            w = self.group_widgets[gname]
            member = w["member_var"].get().strip()
            eq_entry = w["eq_entry"]
            eq_val = None
            if eq_entry is not None:
                raw = w["eq_var"].get().strip()
                if raw:
                    try:
                        eq_val = float(raw)
                    except ValueError:
                        # leave invalid as None; will be validated on save
                        pass

            if member or eq_val is not None:
                entry = cur.setdefault(gname, {})
                if member:
                    entry["member"] = member
                if eq_val is not None:
                    entry["equivalents"] = eq_val
            # If both empty, we don't force clear; you can add clearing logic here if desired.


    # ---------- saving / validation ----------

    def _collect_globals(self) -> Dict[str, float]:
        """
        Merge base + user-entered variable globals; validate ranges.
        """
        globals_for_plate: Dict[str, float] = dict(self.base_globals)
        for name, (vmin, vmax, unit) in self.variable_defs.items():
            raw = self.var_global_vars[name].get().strip()
            if not raw:
                messagebox.showerror("Missing global", f"Please enter a value for global '{name}'.")
                raise ValueError("missing global")
            try:
                val = float(raw)
            except ValueError:
                messagebox.showerror("Invalid global", f"Global '{name}' must be numeric.")
                raise
            if not (vmin <= val <= vmax):
                messagebox.showerror(
                    "Out-of-range global",
                    f"Global '{name}' must be between {vmin} and {vmax} {unit}.",
                )
                raise ValueError("global out of range")
            globals_for_plate[name] = val
        return globals_for_plate

    def _collect_selections(self, globals_for_plate: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Build the final `selections` list for synthesis_writer:
        [{ "groups": {...}, "globals": {...} }, ...]
        """
        selections: List[Dict[str, Any]] = []
        for i in range(self.plate_size):
            # use the same globals for all wells (as requested)
            sel_globals = dict(globals_for_plate)
            sel_groups = self.exp_groups[i]

            # optional: sanity checks; e.g. at least solvent present, etc.
            selections.append({"groups": sel_groups, "globals": sel_globals})
        return selections

    def apply_current_to_selected(self) -> None:
        """
        Apply current experiment's group settings to all experiments
        selected in the batch listbox, for groups where 'apply' is checked.
        """
        # make sure current widget values are stored
        self._sync_widgets_to_current_state()
        src_idx = self.current_exp_index
        src_groups = self.exp_groups[src_idx]

        # which experiments to update? -> from listbox
        targets = list(self.batch_listbox.curselection())
        if not targets:
            messagebox.showinfo(
                "Batch edit",
                "No destination experiments selected in the batch list.",
            )
            return

        for dst_idx in targets:
            if dst_idx == src_idx:
                continue
            dst_groups = self.exp_groups[dst_idx]
            for gname in self.group_names:
                apply_var = self.group_widgets[gname]["apply_var"]
                if not apply_var.get():
                    continue  # this group not included in batch update

                src_entry = src_groups.get(gname)
                if src_entry is None:
                    dst_groups.pop(gname, None)
                else:
                    dst_groups[gname] = dict(src_entry)

        # refresh plate colors to reflect new assignments
        self._update_plate_colors()

    def on_save(self) -> None:
        """
        Validate inputs, build selections, call synthesis_writer, and exit.
        """
        try:
            globals_for_plate = self._collect_globals()
        except Exception:
            return  # validation already showed messagebox

        # ensure current widgets are stored
        self._sync_widgets_to_current_state()

        # compute limiting moles
        if "concentration" in globals_for_plate and self.well_volume_uL:
            limiting_moles = float(globals_for_plate["concentration"]) * float(self.well_volume_uL) * 1e-6
        else:
            # fallback: ask user once
            raw = tk.simpledialog.askstring(
                "Limiting moles",
                "Enter quantity of limiting reagent (in moles):",
                parent=self,
            )
            if not raw:
                return
            try:
                limiting_moles = float(raw)
            except ValueError:
                messagebox.showerror("Invalid input", "Please enter a numeric value for moles.")
                return

        # build selections
        selections = self._collect_selections(globals_for_plate)

        # hci -> opt_spec
        opt_spec = hci_to_optimizer_dict(str(self.hci_path))

        try:
            write_synthesis_json(
                opt_spec=opt_spec,
                selections=selections,
                out_path=str(self.out_path),
                plate_size=self.plate_size,
                limiting_name=self.limiting_name,
                limiting_moles=limiting_moles,
                well_volume_uL=self.well_volume_uL,
            )
        except Exception as exc:
            messagebox.showerror("Error writing synthesis.json", str(exc))
            return

        messagebox.showinfo("Done", f"Wrote synthesis.json to:\n{self.out_path}")
        self.destroy()

    def on_quit(self) -> None:
        """
        Quit without saving.
        """
        if messagebox.askyesno("Quit", "Quit without saving?"):
            self.destroy()

    def open_copy_dialog(self):
        CopyDialog(self)

    def apply_copy_pattern(self, src_label: str, dst_labels: List[str]):
        """Copy the group selections from src to each destination."""
        try:
            src_index = self.exp_labels.index(src_label)
        except ValueError:
            return

        src_groups = self.exp_groups[src_index]

        for dst in dst_labels:
            if dst == src_label:
                continue
            try:
                dst_idx = self.exp_labels.index(dst)
            except ValueError:
                continue

            # deep copy the dict so instances don’t link accidentally
            self.exp_groups[dst_idx] = {
                gname: dict(info)
                for gname, info in src_groups.items()
            }

        # refresh plate coloring + reload the groups for the current well
        self.select_experiment(self.current_exp_index)

    def _is_experiment_complete(self, idx: int) -> bool:
        """
        For 'completion' color mode: an experiment is complete if
        every group has a member, and where equivalents are required
        (eq_min != eq_max), an 'equivalents' value is present.
        """
        groups = self.exp_groups[idx]
        for gname in self.group_names:
            entry = groups.get(gname)
            if not entry or not entry.get("member"):
                return False
            eq_info = self.eq_ranges.get(gname)
            if eq_info is not None:
                eq_min, eq_max, unit = eq_info
                if eq_min != eq_max:
                    if "equivalents" not in entry:
                        return False
        return True

    def _update_plate_colors(self) -> None:
        """
        Color plate buttons according to current color_mode:
        - 'completion': green if experiment complete, default bg otherwise
        - group name: color-code by group member for that group
        """
        mode = self.color_mode_var.get()
        default_bg = self.cget("bg")

        # build palette if coloring by group
        palette = [
            "#ffcccc", "#ccffcc", "#ccccff", "#ffe0b3", "#e0ccff",
            "#ffd9e6", "#d9ffd9", "#d9e6ff", "#fff2cc", "#e6d9ff",
        ]
        color_by_member: Dict[str, str] = {}

        if mode != "completion" and mode in self.group_names:
            # collect members used for this group
            members = []
            for g in self.exp_groups:
                entry = g.get(mode, {})
                m = entry.get("member")
                if m and m not in members:
                    members.append(m)
            for i, m in enumerate(members):
                color_by_member[m] = palette[i % len(palette)]

        for i, btn in enumerate(self.plate_buttons):
            if mode == "completion":
                bg = "#b6ffb6" if self._is_experiment_complete(i) else default_bg
            elif mode in self.group_names:
                entry = self.exp_groups[i].get(mode, {})
                m = entry.get("member")
                bg = color_by_member.get(m, default_bg)
            else:
                bg = default_bg

            # highlight current experiment specially
            if i == self.current_exp_index:
                btn.config(relief="sunken", bg="#ffd080")
            else:
                btn.config(relief="raised", bg=bg)



class CopyDialog(tk.Toplevel):
    def __init__(self, parent: "ManualPlateGUI"):
        super().__init__(parent)
        self.parent = parent
        self.title("Copy Pattern")
        self.resizable(False, False)

        padx = 8
        pady = 6

        # FROM selector
        ttk.Label(self, text="Copy FROM experiment:").grid(row=0, column=0, sticky="w", padx=padx, pady=pady)
        from_values = parent.exp_labels
        self.from_var = tk.StringVar(value=from_values[0])
        ttk.Combobox(self, textvariable=self.from_var, values=from_values, state="readonly").grid(
            row=0, column=1, padx=padx, pady=pady
        )

        # TO selector
        ttk.Label(self, text="Copy TO:").grid(row=1, column=0, sticky="nw", padx=padx, pady=pady)
        self.to_list = tk.Listbox(self, selectmode="multiple", exportselection=False, height=10)
        for label in parent.exp_labels:
            self.to_list.insert("end", label)
        self.to_list.grid(row=1, column=1, sticky="ew", padx=padx, pady=pady)

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=(10, 6))
        ttk.Button(btn_frame, text="Copy", command=self.on_copy).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).grid(row=0, column=1, padx=5)

    def on_copy(self):
        src = self.from_var.get()
        if not src:
            messagebox.showerror("Error", "No source well selected.")
            return

        dst_indices = self.to_list.curselection()
        if not dst_indices:
            messagebox.showerror("Error", "No destination wells selected.")
            return

        dst_labels = [self.to_list.get(i) for i in dst_indices]
        self.parent.apply_copy_pattern(src, dst_labels)
        self.destroy()


# ---------- CLI entry point ----------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "GUI manual plate builder:\n"
            "- read globals from hasRanges (auto-fill constants, editable ranges)\n"
            "- read groups/members from hasGroups\n"
            "- let the user pick per-well groups in a plate layout\n"
            "- write synthesis.json via synthesis_writer"
        )
    )
    ap.add_argument("--hci-file", required=True, help="HCI JSON (your schema).")
    ap.add_argument("--out", required=True, help="Output synthesis.json (relative to out-dir).")
    ap.add_argument("--plate-size", type=int, help="Override plate size (else read from HCI).")
    ap.add_argument("--limiting-name", help="Limiting reagent name (e.g. substrate A).")
    ap.add_argument("--well-volume-uL", type=float, required=True, help="Total volume per well (µL).")

    ap.add_argument("--out-dir", default=str(OUT_DIR), help="Directory to write synthesis.json")
    ap.add_argument("--data-dir", default=str(DATA_DIR), help="(unused here; kept for consistency)")

    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    out_path = out_dir / args.out

    hci_path = Path(args.hci_file).resolve()
    if not hci_path.is_file():
        raise SystemExit(f"HCI file not found: {hci_path}")

    # Load HCI, derive plate size and globals
    hci = load_hci(hci_path)
    plate_size = int(args.plate_size or hci_plate_size(hci))

    base_globals, variable_defs = hci_globals_from_ranges(hci)

    app = ManualPlateGUI(
        hci=hci,
        hci_path=hci_path,
        plate_size=plate_size,
        base_globals=base_globals,
        variable_defs=variable_defs,
        out_path=out_path,
        limiting_name=args.limiting_name,
        well_volume_uL=args.well_volume_uL,
    )
    app.mainloop()


if __name__ == "__main__":
    main()
