"""
Tkinter GUI for the Tidal Predictor application.

Layout:
  - Left panel: station info, constituents table, prediction settings, Predict button
  - Right panel: notebook with Graph tab (matplotlib) and Table tab (treeview)
  - Menu bar: File menu for profile management and CSV export
  - Status bar at bottom
"""

import csv
import json
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from tide_math import predict_tides, get_constituent_names, CONSTITUENTS
from station import Station, Constituent, list_profiles, delete_profile

# Resolve base path (works both normally and when bundled by PyInstaller)
if getattr(sys, 'frozen', False):
    _BASE_DIR = sys._MEIPASS
else:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load preset Malaysian stations
_PRESETS_PATH = os.path.join(_BASE_DIR, "malaysia_stations.json")
PRESET_STATIONS = {}
if os.path.exists(_PRESETS_PATH):
    with open(_PRESETS_PATH, "r", encoding="utf-8") as _f:
        for _s in json.load(_f):
            PRESET_STATIONS[_s["name"]] = _s


class TidalPredictorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Tidal Predictor")
        self.root.geometry("1200x750")
        self.root.minsize(900, 600)

        # Current prediction data
        self.pred_times = []
        self.pred_heights = None

        self._build_menu()
        self._build_ui()
        self._update_status("Ready. Enter station data and click Predict.")

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------
    def _build_menu(self):
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New", command=self._new_station, accelerator="Ctrl+N")
        file_menu.add_command(label="Load Profile...", command=self._load_profile)
        file_menu.add_command(label="Save Profile", command=self._save_profile, accelerator="Ctrl+S")
        file_menu.add_command(label="Delete Profile...", command=self._delete_profile)
        file_menu.add_separator()
        file_menu.add_command(label="Export CSV...", command=self._export_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

        self.root.bind("<Control-n>", lambda e: self._new_station())
        self.root.bind("<Control-s>", lambda e: self._save_profile())

    # ------------------------------------------------------------------
    # UI Layout
    # ------------------------------------------------------------------
    def _build_ui(self):
        # Main paned window
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left panel
        left = ttk.Frame(paned, width=360)
        paned.add(left, weight=0)

        # Right panel
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        self._build_left_panel(left)
        self._build_right_panel(right)

        # Status bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=(0, 5))

    def _build_left_panel(self, parent):
        self._left_canvas = canvas = tk.Canvas(parent, highlightthickness=0)
        self._left_scrollbar = scrollbar = ttk.Scrollbar(
            parent, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", self._update_left_scroll)
        canvas.bind("<Configure>", self._update_left_scroll)
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # scrollbar starts hidden; shown only when needed

        def _on_mousewheel(event):
            if not self._left_scrollbar_visible:
                return
            # Only scroll canvas when mouse is over the canvas itself
            w = event.widget
            try:
                while w:
                    if w is canvas:
                        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                        return
                    w = w.master
            except Exception:
                pass
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._left_scrollbar_visible = False

        self._build_preset_selector(scroll_frame)
        self._build_station_info(scroll_frame)
        self._build_constituents_section(scroll_frame)
        self._build_prediction_settings(scroll_frame)
        self._build_predict_button(scroll_frame)

    def _update_left_scroll(self, event=None):
        canvas = self._left_canvas
        scrollbar = self._left_scrollbar
        canvas.configure(scrollregion=canvas.bbox("all"))
        content_height = canvas.bbox("all")[3] if canvas.bbox("all") else 0
        visible_height = canvas.winfo_height()
        need_scroll = content_height > visible_height
        if need_scroll and not self._left_scrollbar_visible:
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self._left_scrollbar_visible = True
        elif not need_scroll and self._left_scrollbar_visible:
            scrollbar.pack_forget()
            self._left_scrollbar_visible = False

    def _build_preset_selector(self, parent):
        frame = ttk.LabelFrame(parent, text="Malaysia Preset Stations", padding=8)
        frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(frame, text="Select Port:").pack(anchor=tk.W)
        preset_names = ["-- Select a port --"] + sorted(PRESET_STATIONS.keys())
        self.preset_var = tk.StringVar(value=preset_names[0])
        combo = ttk.Combobox(frame, textvariable=self.preset_var,
                             values=preset_names, state="readonly", width=30)
        combo.pack(fill=tk.X, pady=(4, 0))
        combo.bind("<<ComboboxSelected>>", self._on_preset_selected)

    def _on_preset_selected(self, event=None):
        name = self.preset_var.get()
        if name not in PRESET_STATIONS:
            return
        ps = PRESET_STATIONS[name]

        self.station_name_var.set(ps["name"])
        self.lat_var.set(str(ps["latitude"]))
        self.lon_var.set(str(ps["longitude"]))
        self.datum_var.set("MSL")
        self.offset_var.set("0.0")

        # Load all constituents
        self.const_tree.delete(*self.const_tree.get_children())
        for c in ps["constituents"]:
            self.const_tree.insert("", tk.END,
                                   values=(c["name"], f"{c['amplitude']:.4f}", f"{c['phase']:.2f}"))

        self._update_status(f"Loaded preset: {ps['name']} ({len(ps['constituents'])} constituents)")

    def _build_station_info(self, parent):
        frame = ttk.LabelFrame(parent, text="Station Info", padding=8)
        frame.pack(fill=tk.X, padx=5, pady=5)

        labels = ["Name:", "Latitude:", "Longitude:", "Datum:", "Datum Offset:"]
        self.station_name_var = tk.StringVar()
        self.lat_var = tk.StringVar(value="0.0")
        self.lon_var = tk.StringVar(value="0.0")
        self.datum_var = tk.StringVar(value="MLLW")
        self.offset_var = tk.StringVar(value="0.0")

        vars_ = [self.station_name_var, self.lat_var, self.lon_var,
                 self.datum_var, self.offset_var]

        for i, (label, var) in enumerate(zip(labels, vars_)):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            ttk.Entry(frame, textvariable=var, width=25).grid(
                row=i, column=1, sticky=tk.EW, pady=2, padx=(5, 0))

        frame.columnconfigure(1, weight=1)

    def _build_constituents_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Harmonic Constituents", padding=8)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # Treeview for constituents
        cols = ("name", "amplitude", "phase")
        self.const_tree = ttk.Treeview(frame, columns=cols, show="headings", height=8)
        self.const_tree.heading("name", text="Name")
        self.const_tree.heading("amplitude", text="Amplitude (m)")
        self.const_tree.heading("phase", text="Phase (°)")
        self.const_tree.column("name", width=80)
        self.const_tree.column("amplitude", width=100)
        self.const_tree.column("phase", width=80)

        tree_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL,
                                    command=self.const_tree.yview)
        self.const_tree.configure(yscrollcommand=tree_scroll.set)
        self.const_tree.grid(row=0, column=0, columnspan=3, sticky=tk.NSEW)
        tree_scroll.grid(row=0, column=3, sticky=tk.NS)

        # Add/Edit/Remove controls
        add_frame = ttk.Frame(frame)
        add_frame.grid(row=1, column=0, columnspan=4, sticky=tk.EW, pady=(8, 0))

        ttk.Label(add_frame, text="Name:").grid(row=0, column=0, padx=(0, 2))
        self.const_name_var = tk.StringVar()
        self.const_combo = ttk.Combobox(add_frame, textvariable=self.const_name_var,
                                        values=get_constituent_names(), width=10,
                                        state="readonly")
        self.const_combo.grid(row=0, column=1, padx=2)

        ttk.Label(add_frame, text="Amp:").grid(row=0, column=2, padx=(8, 2))
        self.const_amp_var = tk.StringVar()
        ttk.Entry(add_frame, textvariable=self.const_amp_var, width=8).grid(
            row=0, column=3, padx=2)

        ttk.Label(add_frame, text="Phase:").grid(row=0, column=4, padx=(8, 2))
        self.const_phase_var = tk.StringVar()
        ttk.Entry(add_frame, textvariable=self.const_phase_var, width=8).grid(
            row=0, column=5, padx=2)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=2, column=0, columnspan=4, sticky=tk.EW, pady=(5, 0))
        ttk.Button(btn_frame, text="Add", command=self._add_constituent, width=8).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Edit", command=self._edit_constituent, width=8).pack(
            side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Remove", command=self._remove_constituent, width=8).pack(
            side=tk.LEFT, padx=2)

        frame.columnconfigure(0, weight=1)

    def _build_prediction_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="Prediction Settings", padding=8)
        frame.pack(fill=tk.X, padx=5, pady=5)

        # Date range
        today = datetime.utcnow()
        default_start = today.strftime("%Y-%m-%d")
        default_end = (today + timedelta(days=7)).strftime("%Y-%m-%d")

        ttk.Label(frame, text="Start Date (YYYY-MM-DD):").grid(
            row=0, column=0, sticky=tk.W, pady=2)
        self.start_var = tk.StringVar(value=default_start)
        ttk.Entry(frame, textvariable=self.start_var, width=15).grid(
            row=0, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        ttk.Label(frame, text="End Date (YYYY-MM-DD):").grid(
            row=1, column=0, sticky=tk.W, pady=2)
        self.end_var = tk.StringVar(value=default_end)
        ttk.Entry(frame, textvariable=self.end_var, width=15).grid(
            row=1, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        # Timezone
        ttk.Label(frame, text="Display Timezone:").grid(
            row=2, column=0, sticky=tk.W, pady=2)
        self.tz_mode_var = tk.StringVar(value="UTC")
        tz_frame = ttk.Frame(frame)
        tz_frame.grid(row=2, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        ttk.Radiobutton(tz_frame, text="UTC", variable=self.tz_mode_var,
                        value="UTC").pack(side=tk.LEFT)
        ttk.Radiobutton(tz_frame, text="Local", variable=self.tz_mode_var,
                        value="Local").pack(side=tk.LEFT, padx=(10, 0))

        ttk.Label(frame, text="UTC Offset (hours):").grid(
            row=3, column=0, sticky=tk.W, pady=2)
        self.utc_offset_var = tk.StringVar(value="0")
        ttk.Entry(frame, textvariable=self.utc_offset_var, width=8).grid(
            row=3, column=1, sticky=tk.W, padx=(5, 0), pady=2)

        frame.columnconfigure(1, weight=1)

    def _build_predict_button(self, parent):
        btn = ttk.Button(parent, text="PREDICT", command=self._run_prediction)
        btn.pack(fill=tk.X, padx=5, pady=(10, 2), ipady=8)

        export_btn = ttk.Button(parent, text="Export CSV", command=self._export_csv)
        export_btn.pack(fill=tk.X, padx=5, pady=(2, 10), ipady=4)

    def _build_right_panel(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Graph tab
        graph_frame = ttk.Frame(self.notebook)
        self.notebook.add(graph_frame, text="  Graph  ")

        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Water Level (m)")
        self.ax.set_title("Tidal Prediction")
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.draw()

        toolbar = NavigationToolbar2Tk(self.canvas, graph_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Table tab
        table_frame = ttk.Frame(self.notebook)
        self.notebook.add(table_frame, text="  Table  ")

        table_cols = ("datetime", "height")
        self.result_tree = ttk.Treeview(table_frame, columns=table_cols,
                                        show="headings")
        self.result_tree.heading("datetime", text="Date/Time")
        self.result_tree.heading("height", text="Height (m)")
        self.result_tree.column("datetime", width=200)
        self.result_tree.column("height", width=120)

        result_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL,
                                      command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=result_scroll.set)
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ------------------------------------------------------------------
    # Constituent management
    # ------------------------------------------------------------------
    def _add_constituent(self):
        name = self.const_name_var.get().strip()
        amp_str = self.const_amp_var.get().strip()
        phase_str = self.const_phase_var.get().strip()

        if not name:
            messagebox.showwarning("Input Error", "Select a constituent name.")
            return
        try:
            amp = float(amp_str)
        except ValueError:
            messagebox.showwarning("Input Error", "Amplitude must be a number.")
            return
        try:
            phase = float(phase_str)
        except ValueError:
            messagebox.showwarning("Input Error", "Phase must be a number.")
            return
        if amp < 0:
            messagebox.showwarning("Input Error", "Amplitude must be non-negative.")
            return

        # Check for duplicate
        for item in self.const_tree.get_children():
            vals = self.const_tree.item(item, "values")
            if vals[0] == name:
                messagebox.showwarning("Duplicate",
                                       f"Constituent {name} already exists. Use Edit to modify.")
                return

        self.const_tree.insert("", tk.END, values=(name, f"{amp:.4f}", f"{phase:.2f}"))
        self.const_amp_var.set("")
        self.const_phase_var.set("")
        self._update_status(f"Added constituent: {name}")

    def _edit_constituent(self):
        sel = self.const_tree.selection()
        if not sel:
            messagebox.showinfo("Edit", "Select a constituent in the table first.")
            return
        item = sel[0]

        name = self.const_name_var.get().strip()
        amp_str = self.const_amp_var.get().strip()
        phase_str = self.const_phase_var.get().strip()

        if not name:
            messagebox.showwarning("Input Error", "Select a constituent name.")
            return
        try:
            amp = float(amp_str)
        except ValueError:
            messagebox.showwarning("Input Error", "Amplitude must be a number.")
            return
        try:
            phase = float(phase_str)
        except ValueError:
            messagebox.showwarning("Input Error", "Phase must be a number.")
            return

        self.const_tree.item(item, values=(name, f"{amp:.4f}", f"{phase:.2f}"))
        self._update_status(f"Updated constituent: {name}")

    def _remove_constituent(self):
        sel = self.const_tree.selection()
        if not sel:
            messagebox.showinfo("Remove", "Select a constituent in the table first.")
            return
        name = self.const_tree.item(sel[0], "values")[0]
        self.const_tree.delete(sel[0])
        self._update_status(f"Removed constituent: {name}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _get_constituents_from_tree(self):
        """Extract constituents from the treeview as list of dicts."""
        constituents = []
        for item in self.const_tree.get_children():
            vals = self.const_tree.item(item, "values")
            constituents.append({
                "name": vals[0],
                "amplitude": float(vals[1]),
                "phase": float(vals[2]),
            })
        return constituents

    def _run_prediction(self):
        # Validate inputs
        constituents = self._get_constituents_from_tree()
        if not constituents:
            messagebox.showwarning("No Constituents",
                                   "Add at least one harmonic constituent.")
            return

        try:
            start_dt = datetime.strptime(self.start_var.get().strip(), "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Invalid Date",
                                 "Start date must be YYYY-MM-DD format.")
            return
        try:
            end_dt = datetime.strptime(self.end_var.get().strip(), "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Invalid Date",
                                 "End date must be YYYY-MM-DD format.")
            return

        if end_dt <= start_dt:
            messagebox.showerror("Invalid Range", "End date must be after start date.")
            return

        try:
            datum_offset = float(self.offset_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Input", "Datum offset must be a number.")
            return

        try:
            utc_offset = float(self.utc_offset_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Input", "UTC offset must be a number.")
            return

        self._update_status("Computing prediction...")
        self.root.update_idletasks()

        try:
            times, heights = predict_tides(
                constituents=constituents,
                start_dt=start_dt,
                end_dt=end_dt,
                interval_hours=1.0,
                datum_offset=datum_offset,
            )
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            self._update_status("Prediction failed.")
            return

        # Apply display timezone offset
        display_offset = timedelta(hours=utc_offset) if self.tz_mode_var.get() == "Local" else timedelta(0)
        display_times = [t + display_offset for t in times]
        tz_label = f"UTC{utc_offset:+.1f}" if self.tz_mode_var.get() == "Local" else "UTC"

        # Store for CSV export
        self.pred_times = display_times
        self.pred_heights = heights

        # Update graph
        self.ax.clear()
        self.ax.plot(display_times, heights, color="#1f77b4", linewidth=0.8)
        self.ax.set_xlabel(f"Time ({tz_label})")
        self.ax.set_ylabel("Water Level (m)")
        station_name = self.station_name_var.get().strip() or "Station"
        self.ax.set_title(f"Tidal Prediction — {station_name}")
        self.ax.grid(True, alpha=0.3)
        self.fig.autofmt_xdate()
        self.fig.tight_layout()
        self.canvas.draw()

        # Update table
        self.result_tree.delete(*self.result_tree.get_children())
        for t, h in zip(display_times, heights):
            self.result_tree.insert("", tk.END, values=(
                t.strftime("%Y-%m-%d %H:%M"),
                f"{h:.3f}",
            ))

        self._update_status(
            f"Prediction complete: {len(times)} points, "
            f"{min(heights):.3f} m to {max(heights):.3f} m"
        )

    # ------------------------------------------------------------------
    # Profile management
    # ------------------------------------------------------------------
    def _station_from_ui(self) -> Station:
        """Build a Station object from current UI state."""
        constituents = []
        for item in self.const_tree.get_children():
            vals = self.const_tree.item(item, "values")
            constituents.append(Constituent(
                name=vals[0],
                amplitude=float(vals[1]),
                phase=float(vals[2]),
            ))

        try:
            lat = float(self.lat_var.get())
        except ValueError:
            lat = 0.0
        try:
            lon = float(self.lon_var.get())
        except ValueError:
            lon = 0.0
        try:
            offset = float(self.offset_var.get())
        except ValueError:
            offset = 0.0

        return Station(
            name=self.station_name_var.get().strip(),
            latitude=lat,
            longitude=lon,
            datum_name=self.datum_var.get().strip(),
            datum_offset=offset,
            timezone_label=self.tz_mode_var.get(),
            constituents=constituents,
        )

    def _load_station_to_ui(self, station: Station):
        """Populate UI from a Station object."""
        self.station_name_var.set(station.name)
        self.lat_var.set(str(station.latitude))
        self.lon_var.set(str(station.longitude))
        self.datum_var.set(station.datum_name)
        self.offset_var.set(str(station.datum_offset))

        # Clear and reload constituents
        self.const_tree.delete(*self.const_tree.get_children())
        for c in station.constituents:
            self.const_tree.insert("", tk.END,
                                   values=(c.name, f"{c.amplitude:.4f}", f"{c.phase:.2f}"))

    def _new_station(self):
        self.station_name_var.set("")
        self.lat_var.set("0.0")
        self.lon_var.set("0.0")
        self.datum_var.set("MLLW")
        self.offset_var.set("0.0")
        self.const_tree.delete(*self.const_tree.get_children())
        self.ax.clear()
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Water Level (m)")
        self.ax.set_title("Tidal Prediction")
        self.canvas.draw()
        self.result_tree.delete(*self.result_tree.get_children())
        self.pred_times = []
        self.pred_heights = None
        self._update_status("New station created.")

    def _save_profile(self):
        station = self._station_from_ui()
        if not station.name:
            messagebox.showwarning("Save", "Enter a station name first.")
            return
        try:
            path = station.save()
            self._update_status(f"Profile saved: {path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _load_profile(self):
        profiles = list_profiles()
        if not profiles:
            messagebox.showinfo("Load", "No saved profiles found.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Load Profile")
        dialog.geometry("350x300")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Select a profile:").pack(padx=10, pady=(10, 5))

        listbox = tk.Listbox(dialog, selectmode=tk.SINGLE)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10)
        for p in profiles:
            listbox.insert(tk.END, p)

        def do_load():
            sel = listbox.curselection()
            if not sel:
                return
            filename = profiles[sel[0]]
            try:
                station = Station.load(filename)
                self._load_station_to_ui(station)
                self._update_status(f"Loaded profile: {station.name}")
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Load Error", str(e))

        ttk.Button(dialog, text="Load", command=do_load).pack(pady=10)

    def _delete_profile(self):
        profiles = list_profiles()
        if not profiles:
            messagebox.showinfo("Delete", "No saved profiles found.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Delete Profile")
        dialog.geometry("350x300")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Select a profile to delete:").pack(padx=10, pady=(10, 5))

        listbox = tk.Listbox(dialog, selectmode=tk.SINGLE)
        listbox.pack(fill=tk.BOTH, expand=True, padx=10)
        for p in profiles:
            listbox.insert(tk.END, p)

        def do_delete():
            sel = listbox.curselection()
            if not sel:
                return
            filename = profiles[sel[0]]
            if messagebox.askyesno("Confirm Delete",
                                   f"Delete profile '{filename}'?"):
                try:
                    delete_profile(filename)
                    self._update_status(f"Deleted profile: {filename}")
                    dialog.destroy()
                except Exception as e:
                    messagebox.showerror("Delete Error", str(e))

        ttk.Button(dialog, text="Delete", command=do_delete).pack(pady=10)

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------
    def _export_csv(self):
        if not self.pred_times or self.pred_heights is None:
            messagebox.showinfo("Export", "Run a prediction first.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Prediction to CSV",
        )
        if not filepath:
            return

        try:
            constituents = self._get_constituents_from_tree()
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                # Station info header
                writer.writerow(["Station", self.station_name_var.get()])
                writer.writerow(["Latitude", self.lat_var.get()])
                writer.writerow(["Longitude", self.lon_var.get()])
                writer.writerow(["Datum", self.datum_var.get()])
                writer.writerow(["Datum Offset", self.offset_var.get()])
                writer.writerow([])

                # Harmonic constants table
                writer.writerow(["Harmonic Constants"])
                writer.writerow(["Constituent", "Amplitude_m", "Phase_deg"])
                for c in constituents:
                    writer.writerow([c["name"], f"{c['amplitude']:.4f}", f"{c['phase']:.2f}"])
                writer.writerow([])

                # Prediction table
                writer.writerow(["Tidal Predictions"])
                writer.writerow(["DateTime", "Height_m"])
                for t, h in zip(self.pred_times, self.pred_heights):
                    writer.writerow([t.strftime("%Y-%m-%d %H:%M"), f"{h:.4f}"])

            self._update_status(f"CSV exported: {filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    # ------------------------------------------------------------------
    # Status bar
    # ------------------------------------------------------------------
    def _update_status(self, msg: str):
        self.status_var.set(msg)
