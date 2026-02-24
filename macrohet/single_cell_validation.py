import io
import os
import re

import imageio.v3 as iio
import matplotlib.pyplot as plt
import napari
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

from macrohet.growth_model import (
    compute_doubling_metrics,
    fit_lowess,
    process_mtb_area,
)


class GlimpseAnnotator:
    def __init__(self, full_df, id_list, glimpse_dir, save_path):
        """
        Parameters
        ----------
        full_df : pd.DataFrame
            The main metrics dataframe.
        id_list : list
            The specific IDs you want to iterate through.
        glimpse_dir : str
            Path to the directory containing .mp4 files.
        save_path : str
            Path where the updated pickle file should be saved.
        """
        self.df = full_df.copy()
        self.id_list = id_list
        self.glimpse_dir = glimpse_dir
        self.save_path = save_path

        self.current_idx = 0
        self.total = len(id_list)

        self.viewer = napari.Viewer(title="Single-cell validator")
        self.setup_bindings()
        self.load_sample()

    def setup_bindings(self):
        self.viewer.bind_key("n", self.next_sample)
        self.viewer.bind_key("b", self.prev_sample)
        self.viewer.bind_key("s", self.save_df)
        self.viewer.bind_key("t", self.mark_transfer)
        self.viewer.bind_key("u", self.mark_uptake)
        self.viewer.bind_key("j", self.mark_junk)
        self.viewer.bind_key("g", self.mark_growth)
        self.viewer.bind_key("e", self.mark_edge)
        self.viewer.bind_key("k", self.split_track)

    def load_sample(self):
        if self.current_idx >= self.total:
            print("All samples processed!")
            return

        self.current_id = self.id_list[self.current_idx]
        video_path = os.path.join(self.glimpse_dir, f"{self.current_id}.mp4")

        if not os.path.exists(video_path):
            print(f"Missing Video for {self.current_id}. Skipping...")
            self.current_idx += 1
            self.load_sample()
            return

        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()

        print(f"--- Processing {self.current_idx + 1}/{self.total}: {self.current_id} ---")

        video_loaded = False
        try:
            video = iio.imread(video_path, plugin="pyav")
            self.viewer.add_image(video, name=f"{self.current_id} Video")
            video_loaded = True
        except Exception as e:
            print(f"Warning: Could not read video. Error: {e}")

        try:
            plot_img = self.create_plot_image(self.current_id)
            if plot_img is not None:
                self.viewer.add_image(plot_img, name="Growth Plot")
        except Exception as e:
            print(f"Error generating plot: {e}")

        self.viewer.grid.enabled = True
        self.viewer.grid.shape = (1, 2) if video_loaded else (1, 1)
        self.viewer.reset_view()

        if self.viewer.dims.ndim > 0:
            current_step = list(self.viewer.dims.current_step)
            current_step[0] = 0
            self.viewer.dims.current_step = tuple(current_step)

    def create_plot_image(self, ID):
        sc_df = self.df[self.df["ID"] == ID].copy()
        if sc_df.empty:
            return None
        sc_df = sc_df.sort_values(by="Time Model (hours)")

        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

        try:
            times = sc_df["Time Model (hours)"].values
            area_model = sc_df["Mtb Area Model (µm)"].values
            area_proc = sc_df["Mtb Area Processed (µm)"].values

            meta = sc_df.iloc[0]
            r2 = meta.get("r2", np.nan)
            strain = meta.get("Strain", "Unknown")
            compound = meta.get("Compound", "N/A")
            conc = meta.get("Concentration", "N/A")

            def find_crossing(target, t_arr, a_arr):
                if not np.any(a_arr >= target):
                    return None
                filled_a = np.nan_to_num(a_arr, nan=-np.inf)
                idx = np.argmax(filled_a >= target)
                if idx == 0 and filled_a[0] < target:
                    return None
                if idx == 0:
                    return t_arr[0]

                t1, t2 = t_arr[idx - 1], t_arr[idx]
                a1, a2 = filled_a[idx - 1], filled_a[idx]
                if a2 == a1:
                    return t1
                return t1 + (t2 - t1) * ((target - a1) / (a2 - a1))

            raw_start_area = area_model[0] if not np.isnan(area_model[0]) else 1.92
            baseline = max(1.92, raw_start_area)

            if raw_start_area < baseline:
                t_start = find_crossing(baseline, times, area_model)
            else:
                t_start = times[0]

            if t_start is None:
                crossings_x, crossings_y = [], []
            else:
                crossings_x, crossings_y = [t_start], [baseline]

            grid = [baseline * (2**i) for i in range(1, 6)]

            for target in grid:
                t_cross = find_crossing(target, times, area_model)
                if t_cross is not None:
                    if t_cross >= crossings_x[-1]:
                        crossings_x.append(t_cross)
                        crossings_y.append(target)
                    else:
                        break
                else:
                    break

            ax.plot(times, area_model, color="#d02c91", lw=2, label="Model")
            ax.scatter(times, area_proc, color="#1a9641", s=10, alpha=0.6, label="Data")

            if crossings_x:
                ax.axhline(y=baseline, color="lightgrey", linestyle="--", lw=1.5)
                ax.axvline(x=crossings_x[0], color="lightgrey", linestyle="--", lw=1.5)

            for i in range(1, len(crossings_x)):
                t_prev, t_curr = crossings_x[i - 1], crossings_x[i]
                y_prev, y_curr = crossings_y[i - 1], crossings_y[i]

                ax.axhline(y=y_curr, color="lightgrey", linestyle="--", lw=1.5)
                ax.axvline(x=t_curr, color="lightgrey", linestyle="--", lw=1.5)

                dt = t_curr - t_prev
                delta_y = y_curr - y_prev
                label_text = f"∆T={dt:.1f}h | ∆Mtb={delta_y:.2f}µm²"

                ax.text(
                    t_prev + 0.2,
                    y_prev * 1.02,
                    label_text,
                    color="#505050",
                    fontsize=8,
                    fontweight="bold",
                    ha="left",
                    va="bottom",
                    bbox=dict(
                        boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.7
                    ),
                )

            title_str = f"ID: {ID}\n{strain} | {compound} {conc} | R2: {r2:.2f}"
            ax.set_title(title_str, fontsize=10)

            custom_lines = [
                Line2D([0], [0], color="#d02c91", lw=2),
                Line2D([0], [0], color="#1a9641", marker="o", lw=0),
                Line2D([0], [0], color="lightgrey", linestyle="--", lw=1.5),
            ]
            ax.legend(
                custom_lines,
                ["Model", "Data", "Doubling Grid"],
                loc="upper left",
                fontsize=8,
            )

            ax.set_xlabel("Time (hours)")
            ax.set_ylabel("Mtb Area (µm²)")
            sns.despine(offset=10)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return iio.imread(buf)

        except Exception as e:
            print(f"Plotting error for {ID}: {e}")
            plt.close(fig)
            return None

    def next_sample(self, viewer):
        self.current_idx += 1
        self.load_sample()

    def prev_sample(self, viewer):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_sample()

    def update_origin(self, status):
        self.df.loc[self.df["ID"] == self.current_id, "mtb_origin"] = status
        print(f"Set {self.current_id} -> {status}")

    def mark_transfer(self, viewer):
        self.update_origin("Transfer")

    def mark_uptake(self, viewer):
        self.update_origin("Uptake")

    def mark_junk(self, viewer):
        self.update_origin("Junk")

    def mark_growth(self, viewer):
        self.update_origin("Growth")

    def mark_edge(self, viewer):
        self.df.loc[self.df["ID"] == self.current_id, "Edge Status"] = True
        print(f"ID {self.current_id} marked as 'Edge Status'.")

    def save_df(self, viewer):
        """Saves the current state of the dataframe to the specified save_path."""
        print(f"Saving to {self.save_path}...")
        tmp_path = self.save_path + ".tmp"
        self.df.to_pickle(tmp_path)
        os.replace(tmp_path, self.save_path)
        print("Save successful.")

    def split_track(self, viewer):
        print(f"\nAttempting Split on {self.current_id}")
        try:
            current_frame = viewer.dims.current_step[0]
        except Exception:
            return

        track_df = self.df[self.df["ID"] == self.current_id]
        if track_df.empty:
            return

        start_time = track_df["Time (hours)"].min()
        end_time = track_df["Time (hours)"].max()
        split_time = start_time + (current_frame * 0.5)

        if split_time > end_time or split_time < start_time:
            return

        try:
            root_id = re.sub(r"[a-z]+$", "", self.current_id)
            existing_ids = self.df[self.df["ID"].str.startswith(root_id)]["ID"].unique()
            suffixes = [uid.replace(root_id, "") for uid in existing_ids]
            letters = [s for s in suffixes if s.isalpha()]
            next_char = "b" if not letters else chr(ord(max(letters)) + 1)
            new_id = root_id + next_char
        except Exception:
            return

        mask = (self.df["ID"] == self.current_id) & (
            self.df["Time (hours)"] >= split_time
        )
        rows_affected = mask.sum()
        if rows_affected == 0:
            return

        self.df.loc[mask, "ID"] = new_id
        print(f"Moved {rows_affected} rows to {new_id}")

        try:
            self.refresh_metrics_for_ids([self.current_id, new_id])
        except Exception as e:
            print(f"Metric recalculation failed: {e}")

        self.load_sample()

    def refresh_metrics_for_ids(self, id_list):
        print(f"Recalculating pipeline for {id_list}")
        mask = self.df["ID"].isin(id_list)
        if mask.sum() == 0:
            return

        sub_df = self.df[mask].copy()
        try:
            sub_df = process_mtb_area(sub_df, window=10, spike_threshold=10.0)
            sub_df = fit_lowess(sub_df, frac=0.25)
            sub_df = compute_doubling_metrics(sub_df, min_area=1.92, r2_threshold=0.7)

            cols_to_update = [
                "Mtb Area Processed (µm)",
                "Time Model (hours)",
                "Mtb Area Model (µm)",
                "r2",
                "Doubling Amounts",
                "Doubling Times",
            ]
            available_cols = [c for c in cols_to_update if c in sub_df.columns]

            self.df.loc[mask, available_cols] = sub_df[available_cols]
            print("Recalculation Complete.")
        except Exception as e:
            print(f"Recalculation Error: {e}")