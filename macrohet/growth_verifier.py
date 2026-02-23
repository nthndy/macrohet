import io
import os
import re

import imageio.v2 as imageio  # Explicit v2 API for reliability with ffmpeg
import imageio.v3 as iio
import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
import seaborn as sns

from macrohet.growth_model import (
    compute_doubling_metrics,
    fit_lowess,
    process_mtb_area,
)

# ==========================================
# THE ANNOTATOR GUI CLASS
# ==========================================

class GlimpseAnnotator:
    def __init__(self, full_df, id_list, video_dir='./glimpses/', save_path='./verified_df.pkl'):
        self.df = full_df
        self.id_list = id_list
        self.video_dir = video_dir
        self.save_path = save_path

        self.current_idx = 0
        self.total = len(id_list)

        # Initialize Viewer
        self.viewer = napari.Viewer(title="Growth Profile Verifier")

        # Setup bindings and load first sample
        self.setup_bindings()
        self.load_sample()

    def setup_bindings(self):
        # Navigation
        self.viewer.bind_key('n', self.next_sample)
        self.viewer.bind_key('b', self.prev_sample)
        self.viewer.bind_key('s', self.save_df)

        # Actions
        self.viewer.bind_key('k', self.split_track) # 'k' for Knife/Cut

        # Classification
        self.viewer.bind_key('t', self.mark_transfer)
        self.viewer.bind_key('u', self.mark_uptake)
        self.viewer.bind_key('j', self.mark_junk)
        self.viewer.bind_key('g', self.mark_growth)
        self.viewer.bind_key('e', self.mark_edge)

    def load_sample(self):
        """Loads video/plot. Automatically skips IDs if video is missing."""
        if self.current_idx >= self.total:
            print("All samples processed!")
            return

        self.current_id = self.id_list[self.current_idx]

        # 1. Video Check & Load
        video_path = os.path.join(self.video_dir, f'{self.current_id}.mp4')

        if not os.path.exists(video_path):
            print(f"⚠️ Missing Video for {self.current_id}. Skipping...")
            self.current_idx += 1
            if self.current_idx < self.total:
                self.load_sample()
            return

        # Clear layers
        self.viewer.layers.select_all()
        self.viewer.layers.remove_selected()

        print(f"--- Processing {self.current_idx + 1}/{self.total}: {self.current_id} ---")

        try:
            reader = imageio.get_reader(video_path, format='ffmpeg')
            video = np.stack([frame for frame in reader])
            reader.close()
            self.viewer.add_image(video, name=f'{self.current_id}')
        except Exception as e:
            print(f"Error loading video: {e}")

        # 2. Plot Graph
        try:
            plot_img = self.create_plot_image(self.current_id)
            if plot_img is not None:
                self.viewer.add_image(plot_img, name='Growth Profile')
        except Exception as e:
            print(f"Error generating plot: {e}")

        # 3. Layout & Reset
        self.viewer.grid.enabled = True
        self.viewer.grid.shape = (1, 2)
        self.viewer.reset_view()

        # Reset Time Slider to 0
        if self.viewer.dims.ndim > 0:
            current_step = list(self.viewer.dims.current_step)
            current_step[0] = 0
            self.viewer.dims.current_step = tuple(current_step)

    def create_plot_image(self, ID):
        """Generates matplotlib graph with detailed doubling annotations."""
        sc_df = self.df[self.df['ID'] == ID].copy()
        sc_df = sc_df.dropna(subset=['Time Model (hours)', 'Mtb Area Model (µm)'])
        if sc_df.empty:
            return None
        sc_df = sc_df.sort_values(by='Time Model (hours)')

        # Metadata
        meta = sc_df.iloc[0]
        strain = meta.get('Strain', '')
        compound = meta.get('Compound', '')
        conc = meta.get('Concentration', '')
        r2 = meta.get('r2', np.nan)

        # Doubling Data from DF
        d_amounts = meta.get('Doubling Amounts')
        if not isinstance(d_amounts, list):
            d_amounts = []

        # Plot Setup
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        colors = ['#1a9641', '#a6d96a', '#978897', '#d1d1ca', '#f1b6da', '#d02c91'] # piYG

        ax.plot(sc_df['Time Model (hours)'], sc_df['Mtb Area Model (µm)'],
                label='Model', color=colors[-1], lw=2)
        ax.scatter(sc_df['Time Model (hours)'], sc_df['Mtb Area Processed (µm)'],
                   label='Data', color=colors[0], s=10, alpha=0.6)

        # Draw Doubling Lines
        if len(d_amounts) >= 2:
            # Re-calculate exact crossing times for plotting
            doubling_indices = np.clip(
                np.searchsorted(sc_df['Mtb Area Model (µm)'], d_amounts),
                0, len(sc_df) - 1
            )
            doubling_times = sc_df['Time Model (hours)'].iloc[doubling_indices].values

            for i, (t, amt) in enumerate(zip(doubling_times, d_amounts)):
                c = colors[i % len(colors)]
                ax.axvline(x=t, color=c, linestyle=':', alpha=0.6)
                ax.axhline(y=amt, color=c, linestyle=':', alpha=0.6)

                # Annotations (Delta & Interval)
                if i > 0:
                    prev_t = doubling_times[i-1]
                    interval = t - prev_t
                    delta = amt - d_amounts[i-1]

                    # Interval Label
                    ax.text(t, amt + (amt*0.02), f'{interval:.1f}h',
                            ha='right', va='bottom', fontsize=7, color=c, weight='bold')
                    # Delta Label
                    ax.text(t + 0.5, amt - (amt*0.1), f'Δ{delta:.1f}µm²',
                            fontsize=7, color=c, ha='left', va='top', rotation=270, alpha=0.7)

        # Formatting
        title_str = f"ID: {ID}\n{strain} | {compound} {conc} | R2: {r2:.2f}"
        ax.set_title(title_str, fontsize=10)
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Mtb Area (µm²)")
        sns.despine(offset=10)

        # Buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return iio.imread(buf, index=0)

    def split_track(self, viewer):
        """Splits track at current frame, renames tail, and refreshes metrics."""
        # 1. Get Split Time
        try:
            curr_frame = viewer.dims.current_step[0]
            track_df = self.df[self.df['ID'] == self.current_id]
            if track_df.empty:
                return

            start_time = track_df['Time (hours)'].min()
            split_time = start_time + (curr_frame * 0.5)

            print(f"\n✂️ Splitting {self.current_id} at {split_time:.2f}h (Frame {curr_frame})")
        except Exception:
            return

        # 2. Generate New ID (a, b, c...)
        root_id = re.sub(r'[a-z]+$', '', self.current_id)
        existing = self.df[self.df['ID'].str.startswith(root_id)]['ID'].unique()
        letters = [id.replace(root_id, '') for id in existing if id.replace(root_id, '').isalpha()]

        next_char = chr(ord(max(letters)) + 1) if letters else 'b'
        new_id = root_id + next_char

        # 3. Apply Split in DF
        mask = (self.df['ID'] == self.current_id) & (self.df['Time (hours)'] >= split_time)
        if mask.sum() == 0:
            print("⚠️ No data points found after split time.")
            return

        self.df.loc[mask, 'ID'] = new_id
        print(f"✅ Moved {mask.sum()} rows to new ID: {new_id}")

        # 4. Trigger Recalculation
        self.refresh_metrics_for_ids([self.current_id, new_id])
        self.load_sample()

    def refresh_metrics_for_ids(self, id_list):
        """Runs the full analysis pipeline on specific IDs."""
        print(f"🔄 Recalculating metrics for: {id_list}")
        mask = self.df['ID'].isin(id_list)
        if mask.sum() == 0:
            return

        sub_df = self.df[mask].copy()

        # Run Pipeline — relaxed spike threshold for verifier context
        sub_df = process_mtb_area(sub_df, spike_threshold=10.0)
        sub_df = fit_lowess(sub_df)
        sub_df = compute_doubling_metrics(sub_df, min_doubling_time=4.0)

        # Update Main DF
        cols = ['Mtb Area Processed (µm)', 'Time Model (hours)',
                'Mtb Area Model (µm)', 'r2', 'Doubling Amounts', 'Doubling Times']

        # Only update columns that exist
        valid_cols = [c for c in cols if c in sub_df.columns]
        self.df.loc[mask, valid_cols] = sub_df[valid_cols]
        print("✨ Metrics updated.")

    # --- Navigation & Saving ---
    def next_sample(self, viewer):
        self.current_idx += 1
        self.load_sample()

    def prev_sample(self, viewer):
        if self.current_idx > 0:
            self.current_idx -= 1
        self.load_sample()

    def save_df(self, viewer):
        print(f"💾 Saving DataFrame to {self.save_path}...")
        self.df.to_pickle(self.save_path)
        print("✅ Saved.")

    # --- Classification ---
    def update_status(self, col, val):
        self.df.loc[self.df['ID'] == self.current_id, col] = val
        print(f"🏷️ {self.current_id} -> {col}: {val}")

    def mark_transfer(self, v): self.update_status('mtb_origin', 'Transfer')
    def mark_uptake(self, v): self.update_status('mtb_origin', 'Uptake')
    def mark_junk(self, v): self.update_status('mtb_origin', 'Junk')
    def mark_growth(self, v): self.update_status('mtb_origin', 'Growth')
    def mark_edge(self, v): self.update_status('Edge Status', True)

if __name__ == "__main__":
    print("This module is designed to be imported.")
    print("Usage: from macrohet.growth_verifier import GlimpseAnnotator")