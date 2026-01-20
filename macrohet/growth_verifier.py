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
import statsmodels.api as sm
from sklearn.metrics import r2_score

# ==========================================
# PART 1: ROBUST PROCESSING LOGIC
# ==========================================


#### TO-DO: ensure smooth and process mtb signals are same as in main modules
#### and load from growth_model.py rather than in here


def smooth_and_fix_relaxed(area_series, window=10, spike_threshold=10.0):
    """
    Smooths data using a CENTERED rolling mean to detect spikes.
    Allows for large biological jumps (10x) but removes single-pixel noise.
    """
    original_index = area_series.index
    area_series = area_series.reset_index(drop=True)

    # center=True allows the mean to 'see' the jump coming
    rolling_mean = area_series.rolling(window=window, min_periods=1, center=True).mean()

    cleaned = area_series.copy()
    for i in range(1, len(cleaned) - 1):
        # Relaxed Threshold Check
        if cleaned.iloc[i] > spike_threshold * rolling_mean.iloc[i]:
            cleaned.iloc[i] = np.nan
        # Zero-Bounce Check
        elif (
            cleaned.iloc[i] == 0
            and cleaned.iloc[i - 1] > 0
            and cleaned.iloc[i + 1] > 0
        ):
            cleaned.iloc[i] = np.nan

    result = cleaned.interpolate(limit_direction='both')
    result.index = original_index
    return result

def process_mtb_area_relaxed(df, window=10, spike_threshold=10.0):
    """Applies relaxed smoothing to the DataFrame group-wise."""
    df = df.copy()
    # We use transform to keep the index aligned strictly
    cleaned_series = df.groupby("ID")["Mtb Area (\u00b5m)"].transform(
        lambda x: smooth_and_fix_relaxed(x, window, spike_threshold)
    )
    df["Mtb Area Processed (\u00b5m)"] = cleaned_series
    return df

def fit_lowess(df, frac=0.25):
    """Robust Lowess fitting with explicit index alignment."""
    df = df.copy()
    cols = ["Time Model (hours)", "Mtb Area Model (\u00b5m)", "r2"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    # Iterate unique IDs (using list to avoid tqdm spam in GUI mode)
    for ID in df["ID"].unique():
        mask = df["ID"] == ID
        sc_df = df[mask]

        valid_df = sc_df.dropna(subset=["Time (hours)", "Mtb Area Processed (\u00b5m)"])
        if len(valid_df) < 5:
            continue

        valid_df = valid_df.sort_values("Time (hours)")
        valid_idx = valid_df.index

        time = valid_df["Time (hours)"].values
        pop = valid_df["Mtb Area Processed (\u00b5m)"].values

        try:
            z = sm.nonparametric.lowess(endog=pop, exog=time, frac=frac)
            df.loc[valid_idx, "Time Model (hours)"] = z[:, 0]
            df.loc[valid_idx, "Mtb Area Model (\u00b5m)"] = z[:, 1]
            df.loc[mask, "r2"] = r2_score(pop, z[:, 1])
        except Exception:
            continue
    return df

def compute_doubling_metrics(df, min_area=1.92, r2_threshold=0.7, min_doubling_time=4.0):
    """Calculates doubling metrics, filtering out impossible speeds (<4h)."""
    df = df.copy()
    df["Doubling Amounts"] = None
    df["Doubling Times"] = None

    for ID in df["ID"].unique():
        mask = df["ID"] == ID
        full_group_len = mask.sum()

        sc_df = df[mask].dropna(subset=["Time Model (hours)", "Mtb Area Model (\u00b5m)"])
        if sc_df.empty:
            continue

        if sc_df["r2"].iloc[0] < r2_threshold:
            continue

        min_val = max(sc_df["Mtb Area Model (\u00b5m)"].min(), min_area)
        max_val = sc_df["Mtb Area Model (\u00b5m)"].max()
        if max_val <= min_val:
            continue

        N_series = []
        curr = min_val
        while curr <= max_val:
            N_series.append(curr)
            curr *= 2

        if len(N_series) < 2:
            continue

        times = sc_df["Time Model (hours)"]
        vals = sc_df["Mtb Area Model (\u00b5m)"]

        doubling_idx = [np.abs(vals - target).idxmin() for target in N_series]
        dt_intervals = np.diff(times.loc[doubling_idx].values)

        # Filter noise
        valid_amounts = [N_series[0]]
        valid_intervals = []
        for i, dt in enumerate(dt_intervals):
            if dt >= min_doubling_time:
                valid_intervals.append(dt)
                valid_amounts.append(N_series[i+1])

        if not valid_intervals:
            continue

        df.loc[mask, "Doubling Amounts"] = pd.Series([valid_amounts] * full_group_len, index=df[mask].index)
        df.loc[mask, "Doubling Times"] = pd.Series([valid_intervals] * full_group_len, index=df[mask].index)

    return df

# ==========================================
# PART 2: THE ANNOTATOR GUI CLASS
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

        # Run Pipeline
        sub_df = process_mtb_area_relaxed(sub_df)
        sub_df = fit_lowess(sub_df)
        sub_df = compute_doubling_metrics(sub_df)

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