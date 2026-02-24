import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D


def plot_single_cell_growth(df, ID):
    """
    Visual verification of the LOWESS fit against processed data.
    """
    sc_df = df[df['ID'] == ID].sort_values(by='Time Model (hours)')
    if sc_df.empty:
        return

    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=sc_df, x='Time (hours)', y='Mtb Area Processed (µm)',
        label='Processed Data', s=15, color='#1a9641', alpha=0.6
    )
    sns.lineplot(
        data=sc_df, x='Time Model (hours)', y='Mtb Area Model (µm)',
        label='LOWESS Fit', color='#d02c91', lw=2
    )

    r2 = sc_df['r2'].iloc[0] if 'r2' in sc_df.columns else np.nan
    plt.title(f"Fit Verification | ID: {ID} | r² = {r2:.2f}", weight='bold')
    plt.xlabel("Time (hours)")
    plt.ylabel("Mtb Area (µm²)")
    sns.despine(offset=10)
    plt.tight_layout()
    plt.show()


def plot_single_cell_doubling_times(df, ID):
    """
    Generates a standalone Matplotlib graph for a specific single-cell track ID,
    using robust dynamic baseline logic and custom doubling annotations.
    """
    # 1. Filter and Sort
    sc_df = df[df['ID'] == ID].copy()
    if sc_df.empty:
        print(f"ID {ID} not found in DataFrame.")
        return None

    sc_df = sc_df.sort_values(by='Time Model (hours)')

    # 2. Extract Data & Metadata
    times = sc_df['Time Model (hours)'].values
    area_model = sc_df['Mtb Area Model (µm)'].values
    area_proc = sc_df['Mtb Area Processed (µm)'].values

    meta = sc_df.iloc[0]
    r2 = meta.get('r2', np.nan)
    strain = meta.get('Strain', 'Unknown')
    compound = meta.get('Compound', 'N/A')
    conc = meta.get('Concentration', 'N/A')

    # Create Figure
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

    # --- 3. Live Calculation Logic (Robust) ---

    # A. Helper Function for interpolation
    def find_crossing(target, t_arr, a_arr):
        if not np.any(a_arr >= target):
            return None
        filled_a = np.nan_to_num(a_arr, nan=-np.inf)
        idx = np.argmax(filled_a >= target)
        if idx == 0 and filled_a[0] < target:
            return None
        if idx == 0:
            return t_arr[0]

        t1, t2 = t_arr[idx-1], t_arr[idx]
        a1, a2 = filled_a[idx-1], filled_a[idx]
        if a2 == a1:
            return t1
        return t1 + (t2 - t1) * ((target - a1) / (a2 - a1))

    # B. Determine Start Baseline
    raw_start_area = area_model[0] if not np.isnan(area_model[0]) else 1.92
    baseline = max(1.92, raw_start_area)

    # C. Determine Start Points
    if raw_start_area < baseline:
        t_start = find_crossing(baseline, times, area_model)
    else:
        t_start = times[0]

    if t_start is None:
        crossings_x = []
        crossings_y = []
    else:
        crossings_x = [t_start]
        crossings_y = [baseline]

    # D. Find Subsequent Crossings
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

    # --- 4. Plotting ---
    # Main Curves
    ax.plot(times, area_model, color='#d02c91', lw=2, label='Model')
    ax.scatter(times, area_proc, color='#1a9641', s=10, alpha=0.6, label='Data')

    # Draw Initial Baseline (Floor of first interval)
    if crossings_x:
        ax.axhline(y=baseline, color='lightgrey', linestyle='--', lw=1.5)
        ax.axvline(x=crossings_x[0], color='lightgrey', linestyle='--', lw=1.5)

    # Loop Intervals & Label
    for i in range(1, len(crossings_x)):
        t_prev, t_curr = crossings_x[i-1], crossings_x[i]
        y_prev, y_curr = crossings_y[i-1], crossings_y[i]

        # Draw Grid Lines (Ceiling)
        ax.axhline(y=y_curr, color='lightgrey', linestyle='--', lw=1.5)
        ax.axvline(x=t_curr, color='lightgrey', linestyle='--', lw=1.5)

        # Calculate Deltas
        dt = t_curr - t_prev
        delta_y = y_curr - y_prev

        # Label Logic: Left Justified, Sitting on the "Floor" line
        label_text = f"∆T={dt:.1f}h | ∆Mtb={delta_y:.2f}µm²"

        ax.text(t_prev + 0.2,
                y_prev * 1.02, # Sit on the previous line
                label_text,
                color='#505050', # Dark Grey
                fontsize=8,
                fontweight='bold',
                ha='left',
                va='bottom',
                bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none', alpha=0.7))

    # --- 5. Formatting ---
    title_str = (f"ID: {ID}\n"
                 f"{strain} | {compound} {conc} | R2: {r2:.2f}")

    ax.set_title(title_str, fontsize=10)

    # Custom Legend
    custom_lines = [
        Line2D([0], [0], color='#d02c91', lw=2),
        Line2D([0], [0], color='#1a9641', marker='o', lw=0),
        Line2D([0], [0], color='lightgrey', linestyle='--', lw=1.5)
    ]
    ax.legend(custom_lines, ['Model', 'Data', 'Doubling Grid'], loc='upper left', fontsize=8)

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Mtb Area (µm²)")
    sns.despine(offset=10)

    return fig, ax
