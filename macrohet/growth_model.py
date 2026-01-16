"""growth_model.py

Tools for processing single-cell intracellular Mtb growth data,
including smoothing, fitting, and growth/doubling metrics.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
from tqdm.auto import tqdm

from macrohet.tools import euc_dist


def collate_tracks_to_df(
    tracks,
    expt_ID,
    acq_ID,
    gfp_channel=0,
    mtb_channel=1,
    pixel_to_mum_sq_scale_factor=1.0,
):
    dfs = []
    for track in tqdm(tracks, desc="Processing tracks"):
        t = np.array([p.t for p in track])
        x = np.array([p.x for p in track])
        y = np.array([p.y for p in track])
        area = np.array([p.properties["area"] for p in track])
        mean_intensity = np.stack(
            [p.properties["mean_intensity"] for p in track]
        )
        infected = np.array(
            [p.properties.get("Infected", False) for p in track]
        )
        major_axis = np.array(
            [p.properties["major_axis_length"] for p in track]
        )
        minor_axis = np.array(
            [p.properties["minor_axis_length"] for p in track]
        )
        eccentricity = np.sqrt(1 - ((minor_axis**2) / (major_axis**2)))
        mtb_area_px = np.array(
            [p.properties.get("Mtb area px", np.nan) for p in track]
        )

        d_mtb_area = (
            (mtb_area_px[-1] - mtb_area_px[0]) * pixel_to_mum_sq_scale_factor
            if len(mtb_area_px) > 1
            else 0
        )
        d_mphi_area = (
            (area[-1] - area[0]) * pixel_to_mum_sq_scale_factor
            if len(area) > 1
            else 0
        )
        msd = [
            euc_dist(x[i - 1], y[i - 1], x[i], y[i]) if i > 0 else 0
            for i in range(len(t))
        ]

        d = {
            "Time (hours)": t / 2,
            "x": x,
            "y": y,
            "MSD": msd,
            "Mphi Area (\u00b5m)": area * pixel_to_mum_sq_scale_factor,
            "dMphi Area (\u00b5m)": [d_mphi_area] * len(t),
            "Eccentricity": eccentricity,
            "GFP": mean_intensity[:, gfp_channel],
            "RFP": mean_intensity[:, mtb_channel],
            "Mtb Area (\u00b5m)": mtb_area_px * pixel_to_mum_sq_scale_factor,
            "dMtb Area (\u00b5m)": [d_mtb_area] * len(t),
            "Infection Status": infected,
            "Initial Infection Status": infected[0],
            "Final Infection Status": infected[-1],
            "Cell ID": [track[0].ID] * len(t),
            "Acquisition ID": [acq_ID] * len(t),
            "Experiment ID": [expt_ID] * len(t),
            "Unique ID": [f"{track[0].ID}.{acq_ID[0]}.{acq_ID[1]}"] * len(t),
            "ID": [f"{track[0].ID}.{acq_ID[0]}.{acq_ID[1]}.{expt_ID}"]
            * len(t),
        }

        dfs.append(pd.DataFrame(d))
    return pd.concat(dfs, ignore_index=True)


def smooth_and_fix(area_series, window=10, spike_threshold=2.0):
    """
    Smoothing logic:
    1. Uses center=True to prevent valid jumps from looking like spikes.
    2. Preserves Index to prevent NaN errors when merging back.
    3. Keeps spike_threshold=2.0.
    """
    # 1. Capture the original index so we can restore it later
    original_index = area_series.index

    # 2. Reset index for calculation
    area_series = area_series.reset_index(drop=True)

    # 3. Center=True allows the mean to 'see' the jump coming
    rolling_mean = area_series.rolling(window=window, min_periods=1, center=True).mean()

    cleaned = area_series.copy()
    for i in range(1, len(cleaned) - 1):
        # Strict Threshold Check (2.0)
        if cleaned.iloc[i] > spike_threshold * rolling_mean.iloc[i]:
            cleaned.iloc[i] = np.nan
        # Zero-Bounce Check
        elif (
            cleaned.iloc[i] == 0
            and cleaned.iloc[i - 1] > 0
            and cleaned.iloc[i + 1] > 0
        ):
            cleaned.iloc[i] = np.nan

    # 4. Interpolate to fill gaps
    result = cleaned.interpolate(limit_direction='both')

    # 5. Restore the original index before returning
    result.index = original_index

    return result


def process_mtb_area(df, window=10, spike_threshold=2.0):
    """
    Applies the safe smoothing to the entire DataFrame.
    Includes tqdm progress bar.
    """
    df = df.copy()

    # Initialize tqdm for pandas
    tqdm.pandas(desc="Smoothing Data")

    # Use progress_transform for visibility
    cleaned_series = df.groupby("ID")["Mtb Area (\u00b5m)"].progress_transform(
        lambda x: smooth_and_fix(x, window, spike_threshold)
    )

    df["Mtb Area Processed (\u00b5m)"] = cleaned_series
    return df


def fit_lowess(df, frac=0.25):
    """
    Robust Lowess fitting that uses index alignment to avoid length mismatch errors.
    """
    df = df.copy()
    # Ensure columns exist
    cols = ["Time Model (hours)", "Mtb Area Model (\u00b5m)", "r2"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    for ID in tqdm(df["ID"].unique(), desc="Fitting LOWESS"):
        mask = df["ID"] == ID
        sc_df = df[mask]

        # 1. Filter for valid data only
        # We save the INDICES (valid_idx) to map results back to exact rows later
        valid_df = sc_df.dropna(subset=["Time (hours)", "Mtb Area Processed (\u00b5m)"])

        # Skip if too few points
        if len(valid_df) < 5:
            continue

        # 2. Sort by time (Critical for Lowess)
        valid_df = valid_df.sort_values("Time (hours)")
        valid_idx = valid_df.index

        time = valid_df["Time (hours)"].values
        pop = valid_df["Mtb Area Processed (\u00b5m)"].values

        try:
            # 3. Fit Lowess
            z = sm.nonparametric.lowess(endog=pop, exog=time, frac=frac)
            time_model = z[:, 0]
            area_model = z[:, 1]

            # 4. Assign back using SPECIFIC INDICES
            df.loc[valid_idx, "Time Model (hours)"] = time_model
            df.loc[valid_idx, "Mtb Area Model (\u00b5m)"] = area_model

            # Calculate and assign R2
            score = r2_score(pop, area_model)
            df.loc[mask, "r2"] = score

        except Exception:
            continue

    return df


def compute_doubling_metrics(df, min_area=1.92, r2_threshold=0.7):
    """
    Calculates doubling times and safely assigns them back to the DataFrame
    even if rows were dropped during processing.
    """
    df = df.copy()
    df["Doubling Amounts"] = None
    df["Doubling Times"] = None

    for ID in tqdm(df["ID"].unique(), desc="Doubling Metrics"):
        # 1. Get the Full Mask for this ID
        mask = df["ID"] == ID
        full_group_len = mask.sum()

        # 2. Get Valid Data for Calculation
        sc_df = df[mask].dropna(subset=["Time Model (hours)", "Mtb Area Model (\u00b5m)"])

        if sc_df.empty:
            continue

        # 3. Check R2 (Use the value from the first valid row)
        r2 = sc_df["r2"].iloc[0]
        if r2 < r2_threshold:
            continue

        # 4. Check Growth
        min_val = max(sc_df["Mtb Area Model (\u00b5m)"].min(), min_area)
        max_val = sc_df["Mtb Area Model (\u00b5m)"].max()
        if max_val <= min_val:
            continue

        # 5. Generate Doubling Milestones
        N_series = []
        curr = min_val
        while curr <= max_val:
            N_series.append(curr)
            curr *= 2

        if len(N_series) < 2:
            continue

        # 6. Calculate Times
        times = sc_df["Time Model (hours)"]
        vals = sc_df["Mtb Area Model (\u00b5m)"]

        # Find index of closest value to target
        doubling_idx = [np.abs(vals - target).idxmin() for target in N_series]
        doubling_times = times.loc[doubling_idx].diff().dropna().values.tolist()

        # 7. Safe Assignment
        # We create a Series matching the index of the FULL group (mask)
        df.loc[mask, "Doubling Amounts"] = pd.Series(
            [N_series] * full_group_len, index=df[mask].index
        )

        df.loc[mask, "Doubling Times"] = pd.Series(
            [doubling_times] * full_group_len, index=df[mask].index
        )

    return df