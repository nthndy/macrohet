"""growth_model.py

Tools for processing single-cell intracellular Mtb growth data,
including smoothing, fitting, and growth/doubling metrics.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
from tqdm.auto import tqdm


def euc_dist(x1, y1, x2, y2):
    """Euclidean distance displacement calculation for cell movement between frames."""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def collate_tracks_to_df(
    tracks,
    expt_ID="EXP1",
    acq_ID=(0, 0),
    gfp_channel=0,
    mtb_channel=1,
    pixel_to_mum_sq_scale_factor=1.0,
):
    dfs = []

    # Intelligently parse the input based on its type
    def _track_generator():
        # A. Handle Zarr tuple: (track_array, features_dict)
        if isinstance(tracks, tuple) and len(tracks) == 2 and isinstance(tracks[0], np.ndarray):
            zarr_track_array, zarr_features = tracks
            unique_ids = np.unique(zarr_track_array[:, 0])
            for tid in unique_ids:
                mask = zarr_track_array[:, 0] == tid
                t = zarr_track_array[mask, 1]
                y = zarr_track_array[mask, 2]
                x = zarr_track_array[mask, 3]
                props = {k: v[mask] for k, v in zarr_features.items()}
                yield int(tid), t, x, y, props

        # B. Handle Legacy btrack Tracklet list
        else:
            for track in tracks:
                yield track.ID, track.t, track.x, track.y, track.properties

    # Estimate total for tqdm progress bar
    total_tracks = len(np.unique(tracks[0][:, 0])) if isinstance(tracks, tuple) else len(tracks)

    for track_id, t, x, y, props in tqdm(_track_generator(), total=total_tracks, desc="Processing tracks"):

        t = np.array(t)
        x = np.array(x)
        y = np.array(y)

        area = np.array(props.get("area", np.zeros(len(t))))
        major_axis = np.array(props.get("major_axis_length", np.zeros(len(t))))
        minor_axis = np.array(props.get("minor_axis_length", np.zeros(len(t))))

        # Safe eccentricity calculation
        eccentricity = np.where(
            major_axis > 0,
            np.sqrt(np.clip(1 - (minor_axis**2 / np.maximum(major_axis**2, 1e-9)), 0, 1)),
            0.0
        )

        mtb_area_px = np.array(props.get("Mtb area px", props.get("mtb_area_px", np.full(len(t), np.nan))))
        infected = np.array(props.get("Infected", props.get("infected", np.zeros(len(t), dtype=bool))))

        if "mean_intensity" in props:
            mean_intensity = np.stack(props["mean_intensity"])
            gfp = mean_intensity[:, gfp_channel]
            rfp = mean_intensity[:, mtb_channel]
        elif f"mean_intensity-{gfp_channel}" in props:
            gfp = np.array(props[f"mean_intensity-{gfp_channel}"])
            rfp = np.array(props[f"mean_intensity-{mtb_channel}"])
        else:
            gfp = np.array(props.get("gfp_intensity", np.zeros(len(t))))
            rfp = np.array(props.get("rfp_intensity", np.zeros(len(t))))

        d_mtb_area = (mtb_area_px[-1] - mtb_area_px[0]) * pixel_to_mum_sq_scale_factor if len(mtb_area_px) > 1 else 0
        d_mphi_area = (area[-1] - area[0]) * pixel_to_mum_sq_scale_factor if len(area) > 1 else 0

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
            "GFP": gfp,
            "RFP": rfp,
            "Mtb Area (\u00b5m)": mtb_area_px * pixel_to_mum_sq_scale_factor,
            "dMtb Area (\u00b5m)": [d_mtb_area] * len(t),
            "Infection Status": infected,
            "Initial Infection Status": infected[0] if len(infected) > 0 else False,
            "Final Infection Status": infected[-1] if len(infected) > 0 else False,
            "Cell ID": [track_id] * len(t),
            "Acquisition ID": [acq_ID] * len(t),
            "Experiment ID": [expt_ID] * len(t),
            "Unique ID": [f"{track_id}.{acq_ID[0]}.{acq_ID[1]}"] * len(t),
            "ID": [f"{track_id}.{acq_ID[0]}.{acq_ID[1]}.{expt_ID}"] * len(t),
        }

        dfs.append(pd.DataFrame(d))

    return pd.concat(dfs, ignore_index=True)


def smooth_and_fix(area_series, window=10, spike_threshold=10.0):
    """
    Smoothing logic:
    1. Uses center=True to prevent valid jumps from looking like spikes.
    2. Preserves Index to prevent NaN errors when merging back.
    3. Uses relaxed spike_threshold=10.0 to prevent valid biological jumps being dropped.
    """
    original_index = area_series.index
    area_series = area_series.reset_index(drop=True)
    rolling_mean = area_series.rolling(window=window, min_periods=1, center=True).mean()

    cleaned = area_series.copy()
    for i in range(1, len(cleaned) - 1):
        if cleaned.iloc[i] > spike_threshold * rolling_mean.iloc[i]:
            cleaned.iloc[i] = np.nan
        elif (
            cleaned.iloc[i] == 0
            and cleaned.iloc[i - 1] > 0
            and cleaned.iloc[i + 1] > 0
        ):
            cleaned.iloc[i] = np.nan

    result = cleaned.interpolate(limit_direction='both')
    result.index = original_index

    return result


def process_mtb_area(df, window=10, spike_threshold=10.0):
    """
    Applies the safe smoothing to the entire DataFrame.
    Includes tqdm progress bar.
    """
    df = df.copy()
    tqdm.pandas(desc="Smoothing Data")

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
    cols = ["Time Model (hours)", "Mtb Area Model (\u00b5m)", "r2"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    for ID in tqdm(df["ID"].unique(), desc="Fitting LOWESS"):
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
            time_model = z[:, 0]
            area_model = z[:, 1]

            df.loc[valid_idx, "Time Model (hours)"] = time_model
            df.loc[valid_idx, "Mtb Area Model (\u00b5m)"] = area_model

            score = r2_score(pop, area_model)
            df.loc[mask, "r2"] = score

        except Exception:
            continue

    return df


def _find_crossing(target, t_arr, a_arr):
    """
    Helper function using linear interpolation to find the exact
    time a model crosses a target area threshold.
    """
    valid_mask = ~np.isnan(a_arr)
    clean_a = a_arr[valid_mask]

    if not np.any(clean_a >= target):
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

    fraction = (target - a1) / (a2 - a1)
    return t1 + (t2 - t1) * fraction


def compute_doubling_metrics(df, min_area=1.92, r2_threshold=0.7):
    """
    Calculates robust doubling times AND amounts dynamically, starting from
    the actual interpolated crossing of the baseline.
    Filters by R-squared, but allows sub-physiological intervals to remain.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing Mtb area models and tracking data.
    min_area : float, optional
        The minimum initial Mtb area (µm) to begin doubling calculations (default is 1.92).
    r2_threshold : float, optional
        The minimum acceptable R-squared value for the growth model (default is 0.7).

    Returns
    -------
    pandas.DataFrame
    """
    df = df.copy()

    # Initialize columns as object type to safely store lists
    df['Doubling Times'] = np.nan
    df['Doubling Times'] = df['Doubling Times'].astype(object)
    df['Doubling Amounts'] = np.nan
    df['Doubling Amounts'] = df['Doubling Amounts'].astype(object)

    for ID in tqdm(df["ID"].unique(), desc="Doubling Metrics"):
        mask = df["ID"] == ID
        full_group_len = mask.sum()

        sc_df = df[mask].dropna(subset=["Time Model (hours)", "Mtb Area Model (\u00b5m)"])
        if sc_df.empty:
            continue

        r2 = sc_df["r2"].iloc[0]
        if r2 < r2_threshold:
            continue

        sc_df = sc_df.sort_values(by='Time Model (hours)')
        times = sc_df['Time Model (hours)'].values
        area_model = sc_df['Mtb Area Model (\u00b5m)'].values

        if len(times) < 2:
            continue

        start_area = area_model[0] if not np.isnan(area_model[0]) else min_area
        baseline = max(min_area, start_area)

        # Dynamic grid generation (Baseline -> 2x -> 4x -> 8x...)
        grid = [baseline * (2**i) for i in range(1, 6)]

        calc_intervals = []
        calc_amounts = [round(baseline, 2)]
        prev_time = times[0]

        if start_area < baseline:
            t_start_real = _find_crossing(baseline, times, area_model)
            if t_start_real is not None:
                prev_time = t_start_real
            else:
                continue

        for target in grid:
            t_cross = _find_crossing(target, times, area_model)

            if t_cross is not None:
                # Forward-time check (prevents interval noise, replaces the min_time filter)
                if t_cross >= prev_time:
                    dt = t_cross - prev_time
                    calc_intervals.append(round(dt, 1))
                    calc_amounts.append(round(target, 2))
                    prev_time = t_cross
                else:
                    break
            else:
                break

        if not calc_intervals:
            continue

        # Create properly formatted object arrays for the pandas DataFrame assignment
        dt_series = np.empty(full_group_len, dtype=object)
        dt_series[:] = [calc_intervals] * full_group_len

        amt_series = np.empty(full_group_len, dtype=object)
        amt_series[:] = [calc_amounts] * full_group_len

        df.loc[mask, 'Doubling Times'] = dt_series
        df.loc[mask, 'Doubling Amounts'] = amt_series

    return df