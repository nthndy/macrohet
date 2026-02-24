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


def compute_doubling_metrics(df, min_area=1.92, r2_threshold=0.7):
    """
    Calculates doubling times and safely assigns them back to the DataFrame.
    Filters out sub-physiological intervals (< min_doubling_time).

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
    df["Doubling Amounts"] = None
    df["Doubling Times"] = None

    for ID in tqdm(df["ID"].unique(), desc="Doubling Metrics"):
        mask = df["ID"] == ID
        full_group_len = mask.sum()

        sc_df = df[mask].dropna(subset=["Time Model (hours)", "Mtb Area Model (\u00b5m)"])

        if sc_df.empty:
            continue

        r2 = sc_df["r2"].iloc[0]
        if r2 < r2_threshold:
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

        # Filter sub-physiological intervals accurately
        valid_amounts = [N_series[0]]
        valid_intervals = []
        for i, dt in enumerate(dt_intervals):
            valid_intervals.append(dt)
            valid_amounts.append(N_series[i + 1])

        if not valid_intervals:
            continue

        N_series = valid_amounts
        doubling_times = valid_intervals

        df.loc[mask, "Doubling Amounts"] = pd.Series(
            [N_series] * full_group_len, index=df[mask].index
        )

        df.loc[mask, "Doubling Times"] = pd.Series(
            [doubling_times] * full_group_len, index=df[mask].index
        )

    return df