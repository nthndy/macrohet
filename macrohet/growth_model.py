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
    area_series = area_series.reset_index(drop=True)
    rolling_mean = area_series.rolling(window=window, min_periods=1).mean()
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
    return cleaned.interpolate()


def process_mtb_area(df, window=10, spike_threshold=2.0):
    cleaned_series = df.groupby("ID")["Mtb Area (\u00b5m)"].apply(
        lambda x: smooth_and_fix(x, window, spike_threshold)
    )
    df = df.copy()
    df["Mtb Area Processed (\u00b5m)"] = cleaned_series.reset_index(
        level=0, drop=True
    )
    return df


def fit_lowess(df, frac=0.25, default_window=10, control_window=5):
    df = df.copy()
    df["Time Model (hours)"] = np.nan
    df["Mtb Area Model (\u00b5m)"] = np.nan
    df["r2"] = np.nan

    for ID in tqdm(df["ID"].unique(), desc="Fitting LOWESS"):
        sc_df = df[df["ID"] == ID]
        window = control_window if "PS0000" in ID else default_window

        time = sc_df["Time (hours)"].values
        pop = sc_df["Mtb Area Processed (\u00b5m)"].values
        if len(time) < 2:
            continue

        model = sm.nonparametric.lowess(endog=pop, exog=time, frac=frac)
        time_model = model[:, 0] - (window / 2) + 0.5
        population_model = np.clip(model[:, 1], 0, None)

        if len(time_model) < len(sc_df):
            pad = len(sc_df) - len(time_model)
            time_model = np.concatenate(
                [
                    np.full(pad // 2, np.nan),
                    time_model,
                    np.full(pad - pad // 2, np.nan),
                ]
            )
            population_model = np.concatenate(
                [
                    np.full(pad // 2, np.nan),
                    population_model,
                    np.full(pad - pad // 2, np.nan),
                ]
            )

        df.loc[df["ID"] == ID, "Time Model (hours)"] = time_model
        df.loc[df["ID"] == ID, "Mtb Area Model (\u00b5m)"] = population_model
        df.loc[df["ID"] == ID, "r2"] = r2_score(
            sc_df["Mtb Area Processed (\u00b5m)"], model[:, 1]
        )

    return df


def compute_doubling_metrics(df, min_area=1.92, r2_threshold=0.7):
    df = df.copy()
    df["Doubling Amounts"] = None
    df["Doubling Times"] = None
    for ID in tqdm(df["ID"].unique(), desc="Doubling Metrics"):
        sc_df = df[df["ID"] == ID].dropna(
            subset=[
                "Time Model (hours)",
                "Mtb Area Model (\u00b5m)",
                "Mtb Area Processed (\u00b5m)",
            ]
        )
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
        N_i = min_val
        while N_i <= max_val:
            N_series.append(N_i)
            N_i *= 2

        if len(N_series) < 2:
            continue

        times = sc_df["Time Model (hours)"]
        model_vals = sc_df["Mtb Area Model (\u00b5m)"]
        doubling_idx = [np.abs(model_vals - val).idxmin() for val in N_series]
        doubling_times = (
            times.loc[doubling_idx].diff().dropna().values.tolist()
        )

        df.loc[df["ID"] == ID, "Doubling Amounts"] = [N_series] * len(sc_df)
        df.loc[df["ID"] == ID, "Doubling Times"] = [doubling_times] * len(
            sc_df
        )

    return df
