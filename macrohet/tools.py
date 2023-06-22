import math

import numpy as np
import pandas as pd
from skimage.morphology import binary_erosion
from tqdm.auto import tqdm

from macrohet import dataio


def create_track_dictionary(track, info, key):
    """
    Create a dictionary of track information for a single track.

    Parameters:
    - track (dict): Track information dictionary
    - info (Series): Assay layout info for the track

    Returns:
    - dict: Dictionary containing track information
    """
    # Raw MTB values (interpolated)
    raw_mtb_values = pd.Series(track['mean_intensity'][:, 1]).interpolate(method='linear')

    # Raw GFP values (interpolated)
    raw_gfp = pd.Series(track['mean_intensity'][:, 0]).interpolate(method='linear')

    # Thresholded MTB values (interpolated)
    mtb_values = pd.Series(track['mean_intensity'][:, 2]).interpolate(method='linear')

    # Smoothed MTB signal using a rolling window of 4 with median values,
    # then backfilling missing values at the start
    mtb_smooth = np.array(mtb_values.rolling(window=4).median().interpolate(method='backfill'))

    # Interpolate other variables using appropriate methods
    minor_axis_length = pd.Series(track['minor_axis_length']).interpolate(method='linear')
    major_axis_length = pd.Series(track['major_axis_length']).interpolate(method='linear')

    # Interpolate infection status using a combination of approaches
    infection_status = pd.Series(track['Infected'])

    # If the first value is missing, assign it the closest infection value
    if pd.isnull(infection_status.iloc[0]):
        infection_status.iloc[0] = infection_status.iloc[infection_status.first_valid_index()]

    # Fill subsequent missing values with the previous value
    infection_status = infection_status.fillna(method='ffill')

    # Interpolate area based on linear method
    area = pd.Series(track['area']).interpolate(method='linear')

    # Compile single track dictionary of info
    d = {
        'Time (hours)': track['t'],
        'x': track['x'],
        'y': track['y'],
        'x scaled': [track['x'][i] * 5.04 for i, x in enumerate(track['x'])],
        'y scaled': [track['y'][i] * 5.04 for i, y in enumerate(track['y'])],
        'Infection status': track['Infected'],
        'Initial infection status': track['Infected'][0],
        'Final infection status': track['Infected'][-1],
        'Area': track['area'],
        'Intracellular mean Mtb content': raw_mtb_values,
        'Intracellular thresholded Mtb content': mtb_values,
        'Intracellular thresholded Mtb content smooth': mtb_smooth,
        'Macroph. GFP expression': raw_gfp,
        'delta Mtb raw': [np.array(mtb_values)[-1] - np.array(mtb_values)[0] for i in range(len(track))],
        'delta Mtb max raw': [(max(mtb_values) - min(mtb_values)) * (1 if np.argmax(mtb_values) > np.argmin(mtb_values) else -1) for i in range(len(track))],
        'delta Mtb max smooth': [(max(mtb_smooth) - min(mtb_smooth)) * (1 if np.argmax(mtb_smooth) > np.argmin(mtb_smooth) else -1) for i in range(len(track))],
        'delta Mtb max fold-change': [max(mtb_smooth) / min(mtb_smooth[mtb_smooth > 0]) * (1 if np.argmax(mtb_smooth) > np.argmin(mtb_smooth) else -1) if np.any(mtb_smooth > 0) else 0 for i in range(len(track))],
        'delta Mtb max fold-change normalised mean area': [(max(mtb_smooth) / min(mtb_smooth[mtb_smooth > 0]) * (1 if np.argmax(mtb_smooth) > np.argmin(mtb_smooth) else -1)) / np.mean(area) if np.any(mtb_smooth > 0) else 0 for i in range(len(track))],
        'delta Mtb max fold-change normalised max area': [(max(mtb_smooth) / min(mtb_smooth[mtb_smooth > 0]) * (1 if np.argmax(mtb_smooth) > np.argmin(mtb_smooth) else -1)) / np.max(area) if np.any(mtb_smooth > 0) else 0 for i in range(len(track))],
        'delta Mtb/dt': np.polyfit(np.arange(len(mtb_smooth)), mtb_smooth, 1)[0],
        'Eccentricity': np.sqrt(1 - ((minor_axis_length ** 2) / (major_axis_length ** 2))),
        'MSD': [euc_dist(track['x'][i - 1], track['y'][i - 1], track['x'][i], track['y'][i]) if i != 0 else 0 for i in range(0, len(track))],
        'Strain': [info['Strain'] for i in range(len(track['t']))],
        'Compound': [info['Compound'] for i in range(len(track['t']))],
        'Concentration': [info['ConcentrationEC'] for i in range(len(track['t']))],
        'Cell ID': [track.ID for i in range(len(track['t']))],
        'Acquisition ID': [key for i in range(len(track['t']))],
        'Unique ID': [f'{track.ID}.{key[0]}.{key[1]}' for i in range(len(track['t']))]}

    return d


def instance_to_semantic(instance_image):
    """
    Quick function to change instance segmentation map to semantic segmentation
    """

    # Get unique labels from the instance image
    unique_labels = np.unique(instance_image)

    # Create a blank semantic segmentation map
    semantic_map = np.zeros_like(instance_image, dtype=np.uint8)

    # Assign unique labels to the semantic map preserving boundaries
    for label in tqdm(unique_labels[1:]):
        segment = instance_image == label
        eroded_segment = binary_erosion(segment, footprint=np.ones((5, 5)))
        semantic_map[eroded_segment] = 1
        # set background to zero
        semantic_map[instance_image == 0] = 0
        semantic_map = semantic_map.astype('i2')

    return semantic_map


def euc_dist(x1, y1, x2, y2):
    """
    Euclidean distance diplacement calculation for cell movement between frames

    Parameters
    ----------
    x1, y1, x2, y2 : float
        Coordinates for 2 cells in 2 dimensions at 2 different time points.

    Returns
    ----------
    euc_dist : float
        The Euclidean distance between the two cells

    """

    euc_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    return euc_dist


def calc_eccentricity(major_axis, minor_axis):
    """
    Calculates the eccentricity of an object given its major and minor axis
    lengths

    Parameters
    ----------
    major_axis, minor_axis : float
        Scalar values for major and minor axis of object in question

    Returns
    ----------
    eccentricity : float
        The eccentricity of the object
    """

    if major_axis < minor_axis:
        major_axis, minor_axis = minor_axis, major_axis  # swap if major axis is smaller
    eccentricity = math.sqrt(1 - (minor_axis**2 / major_axis**2))

    return eccentricity


def track_euc_dist(track):
    """
    Calculate the Euclidean distance between frames for a track over all frames

    Parameters
    ----------
    track : btrack.btypes.Tracklet
        btrack track of interest

    Returns
    ----------
    euc_dist : list
        List of Euclidean distance between frames for track of interest
    """
    # first convert it to a df
    track = dataio.track_to_df(track)
    # now calculate the diff between rows
    dxs = track['x'].diff()
    dys = track['y'].diff()
    # now calculate the Euclidean distance
    euc_dist = [np.sqrt(dxs[i]**2 + dys[i]**2) for i in range(1, len(track))]

    return euc_dist


def compile_multi_track_df(tracks_dict, assay_layout, track_len=75):
    """
    Iterates over many tracks stored in dictionary format and returns a df with
    additional features calculated

    Parameters
    ----------
    tracks_dict : dict()
        A dictionary containing different sets of tracks from different expts
    track_len : int
        Optional input to only store tracks of a set length
    """

    # list of track info dfs
    dfs = list()
    # empty dictionary for filtered tracks
    filtered_tracks = dict()
    # iterate over all tracks
    for key in tracks_dict.keys():
        # extract tracks only with max length
        filtered_tracks[key] = [track for track in tracks_dict[key]
                                if len(track) == track_len]
        # iterate over full length tracks
        for track in filtered_tracks[key]:
            # get info for assay layout
            info = assay_layout.loc[key]
            # compile single track dictionary of info
            d = {'Time (hours)': track['t'],
                 'x': track['x'],
                 'y': track['y'],
                 'Area': track['area'],
                 'Intracellular Mtb content': track['mean_intensity-1'],
                 'Mean Mtb content': [np.nanmean(track['mean_intensity-1'])
                                      for i in range(len(track['t']))],
                 'Macroph. GFP expression': track['mean_intensity-0'],
                 'Eccentricity': np.sqrt(1 - ((track['minor_axis_length']**2)
                                              / (track['major_axis_length']**2))),
                 'Interframe displacement': [euc_dist(track['x'][i - 1],
                                                      track['y'][i - 1],
                                                      track['x'][i],
                                                      track['y'][i])
                                             if i != 0 else 0
                                             for i in range(0, len(track))],
                 'Strain': [info['Strain'] for i in range(len(track['t']))],
                 'Compound': [info['Compound'] for i in range(len(track['t']))],
                 'Concentration': [info['ConcentrationEC']
                                   for i in range(len(track['t']))],
                 'Cell ID': [track.ID for i in range(len(track['t']))],
                 'Acquisition ID': [key for i in range(len(track['t']))]}
            # append df to list of dfs
            dfs.append(pd.DataFrame(d))
    # concat single track dfs into big df
    df = pd.concat(dfs, ignore_index=True)
    # interpolate missing values as sometimes segmentation drops result in NaN
    df.interpolate(inplace=True)

    return df
