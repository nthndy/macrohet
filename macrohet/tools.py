import math

import numpy as np
import pandas as pd

from macrohet import dataio


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
