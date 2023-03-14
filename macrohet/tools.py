import numpy as np
import pandas as pd


def msd_calc(x1, y1, x2, y2):
    """
    Displacement calculation for cell movement between frames
    """
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def scale_napari_tracks(napari_tracks, scale=6048 / 1200):
    """
    Quick fix for tracking hack:
    Iterates over each track entry from the output of
    btrack.utils.tracks_to_napari() and scales the xy coords up to original
    image size

    Parameters
    ----------
    napari_tracks : array (N, D+1)
        Coordinates for N points in D+1 dimensions. ID,T,Y,X. The first
        axis is the integer ID of the track. D is 3 for planar timeseries only.
    scale : int
        Integer value to scale tracks up by to match original image

    Returns
    ----------
    scaled_tracks : array (N, D+1)
        Scaled coordinates for N points in D+1 dimensions. ID,T,Y,X. The first
        axis is the integer ID of the track. D is 3 for planar timeseries only.

    """

    scaled_tracks = np.zeros((napari_tracks.shape))
    for n, entry in enumerate(napari_tracks):
        y, x = entry[-2], entry[-1]
        scaled_y, scaled_x = y * scale, x * scale
        scaled_entry = [entry[0], entry[1], scaled_y, scaled_x]
        scaled_tracks[n] = scaled_entry

    return scaled_tracks


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
                 'MSD': [msd_calc(track['x'][i - 1],
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
