import math

import numpy as np
import pandas as pd
from skimage.morphology import binary_erosion
from skimage.transform import downscale_local_mean, resize
from tqdm.auto import tqdm

from macrohet import dataio


def upscale_labels_post_manual_annotation(labels, scale_factor):
    """
    Upscales labels after manual annotation to restore to original size.

    Parameters:
    labels (numpy.ndarray): The input labels to be upscaled.
    scale_factor (int): The scale factor for upscaling the labels.

    Returns:
    numpy.ndarray: The upscaled labels.

    This function takes in manually annotated labels and a scale factor and performs
    upscaling to restore the labels to their original size. The scale factor determines
    how much to increase the dimensions of the labels.

    Note:
    - The input labels should be in the form of a binary mask or integer-valued image.

    Example:
    ```
    # Upscale the labels with a scale factor of 2
    upscaled_labels = upscale_labels_post_manual_annotation(labels, scale_factor=2)
    ```

    """
    # Upscale the labels using resize
    upscaled_labels = resize(labels, (labels.shape[0] * scale_factor,
                                      labels.shape[1] * scale_factor),
                             anti_aliasing=False, order=0, preserve_range=True)

    return upscaled_labels


def downscale_images_for_manual_annotation(image, labels, scale_factor):
    """
    Downscale an image and its corresponding labels for manual annotation.

    Parameters:
        image (ndarray): The original image.
        labels (ndarray): The original labels.
        scale_factor (int): The scale factor for downsampling.

    Returns:
        tuple: A tuple containing the downsampled image and downsampled labels.

    This function downscales an image and its corresponding labels to a lower resolution
    to facilitate manual annotation. The downsampling is performed using the
    `downscale_local_mean` function from the `skimage.transform` module.

    The image and labels are downsampled by the specified scale factor, which represents
    the factor by which the image and labels are reduced in size.

    The downscaled labels are rounded to the nearest integer to ensure they remain valid
    pixel labels.

    Note:
    - The image and labels should be NumPy arrays.
    - The image and labels should have the same dimensions.

    Example:
    ```
    import numpy as np
    from skimage.transform import downscale_local_mean

    # Assuming 'image' and 'labels' are your original image and labels
    scale_factor = 4

    # Downscale the image and labels for manual annotation
    downsampled_image, downsampled_labels = downscale_images_for_manual_annotation(image, labels, scale_factor)
    ```
    """
    # Downscale the image using the 'downscale_local_mean' function
    downsampled_image = downscale_local_mean(image, (scale_factor, scale_factor))

    # Downscale the labels using the 'downscale_local_mean' function
    downsampled_labels = downscale_local_mean(labels.astype(float), (scale_factor, scale_factor))

    # Round the downsampled labels to the nearest integer
    downsampled_labels = np.round(downsampled_labels).astype(int)

    return downsampled_image, downsampled_labels


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
