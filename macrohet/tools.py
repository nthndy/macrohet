import math

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion
from skimage.measure import regionprops
from skimage.morphology import label
from tqdm.auto import tqdm

from macrohet import dataio


class ImageDimensionError(Exception):
    def __init__(self, expected_dimensionality, received_dimensionality):
        message = f"Invalid image dimensionality. Expected {expected_dimensionality}-dimensional image, but received {received_dimensionality}-dimensional image."
        super().__init__(message)


def measure_mtb_area(track, masks, rfp_images, threshold=480, scale_factor=5.04, image_resolution=1.4949402023919043e-07):
    """
    Measures the area of a region (presumably microtubule) in each frame of an image sequence.

    Parameters:
    - track (object): An object containing tracking information with attributes 't', 'x', 'y', and 'ID'.
    - masks (array): A numpy array representing the segmented masks of the images.
    - rfp_images (array): A numpy array of RFP (red fluorescent protein) images.
    - threshold (int): The intensity threshold for considering a pixel as part of the microtubule region.
    - scale_factor (float): Factor for scaling the coordinates from the track object.
    - image_resolution (float): The resolution of the images in pixels per meter.

    Returns:
    - mtb_areas (list): A list of areas (in micrometers squared) of the microtubule region for each frame.

    The function iterates over each frame specified in the track object, scales the coordinates,
    and selects the corresponding mask. If a mask exists at the specified coordinates, it calculates
    the area of the region with intensity above the threshold in the RFP image. The area is then converted
    from pixels to micrometers squared using the image resolution.
    """

    mtb_areas = []
    for t, x, y in tqdm(zip(track.t, track.x, track.y),
                        total=len(track),
                        desc=f'Calculating mtb area for every frame in track: {track.ID}'):
        # Scale coordinates
        x, y = int(x * scale_factor), int(y * scale_factor)

        # Select the corresponding frame from the masks
        frame = masks[t, ...]

        # Check to see if mask exists at the specified coordinates
        if frame[y, x]:
            # Select the specific cell of interest based on the mask
            mask = frame == frame[y, x]

            # Apply mask to the corresponding RFP frame
            rfp_frame = rfp_images[t]
            masked_image = rfp_frame * mask

            # Apply threshold to identify microtubule region
            thresholded_masked_image = masked_image >= threshold

            # Calculate the area of the microtubule region
            mtb_area = np.sum(thresholded_masked_image)

            # Convert the resolution to pixels per micrometer
            resolution_micrometers_per_pixel = image_resolution * 1_000_000

            # Convert area from pixels to micrometers squared
            mtb_area = mtb_area * (resolution_micrometers_per_pixel ** 2)

            # Append the calculated area to the list
            mtb_areas.append(mtb_area)
        else:
            # Append 0 if no mask exists at the specified coordinates
            mtb_areas.append(0)

    return mtb_areas


def remove_small_segments(mask_stack, threshold_size=1000):
    """
    Remove small segments from a stack of binary masks.

    This function iterates over a stack of binary masks (mask_stack) representing segmented objects and removes
    small segments whose total area is less than the specified threshold_size.

    Parameters:
        mask_stack (numpy.ndarray): A 3D NumPy array containing a stack of binary masks.
                                   Each 2D mask represents segmented objects with labeled regions.
                                   Non-zero values in each mask indicate different segments.
        threshold_size (int, optional): The minimum area (in pixels) for a segment to be considered significant
                                        and not removed. Segments with an area less than threshold_size will be
                                        set to 0 (removed). Default value is 1000.

    Returns:
        numpy.ndarray: A modified version of the input mask_stack with small segments removed.

    Note:
        This function modifies the input mask_stack in place. If you want to preserve the original data,
        make a copy of the mask_stack before calling this function.

    Examples:
        # Example usage:
        import numpy as np
        from skimage.measure import regionprops

        # Assuming you have a 3D mask_stack and want to remove segments smaller than 500 pixels.
        modified_mask_stack = remove_small_segments(mask_stack, threshold_size=500)
    """
    # Check the dimensionality of the input mask_stack
    if len(mask_stack.shape) != 3:
        raise ImageDimensionError(expected_dimensionality=3, received_dimensionality=len(mask_stack.shape))

    # Iterate over each frame (2D mask) in the mask_stack
    for n, frame in tqdm(enumerate(mask_stack), desc='Iterating over frames', total=len(mask_stack)):
        # Get coordinates of segments with area less than threshold_size using regionprops
        coords = [props.coords for props in regionprops(frame) if props.area < threshold_size]

        # Iterate over each segment's coordinates and set them to 0 (remove small segments)
        for segment_coords in coords:
            for x, y in segment_coords:
                frame[x, y] = 0

    return mask_stack


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
        'Technical replicate': [info['Technical replicate'] for i in range(len(track['t']))],
        'Cell ID': [track.ID for i in range(len(track['t']))],
        'Acquisition ID': [key for i in range(len(track['t']))],
        'Unique ID': [f'{track.ID}.{key[0]}.{key[1]}' for i in range(len(track['t']))]}

    return d


def instance_to_semantic(instance_image):
    """
    Quick function to change instance segmentation map to semantic segmentation
    """

    # check dimensionality of image
    if len(instance_image.shape) == 3:

        # create empty list to store semantic frames in
        semantic_stack = list()

        # iterate over each frame
        for frame in tqdm(instance_image, total=len(instance_image), desc='Iterating over frames'):

            # Get unique labels from the instance image
            unique_labels = np.unique(frame)

            # Create a blank semantic segmentation map
            semantic_map = np.zeros_like(frame, dtype=np.uint8)

            # Set background to zero
            semantic_map[frame == 0] = 0

            # Assign unique labels to the semantic map preserving boundaries
            for sc_label in tqdm(unique_labels[1:], total=len(unique_labels) - 1,
                                 desc='Iterating over segments',
                                 leave=False):

                # Get single cell label
                segment = frame == sc_label

                # Erode segment so that it doesn't touch neighbors
                eroded_segment = binary_erosion(segment, structure=np.ones((5, 5)))

                # Relabel segment semantically
                semantic_map[eroded_segment] = 1

            # Append results to stack
            semantic_stack.append(semantic_map)

        # Convert from list to stack
        semantic_map = np.stack(semantic_stack, axis=0)

    # if it's just a frame then do not iterate over
    elif len(instance_image.shape) == 2:

        # Get unique labels from the instance image
        unique_labels = np.unique(instance_image)

        # Create a blank semantic segmentation map
        semantic_map = np.zeros_like(instance_image, dtype=np.uint8)

        # Set background to zero
        semantic_map[instance_image == 0] = 0

        # Assign unique labels to the semantic map preserving boundaries
        for sc_label in tqdm(unique_labels[1:], total=len(unique_labels) - 1,
                             desc='Iterating over segments',
                             leave=False):

            # Get single cell label
            segment = instance_image == sc_label

            # Erode segment so that it doesn't touch neighbors
            eroded_segment = binary_erosion(segment, structure=np.ones((5, 5)))

            # Relabel segment semantically
            semantic_map[eroded_segment] = 1

    else:
        raise ImageDimensionError(expected_dimensionality="2 or 3", received_dimensionality=len(instance_image.shape))

    return semantic_map


def semantic_to_instance(semantic_image):
    """
    Quick function to change semantic segmentation map to instance segmentation
    """

    # check dimensionality of image
    if len(semantic_image.shape) == 3:

        # create empty list for stack
        instance_stack = list()

        # iterate over frames in stack
        for frame in tqdm(semantic_image, total=len(semantic_image), desc='Iterating over frames'):

            # Get unique labels from the instance image
            instance_image = label(frame)

            # append to stack
            instance_stack.append(instance_image)

        # stack together
        instance_image = np.stack(instance_stack, axis=0)

    # if it's just a frame then do not iterate over
    elif len(semantic_image.shape) == 2:

        # get unique labels from single frame
        instance_image = label(semantic_image)

    else:
        raise ImageDimensionError(expected_dimensionality="2 or 3", received_dimensionality=len(instance_image.shape))

    return instance_image


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


def compile_multi_track_df(tracks_dict, assay_layout, track_len=None):
    """
    Iterates over many tracks stored in dictionary format and returns a df with
    additional features calculated

    Parameters
    ----------
    tracks_dict : dict
        A dictionary containing different sets of tracks from different expts
    track_len : int, optional
        Optional input to only store tracks of a set length
    """

    # list of track info dfs
    dfs = list()
    # empty dictionary for filtered tracks
    filtered_tracks = dict()

    # iterate over all tracks with tqdm
    for key in tqdm(tracks_dict.keys(), desc="Processing Tracks"):
        if track_len:
            # extract tracks only with max length
            filtered_tracks[key] = [track for track in tracks_dict[key]
                                    if len(track) == track_len]
        else:
            filtered_tracks[key] = tracks_dict[key]

        # iterate over full length tracks
        for track in filtered_tracks[key]:
            # get info for assay layout
            info = assay_layout.loc[key]
            # compile single track dictionary of info
            d = create_track_dictionary(track, info, key)
            # append df to list of dfs
            dfs.append(pd.DataFrame(d))

    # concat single track dfs into big df
    df = pd.concat(dfs, ignore_index=True)
    # interpolate missing values as sometimes segmentation drops result in NaN
    df.interpolate(inplace=True)

    return df
