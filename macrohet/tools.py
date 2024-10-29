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


def is_edge_cell(row):
    """
    Determine if a cell is near the boundary of a defined area, marking it as an edge cell
    based on its coordinates and a calculated safe margin.

    Parameters:
    -----------
    row : Series
        A pandas Series representing a row of data with at least the columns 'x', 'y',
        and 'Mphi Area (µm)', where 'x' and 'y' denote cell coordinates, and 'Mphi Area (µm)'
        is used to set the safe margin.

    Returns:
    --------
    bool
        True if the cell is considered an edge cell, meaning it lies within a defined safe margin
        from any boundary of the area; False otherwise.

    Example:
    --------
    # Sample row with columns 'x', 'y', and 'Mphi Area (µm)'
    sample_row = pd.Series({
        'x': 50,
        'y': 1150,
        'Mphi Area (µm)': 500
    })

    # Check if the cell is an edge cell
    is_edge = is_edge_cell(sample_row)
    # or apply over whole df
    df.apply(is_edge_cell, axis=1)
    """

    # Calculate the safe margin for the current row's area
    safe_margin = 60  # Can be dynamically calculated based on 'Mphi Area (µm)'

    # Define bounding box limits
    x_min, x_max = 0, 1200
    y_min, y_max = 0, 1200

    # Check if the cell's coordinates are within the safe margin from the edge
    near_left_edge = row['x'] <= x_min + safe_margin
    near_right_edge = row['x'] >= x_max - safe_margin
    near_bottom_edge = row['y'] <= y_min + safe_margin
    near_top_edge = row['y'] >= y_max - safe_margin

    return near_left_edge or near_right_edge or near_bottom_edge or near_top_edge


def mark_infection_status(group):
    """
    Determine and label the infection status of macrophage cells based on Mtb infection area
    over a time series. The function adds columns to the DataFrame indicating whether the
    macrophage cell is considered infected in the initial, final, and overall time periods.

    Parameters:
    -----------
    group : DataFrame
        A pandas DataFrame grouped by cell or sample, containing columns 'Time Model (hours)'
        and 'Mtb Area Model (µm)' for each timepoint.

    Returns:
    --------
    DataFrame
        The input DataFrame with added columns:
        - 'Infection Status': A boolean indicating if the cell is infected (mean Mtb area ≥ 1.92 µm²).
        - 'Initial Infection Status': A boolean indicating infection within the first 3 hours of
          available Mtb area data.
        - 'Final Infection Status': A boolean indicating infection in the last 3 hours of available
          Mtb area data.

    Example:
    --------
    # Sample DataFrame with columns 'Time Model (hours)' and 'Mtb Area Model (µm)'
    sample_df = pd.DataFrame({
        'Time Model (hours)': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'Mtb Area Model (µm)': [0.5, 1.0, 2.0, 2.5, None, 1.9, 2.1, 2.0, 2.2]
    })

    # Apply function to each group
    result = sample_df.groupby('Cell ID').apply(mark_infection_status)
    """

    # Sort the group by 'Time (hours)' just in case
    group = group.sort_values(by='Time (hours)')

    # Filter out rows where 'Mtb Area (µm)' is NaN
    valid_group = group.dropna(subset=['Mtb Area Model (µm)'])

    # Determine the time range for the first three hours of non-NaN data
    initial_period = valid_group[valid_group['Time Model (hours)'] <= valid_group['Time Model (hours)'].min() + 3]

    # Determine the time range for the last three hours of non-NaN data
    final_period = valid_group[valid_group['Time Model (hours)'] >= valid_group['Time Model (hours)'].max() - 3]

    # Infection Status: If the mean Mtb Area (µm) in the entire group is >= 1.92
    group['Infection Status'] = valid_group['Mtb Area Model (µm)'] >= 1.92

    # Initial Infection Status: If the mean Mtb Area (µm) in the first 3 non-NaN hours is >= 1.92
    group['Initial Infection Status'] = (initial_period['Mtb Area Model (µm)'] >= 1.92).all()

    # Final Infection Status: If the mean Mtb Area (µm) in the last 3 non-NaN hours is >= 1.92
    group['Final Infection Status'] = (final_period['Mtb Area Model (µm)'] >= 1.92).all()

    return group


def process_mtb_area(df, id_column='ID', mtb_column='Mtb Area'):
    """
    Process the 'Mtb Area' column of the DataFrame by applying linear interpolation,
    backfill interpolation, and a rolling median with a dynamic window size based on the 'ID' column.

    The window size is set to 5 if 'PS000' is in the ID, otherwise it is set to 10.

    Parameters:
    - df: pandas.DataFrame
        The input DataFrame containing the data to process.
    - id_column: str, optional (default='ID')
        The name of the column containing the IDs to group by.
    - mtb_column: str, optional (default='Mtb Area')
        The name of the column containing the Mtb Area values to process.

    Returns:
    - pandas.Series
        A Series with the processed 'Mtb Area' values.
    """

    def dynamic_window(id_value):
        """
        Determine the window size based on the presence of 'PS000' in the ID.

        Parameters:
        - id_value: str
            The ID value to check.

        Returns:
        - int
            The window size for the rolling median.
        """
        return 5 if 'PS0000' in id_value else 10

    return df.groupby(id_column)[mtb_column].apply(
        lambda group: group.interpolate(method='linear')
                           .interpolate(method='backfill')
                           .rolling(window=dynamic_window(group.name))
                           .median()
    )


def merge_tracks(track_ID_1, track_ID_2, tracks):
    """
    Merges two tracks identified by their IDs into a single pandas DataFrame.

    This function finds the tracks with the specified IDs from a list of track objects,
    converts them to dictionaries, applies the split_mean_intensity function to split
    the 'mean_intensity' into separate channels, and then merges the two tracks into a
    single DataFrame.

    Parameters:
    track_ID_1 (int): The ID of the first track to merge.
    track_ID_2 (int): The ID of the second track to merge.
    tracks (list): A list of track objects, each with a 'to_dict' method and an 'ID' attribute.

    Returns:
    pandas.DataFrame: A DataFrame containing the merged data from the two specified tracks.

    Raises:
    ValueError: If track IDs are not found in the provided track list.
    """

    # Find and validate tracks by IDs
    track_1 = next((t for t in tracks if t.ID == track_ID_1), None)
    track_2 = next((t for t in tracks if t.ID == track_ID_2), None)

    if not track_1 or not track_2:
        raise ValueError("One or both track IDs not found in the provided track list.")

    # Convert tracks to dictionaries and split mean intensity
    track_1_dict = split_mean_intensity(track_1.to_dict())
    track_2_dict = split_mean_intensity(track_2.to_dict())

    # Convert dictionaries to pandas DataFrames
    track_1_df = pd.DataFrame(track_1_dict)
    track_2_df = pd.DataFrame(track_2_dict)

    # Merge the two DataFrames
    final_track = pd.concat([track_1_df, track_2_df], ignore_index=True)

    return final_track


def split_mean_intensity(input_dict):
    """
    Splits the 'mean_intensity' entry in a track dictionary from a multi-dimensional
    array into separate entries for each channel. This is useful for integrating the data
    into a pandas DataFrame when each channel needs to be a separate column.

    The function dynamically handles any number of intensity channels.

    Parameters:
    input_dict (dict): A dictionary containing a 'mean_intensity' key with a multi-dimensional
                       array as its value.

    Returns:
    dict: The modified dictionary with separate entries for each intensity channel.
          The original 'mean_intensity' key is removed.

    Example:
    input_dict = {
        'mean_intensity': array([[val00, val01, val02], [val10, val11, val12], ...])
    }
    output_dict = split_mean_intensity(input_dict)
    # output_dict will have keys 'mean_intensity_0', 'mean_intensity_1', 'mean_intensity_2', etc.
    """
    # Check if 'mean_intensity' is in the dictionary
    if 'mean_intensity' in input_dict:
        # Extract the array
        mean_intensity_array = input_dict['mean_intensity']

        # Determine the number of channels (columns in the array)
        num_channels = mean_intensity_array.shape[1]

        # Initialize lists for the split entries for each channel
        intensity_channels = {f'mean_intensity_{i}': [] for i in range(num_channels)}

        # Iterate through the array and split the values for each channel
        for row in mean_intensity_array:
            for i in range(num_channels):
                intensity_channels[f'mean_intensity_{i}'].append(row[i])

        # Update the dictionary with new entries for each channel
        input_dict.update(intensity_channels)

        # Remove the original 'mean_intensity' entry
        del input_dict['mean_intensity']

    return input_dict


def measure_mtb_area(track, masks, rfp_images, threshold=480, scale_factor=5.04, image_resolution=1.4949402023919043e-07):
    """
    Measures the area of a region in each frame of an image sequence.

    Parameters:
    - track (object): An object containing tracking information with attributes 't', 'x', 'y', and 'ID'.
    - masks (array): A numpy array representing the segmented masks of the images.
    - rfp_images (array): A numpy array of RFP (red fluorescent protein) images.
    - threshold (int): The intensity threshold for considering a pixel as part of the region.
    - scale_factor (float): Factor for scaling the coordinates from the track object.
    - image_resolution (float): The resolution of the images in pixels per meter.

    Returns:
    - mtb_areas (list): A list of areas (in micrometers squared) of the region for each frame.

    The function iterates over each frame specified in the track object, scales the coordinates,
    and selects the corresponding mask. If a mask exists at the specified coordinates, it calculates
    the area of the region with intensity above the threshold in the RFP image. The area is then converted
    from pixels to micrometers squared using the image resolution.
    """

    mtb_areas = []
    for t, x, y in tqdm(zip(track.t, track.x, track.y),
                        total=len(track),
                        desc=f'Calculating mtb area for every frame in track: {track.ID}',
                        leave=False):
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

            # Apply threshold to identify mtb region
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
