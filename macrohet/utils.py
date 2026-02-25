"""utils.py

Core utility functions for image processing, tracking, and visualisation 
of Mtb-infected macrophages. 
Additional legacy code in here too.
"""

import math
import os

import btrack
import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
import seaborn as sns
from magicgui import magicgui
from scipy.ndimage import binary_erosion
from skimage.measure import regionprops
from skimage.morphology import label
from skimage.transform import downscale_local_mean, resize
from tqdm.auto import tqdm

from macrohet import dataio

# =============================================================================
# GLOBAL CONFIGURATION & COLOUR PALETTES
# =============================================================================

# Default scales taken from harmony metadata
napari_scale = [1.49e-05, 1.4949402023919043e-7, 1.4949402023919043e-7]

# Default scale factor for datasets tracked on scaled-down images
scale_factor = 6048 / 1200

custom_colours = {
    'lavender_raisin': ['#d8d8f6', '#b18fcf', '#978897', '#494850', '#2c2c34'],
    'expanded_piyg': ['#1a9641', '#a6d96a', '#978897', '#d1d1ca', '#f1b6da', '#d02c91'],
    'vaporwave': ['#D02C91', '#F1C2F2', '#C291F2', '#564D8C', '#57AAF2', '#A0D9D9', '#A6D96A', '#1A9641'],
    'mint_taupe': ['#0D1321', '#1D8FE0', '#C5D86D', '#ADD9C5', '#FFEDDF', '#8C6764', '#CC5105', '#8D7494'],
    'super_expiyg': ['#1A9641', '#7BDAA4', '#F0BE38', '#F0D795', '#EF93B9', '#D12B82', '#9F9BC7', '#452A61']
}

class ColorPalette:
    def __init__(self, color_map):
        self.colors = custom_colours[color_map]

    def replace(self, index, new_color):
        """Replace a colour code at the specified index with a new colour."""
        self.colors[index] = new_color

def color_palette(color_map):
    """Get the colour palette of the specified colour map."""
    return ColorPalette(color_map)

# =============================================================================
# EXCEPTIONS
# =============================================================================

class ImageDimensionError(Exception):
    def __init__(self, expected_dimensionality, received_dimensionality):
        message = f"Invalid image dimensionality. Expected {expected_dimensionality}-dimensional image, but received {received_dimensionality}-dimensional image."
        super().__init__(message)

# =============================================================================
# GENERAL MATH & UTILITIES
# =============================================================================

def euc_dist(x1, y1, x2, y2):
    """Euclidean distance displacement calculation for cell movement between frames."""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def calc_eccentricity(major_axis, minor_axis):
    """Calculates the eccentricity of an object given its major and minor axis lengths.
    Safely handles zero-division, invalid geometry, and vectorised array inputs.
    """
    return np.where(
        major_axis > 0,
        np.sqrt(np.clip(1 - (minor_axis**2 / np.maximum(major_axis**2, 1e-9)), 0, 1)),
        0.0
    )

def find_crossing(target, t_arr, a_arr):
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

# =============================================================================
# IMAGE PROCESSING & MASKING
# =============================================================================

def remove_small_segments(mask_stack, threshold_size=1000):
    """Remove small segments from a stack of binary masks."""
    if len(mask_stack.shape) != 3:
        raise ImageDimensionError(expected_dimensionality=3, received_dimensionality=len(mask_stack.shape))

    for n, frame in tqdm(enumerate(mask_stack), desc='Iterating over frames', total=len(mask_stack)):
        coords = [props.coords for props in regionprops(frame) if props.area < threshold_size]
        for segment_coords in coords:
            for x, y in segment_coords:
                frame[x, y] = 0

    return mask_stack

def instance_to_semantic(instance_image):
    """Change instance segmentation map to semantic segmentation."""
    if len(instance_image.shape) == 3:
        semantic_stack = list()
        for frame in tqdm(instance_image, total=len(instance_image), desc='Iterating over frames'):
            unique_labels = np.unique(frame)
            semantic_map = np.zeros_like(frame, dtype=np.uint8)
            semantic_map[frame == 0] = 0
            for sc_label in tqdm(unique_labels[1:], total=len(unique_labels) - 1, desc='Iterating over segments', leave=False):
                segment = frame == sc_label
                eroded_segment = binary_erosion(segment, structure=np.ones((5, 5)))
                semantic_map[eroded_segment] = 1
            semantic_stack.append(semantic_map)
        return np.stack(semantic_stack, axis=0)

    elif len(instance_image.shape) == 2:
        unique_labels = np.unique(instance_image)
        semantic_map = np.zeros_like(instance_image, dtype=np.uint8)
        semantic_map[instance_image == 0] = 0
        for sc_label in tqdm(unique_labels[1:], total=len(unique_labels) - 1, desc='Iterating over segments', leave=False):
            segment = instance_image == sc_label
            eroded_segment = binary_erosion(segment, structure=np.ones((5, 5)))
            semantic_map[eroded_segment] = 1
        return semantic_map
    else:
        raise ImageDimensionError(expected_dimensionality="2 or 3", received_dimensionality=len(instance_image.shape))

def semantic_to_instance(semantic_image):
    """Change semantic segmentation map to instance segmentation."""
    if len(semantic_image.shape) == 3:
        instance_stack = list()
        for frame in tqdm(semantic_image, total=len(semantic_image), desc='Iterating over frames'):
            instance_image = label(frame)
            instance_stack.append(instance_image)
        return np.stack(instance_stack, axis=0)

    elif len(semantic_image.shape) == 2:
        return label(semantic_image)
    else:
        raise ImageDimensionError(expected_dimensionality="2 or 3", received_dimensionality=len(semantic_image.shape))

def upscale_labels_post_manual_annotation(labels, scale_factor):
    """Upscales labels after manual annotation to restore to original size."""
    return resize(labels, (labels.shape[0] * scale_factor, labels.shape[1] * scale_factor),
                  anti_aliasing=False, order=0, preserve_range=True)

def downscale_images_for_manual_annotation(image, labels, scale_factor):
    """Downscale an image and its corresponding labels for manual annotation."""
    downsampled_image = downscale_local_mean(image, (scale_factor, scale_factor))
    downsampled_labels = downscale_local_mean(labels.astype(float), (scale_factor, scale_factor))
    downsampled_labels = np.round(downsampled_labels).astype(int)
    return downsampled_image, downsampled_labels

# =============================================================================
# DATA WRANGLING & TRACK PROCESSING
# =============================================================================

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

def extract_features(masks, max_proj_images, mtb_load_thresh=480, segment_size_thresh=1000, scale_factor=1.0, properties=('area', 'mean_intensity')):
    """Constructs composite intensity images and extracts btrack objects with Mtb burden properties."""
    manual_mtb_thresh = max_proj_images[:, 1, ...] >= mtb_load_thresh
    
    intensity_image = np.stack([
        max_proj_images[:, 0, ...], 
        max_proj_images[:, 1, ...], 
        manual_mtb_thresh.astype(bool)
    ], axis=-1)

    objects = btrack.utils.segmentation_to_objects(
        segmentation=masks,
        intensity_image=intensity_image,
        properties=properties,
        scale=(scale_factor, scale_factor),
        use_weighted_centroid=False
    )

    valid_objects = []
    for obj in objects:
        if obj.properties['area'] > segment_size_thresh:
            mtb_intensity = obj.properties['mean_intensity-2']
            obj.properties['Infected'] = bool(mtb_intensity > 0)
            obj.properties['Mtb area px'] = mtb_intensity * obj.properties['area']
            valid_objects.append(obj)
            
    return valid_objects

def localise(masks, intensity_image, properties=('area', 'mean_intensity', 'orientation'), scale_factor=1.0, use_weighted_centroid=False):
    """Extract single-cell objects and their properties from segmentation masks."""
    return btrack.utils.segmentation_to_objects(
        segmentation=masks,
        intensity_image=intensity_image,
        properties=properties,
        scale=(scale_factor, scale_factor),
        use_weighted_centroid=use_weighted_centroid
    )

def track(objects, masks, config_fn, scale_factor=1.0, search_radius=20):
    """Run Bayesian tracking on localised single-cell objects."""
    with btrack.BayesianTracker() as tracker:
        tracker.configure(config_fn)
        tracker.max_search_radius = search_radius
        tracker.tracking_updates = ["MOTION", "VISUAL"]
        tracker.features = list(objects[0].properties.keys())
        tracker.append(objects)
        tracker.volume = (
            (0, masks.shape[-2] * scale_factor),
            (0, masks.shape[-1] * scale_factor)
        )
        tracker.track(step_size=25)
        tracker.optimize()
        return tracker.tracks

def is_edge_cell(row):
    """Determine if a cell is near the boundary of a defined area to filter out artefacts."""
    safe_margin = 60  
    x_min, x_max = 0, 1200
    y_min, y_max = 0, 1200

    return (row['x'] <= x_min + safe_margin or 
            row['x'] >= x_max - safe_margin or 
            row['y'] <= y_min + safe_margin or 
            row['y'] >= y_max - safe_margin)

def mark_infection_status(group):
    """Determine and label the infection status of macrophage cells based on Mtb infection area."""
    group = group.sort_values(by='Time (hours)')
    valid_group = group.dropna(subset=['Mtb Area Model (µm)'])

    initial_period = valid_group[valid_group['Time Model (hours)'] <= valid_group['Time Model (hours)'].min() + 3]
    final_period = valid_group[valid_group['Time Model (hours)'] >= valid_group['Time Model (hours)'].max() - 3]

    group['Infection Status'] = valid_group['Mtb Area Model (µm)'] >= 1.92
    group['Initial Infection Status'] = (initial_period['Mtb Area Model (µm)'] >= 1.92).all()
    group['Final Infection Status'] = (final_period['Mtb Area Model (µm)'] >= 1.92).all()

    return group

def split_mean_intensity(input_dict):
    """Splits a multi-dimensional 'mean_intensity' array into separate entries for each channel."""
    if 'mean_intensity' in input_dict:
        mean_intensity_array = input_dict['mean_intensity']
        num_channels = mean_intensity_array.shape[1]
        intensity_channels = {f'mean_intensity_{i}': [] for i in range(num_channels)}

        for row in mean_intensity_array:
            for i in range(num_channels):
                intensity_channels[f'mean_intensity_{i}'].append(row[i])

        input_dict.update(intensity_channels)
        del input_dict['mean_intensity']

    return input_dict

def merge_tracks(track_ID_1, track_ID_2, tracks):
    """Merges two tracks identified by their IDs into a single pandas DataFrame."""
    track_1 = next((t for t in tracks if t.ID == track_ID_1), None)
    track_2 = next((t for t in tracks if t.ID == track_ID_2), None)

    if not track_1 or not track_2:
        raise ValueError("One or both track IDs not found in the provided track list.")

    track_1_df = pd.DataFrame(split_mean_intensity(track_1.to_dict()))
    track_2_df = pd.DataFrame(split_mean_intensity(track_2.to_dict()))

    return pd.concat([track_1_df, track_2_df], ignore_index=True)

def measure_mtb_area(track, masks, rfp_images, threshold=480, scale_factor=5.04, image_resolution=1.4949402023919043e-07):
    """Measures the physical area of an Mtb region in each frame of an image sequence."""
    mtb_areas = []
    for t, x, y in tqdm(zip(track.t, track.x, track.y), total=len(track), desc=f'Calculating mtb area: {track.ID}', leave=False):
        x, y = int(x * scale_factor), int(y * scale_factor)
        frame = masks[t, ...]

        if frame[y, x]:
            mask = frame == frame[y, x]
            masked_image = rfp_images[t] * mask
            mtb_area_px = np.sum(masked_image >= threshold)
            
            resolution_micrometers_per_pixel = image_resolution * 1_000_000
            mtb_areas.append(mtb_area_px * (resolution_micrometers_per_pixel ** 2))
        else:
            mtb_areas.append(0)

    return mtb_areas

def track_euc_dist(track_obj):
    """Calculate the Euclidean distance between frames for a tracklet."""
    track_df = dataio.track_to_df(track_obj)
    dxs = track_df['x'].diff()
    dys = track_df['y'].diff()
    return [np.sqrt(dxs[i]**2 + dys[i]**2) for i in range(1, len(track_df))]

def create_track_dictionary(track, info, key):
    """Create a dictionary of track information for dataframe compilation."""
    raw_mtb_values = pd.Series(track['mean_intensity'][:, 1]).interpolate(method='linear')
    raw_gfp = pd.Series(track['mean_intensity'][:, 0]).interpolate(method='linear')
    mtb_values = pd.Series(track['mean_intensity'][:, 2]).interpolate(method='linear')
    mtb_smooth = np.array(mtb_values.rolling(window=4).median().interpolate(method='backfill'))

    minor_axis_length = pd.Series(track['minor_axis_length']).interpolate(method='linear')
    major_axis_length = pd.Series(track['major_axis_length']).interpolate(method='linear')

    infection_status = pd.Series(track['Infected'])
    if pd.isnull(infection_status.iloc[0]):
        infection_status.iloc[0] = infection_status.iloc[infection_status.first_valid_index()]
    infection_status = infection_status.fillna(method='ffill')

    area = pd.Series(track['area']).interpolate(method='linear')

    return {
        'Time (hours)': track['t'],
        'x': track['x'],
        'y': track['y'],
        'x scaled': [x * 5.04 for x in track['x']],
        'y scaled': [y * 5.04 for y in track['y']],
        'Infection status': track['Infected'],
        'Initial infection status': track['Infected'][0],
        'Final infection status': track['Infected'][-1],
        'Area': track['area'],
        'Intracellular mean Mtb content': raw_mtb_values,
        'Intracellular thresholded Mtb content': mtb_values,
        'Intracellular thresholded Mtb content smooth': mtb_smooth,
        'Macroph. GFP expression': raw_gfp,
        'delta Mtb raw': [np.array(mtb_values)[-1] - np.array(mtb_values)[0] for _ in range(len(track))],
        'delta Mtb max raw': [(max(mtb_values) - min(mtb_values)) * (1 if np.argmax(mtb_values) > np.argmin(mtb_values) else -1) for _ in range(len(track))],
        'delta Mtb max smooth': [(max(mtb_smooth) - min(mtb_smooth)) * (1 if np.argmax(mtb_smooth) > np.argmin(mtb_smooth) else -1) for _ in range(len(track))],
        'delta Mtb max fold-change': [max(mtb_smooth) / min(mtb_smooth[mtb_smooth > 0]) * (1 if np.argmax(mtb_smooth) > np.argmin(mtb_smooth) else -1) if np.any(mtb_smooth > 0) else 0 for _ in range(len(track))],
        'delta Mtb max fold-change normalised mean area': [(max(mtb_smooth) / min(mtb_smooth[mtb_smooth > 0]) * (1 if np.argmax(mtb_smooth) > np.argmin(mtb_smooth) else -1)) / np.mean(area) if np.any(mtb_smooth > 0) else 0 for _ in range(len(track))],
        'delta Mtb max fold-change normalised max area': [(max(mtb_smooth) / min(mtb_smooth[mtb_smooth > 0]) * (1 if np.argmax(mtb_smooth) > np.argmin(mtb_smooth) else -1)) / np.max(area) if np.any(mtb_smooth > 0) else 0 for _ in range(len(track))],
        'delta Mtb/dt': np.polyfit(np.arange(len(mtb_smooth)), mtb_smooth, 1)[0],
        'Eccentricity': np.sqrt(1 - ((minor_axis_length ** 2) / (major_axis_length ** 2))),
        'MSD': [euc_dist(track['x'][i - 1], track['y'][i - 1], track['x'][i], track['y'][i]) if i != 0 else 0 for i in range(0, len(track))],
        'Strain': [info['Strain'] for _ in range(len(track['t']))],
        'Compound': [info['Compound'] for _ in range(len(track['t']))],
        'Concentration': [info['ConcentrationEC'] for _ in range(len(track['t']))],
        'Technical replicate': [info['Technical replicate'] for _ in range(len(track['t']))],
        'Cell ID': [track.ID for _ in range(len(track['t']))],
        'Acquisition ID': [key for _ in range(len(track['t']))],
        'Unique ID': [f'{track.ID}.{key[0]}.{key[1]}' for _ in range(len(track['t']))]
    }

def compile_multi_track_df(tracks_dict, assay_layout, track_len=None):
    """Iterates over many tracks stored in dictionary format and concatenates them."""
    dfs = list()
    filtered_tracks = dict()

    for key in tqdm(tracks_dict.keys(), desc="Processing Tracks"):
        if track_len:
            filtered_tracks[key] = [track for track in tracks_dict[key] if len(track) == track_len]
        else:
            filtered_tracks[key] = tracks_dict[key]

        for track in filtered_tracks[key]:
            info = assay_layout.loc[key]
            d = create_track_dictionary(track, info, key)
            dfs.append(pd.DataFrame(d))

    df = pd.concat(dfs, ignore_index=True)
    df.interpolate(inplace=True)

    return df

# =============================================================================
# VISUALISATION (NAPARI)
# =============================================================================

def highlight_cell_gui(tracks, viewer):
    @magicgui(call_button='Highlight cell ID',
              cell_ID={"widget_type": "SpinBox", "min": 0, "max": max([track.ID for track in tracks])},
              size={"widget_type": "Slider", "min": 1, "max": 1000},
              opacity={"widget_type": "FloatSlider", "min": 0.0, "max": 1.0},
              symbol={"choices": ["o", "s", "t", "+", "x"]},
              edge_color={"choices": ["white", "black", "gray", "red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta"]},
              edge_width={"widget_type": "FloatSlider", "min": 0.0, "max": 1.0},
              cell_property={"choices": list(tracks[0].properties.keys()) + ["Show All"]},
              )
    def highlight_cell(cell_ID=1, size=100, opacity=1, symbol="o", edge_color="white", edge_width=0.1, cell_property="area", tracking_scale_factor=6048 / 1200) -> napari.types.LayerDataTuple:
        """Highlights and displays a specific cell in Napari viewer."""
        try:
            track = next(track for track in tracks if track.ID == cell_ID)
        except StopIteration:
            print(f"Error: No cell found with ID {cell_ID}")
            return

        data = np.array([[track.t[i], track.y[i] * tracking_scale_factor, track.x[i] * tracking_scale_factor]
                         for i in range(len(track))])

        if cell_property == 'Show All':
            props = {k: list(map(str, v)) for k, v in track.properties.items()}
        else:
            props = {cell_property: list(map(str, track.properties[cell_property]))}

        viewer.dims.current_step = tuple(data[0])

        return (data, {'properties': props,
                       'size': size,
                       'opacity': opacity,
                       'symbol': symbol,
                       'name': f'Cell ID:{cell_ID}',
                       'face_color': 'transparent',
                       'edge_color': edge_color,
                       'edge_width': edge_width},
                'points')

    return highlight_cell

def add_napari_grid_overlay(viewer, N_rows_cols=10, scale_factor=1, edge_width=10, edge_color="cyan"):
    """Adds a rectangular grid overlay to a Napari viewer window."""
    max_coord = max(viewer.layers[0].data.shape) * scale_factor
    edge_width = edge_width * scale_factor

    vertical_grid_lines = [
        np.array([[0, (max_coord / (N_rows_cols)) * i],
                  [max_coord, (max_coord / (N_rows_cols)) * i]])
        for i in range(1, N_rows_cols)
    ]

    horizontal_grid_lines = [
        np.array([[(max_coord / (N_rows_cols)) * i, 0],
                  [(max_coord / (N_rows_cols)) * i, max_coord]])
        for i in range(1, N_rows_cols)
    ]

    grid_lines = vertical_grid_lines + horizontal_grid_lines
    return viewer.add_shapes(grid_lines, shape_type="line", edge_width=edge_width, edge_color=edge_color)

def highlight_cell_fate(cell_ID, viewer, tracks, scale_factor=scale_factor, napari_scale=napari_scale):
    """Puts a napari point layer around the final frame of the cell of interest."""
    track = [track for track in tracks if track.ID == cell_ID][0]
    x, y = track.x[-1] * scale_factor, track.y[-1] * scale_factor
    t = track.t[-1]
    
    highlight = viewer.add_points([t, y, x], size=300,
                                  face_color='transparent',
                                  edge_color='white',
                                  edge_width=0.1,
                                  name=f'cell {cell_ID} fate',
                                  scale=napari_scale)
    viewer.dims.current_step = (t, y, x)

    return highlight

def highlight_cell(cell_ID, viewer, tracks, scale_factor=scale_factor, napari_scale=napari_scale, size=300, opacity=1, symbol='o', reset_position=True):
    """Puts a Napari point layer around the cell of interest over all frames."""
    track = [track for track in tracks if track.ID == cell_ID][0]
    points = [[track.t[i], track.y[i] * scale_factor, track.x[i] * scale_factor]
              for i in range(len(track))]
              
    highlight = viewer.add_points(points, size=size,
                                  symbol=symbol,
                                  face_color='transparent',
                                  edge_color='white',
                                  edge_width=0.1,
                                  name=f'cell {cell_ID}',
                                  opacity=opacity)
    if reset_position:
        viewer.dims.current_step = (points[0])

    return highlight