import os

import btrack
import cv2
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
from skimage.morphology import area_closing, label, remove_small_objects
from skimage.transform import downscale_local_mean, resize
from tqdm.auto import tqdm

from macrohet import dataio, tile

from .colours import custom_colours

# default scale value taken from harmony metadata
napari_scale = [1.49e-05, 1.4949402023919043E-7, 1.4949402023919043E-7]
# default scale factor
# for datasets that have been tracked on scaled down images
scale_factor = 6048 / 1200


class ColorPalette:
    def __init__(self, color_map):
        self.colors = custom_colours[color_map]

    def replace(self, index, new_color):
        """
        Replace a color code at the specified index with a new color.

        Parameters:
            index (int): The index of the color code to replace.
            new_color (str): The new color code.

        Returns:
            None
        """
        self.colors[index] = new_color


def color_palette(color_map):
    """
    Get the color palette of the specified color map.

    Parameters:
        color_map (str): The name of the color map.

    Returns:
        ColorPalette: The color palette object.
    """
    return ColorPalette(color_map)


def show_colors(color_map):
    """
    Display the colors in the specified color map as bars.

    Parameters:
        color_map (str): The name of the color map to display.

    Returns:
        None
    """
    colors = custom_colours[color_map]
    num_colors = len(colors)

    fig, ax = plt.subplots()
    for i, color in enumerate(colors):
        ax.bar(i, 1, color=color)

    ax.set_xlim(-0.5, num_colors - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(num_colors))
    ax.set_xticklabels(['' for _ in range(num_colors)])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.title(color_map)
    plt.show()


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


def add_napari_grid_overlay(
    viewer, N_rows_cols=10, scale_factor=1, edge_width=10, edge_color="cyan"
):
    """
    Adds a rectangular grid overlay to a Napari viewer window.

    Parameters:
    viewer (napari.viewer.Viewer): A Napari viewer instance.
    N_rows_cols (int, optional): The number of divisions to divide the grid
        into. Default is 10.
    scale_factor (float, optional): A scaling factor applied to the grid size.
        Default is 1.
    edge_width (int, optional): The width of the grid lines. Default is 10.
    edge_color (str, optional): The color of the grid lines. Default is 'cyan'.

    Returns:
    napari.layers.Shapes: A shapes layer representing the grid lines.

    This function adds a rectangular grid overlay to the Napari viewer window.
    The grid is divided into N_rows_cols rows and N_rows_cols columns, forming
    a rectangular shape. It can be used to aid with manual labeling of large
    images.

    Note:
    - The function assumes that the first layer in the viewer contains the
      image data used to determine the maximum coordinate value.
    - The viewer should be displayed before calling this function.

    Example:
    ```
    import napari

    # Create a Napari viewer and add an image layer
    viewer = napari.Viewer()
    viewer.add_image(image_data)

    # Add a grid overlay with 5 rows and 5 columns, set the edge width to 5,
    # and scale the grid size by a factor of 1.5
    grid_layer = add_napari_grid_overlay(viewer, N_rows_cols=5,
                                         scale_factor=1.5, edge_width=5)

    # Display the viewer
    napari.run()
    ```
    """

    # Get the spatial extent of what is presumed to be a square image, scaled
    # by the factor
    max_coord = max(viewer.layers[0].data.shape) * scale_factor

    # rescale the edge_width
    edge_width = edge_width * scale_factor

    # Calculate the vertical lines
    vertical_grid_lines = [
        np.array([[0, (max_coord / (N_rows_cols)) * i],
                  [max_coord, (max_coord / (N_rows_cols)) * i]])
        for i in range(1, N_rows_cols)
    ]

    # Calculate the horizontal lines
    horizontal_grid_lines = [
        np.array([[(max_coord / (N_rows_cols)) * i, 0],
                  [(max_coord / (N_rows_cols)) * i, max_coord]])
        for i in range(1, N_rows_cols)
    ]

    # Append the vertical and horizontal lines together
    grid_lines = vertical_grid_lines + horizontal_grid_lines

    # Add the grid lines to a shapes layer with line shape type, specified
    # edge width, and edge color
    shapes_layer = viewer.add_shapes(
        grid_lines, shape_type="line", edge_width=edge_width, edge_color=edge_color
    )

    return shapes_layer


def clear_previous_cell_highlight(viewer):
    """
    Clears latest added points layer (presuming its the points layer)
    Useful if iteratively checking on cell identities, just call before either
    of the other cell highlight functions (highlight_cell/highlight_cell_fate)
    """
    name = viewer.layers[-1].name
    viewer.layers.remove(name)


def highlight_cell_fate(cell_ID, viewer, tracks,
                        scale_factor=scale_factor,
                        napari_scale=napari_scale):
    """
    Puts a napari point layer around the final frame of the cell of interest

    Parameters
    ----------
    cell_ID : int
        ID of the cell of interest
    viewer : napari.viewer.Viewer
        The viewer instance to launch the visualisation in
    tracks : list of btrack.btypes.Tracklet
        List of tracks in which the cell of interest is stored
    scale_factor : float
        If cells have been tracked on downscaled images then rescale tracks
    napari_scale : list of float
        Pixel to m scale for napari in case scale bar is required


    Returns
    ----------
    highlight : napari.layers.points.points.Points
        Napari layer with cell highlighted at final frame
    """

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


def highlight_cell(cell_ID, viewer, tracks, scale_factor=scale_factor,
                   napari_scale=napari_scale, size=300, opacity=1,
                   symbol='o', reset_position=True):
    """
    Puts a Napari point layer around the cell of interest over all frames.

    Parameters
    ----------
    cell_ID : int
        ID of the cell of interest.
    viewer : napari.viewer.Viewer
        The viewer instance to launch the visualization in.
    tracks : list of btrack.btypes.Tracklet
        List of tracks in which the cell of interest is stored.
    scale_factor : float, optional
        Scale factor for rescaling tracks if cells have been tracked on downscaled images.
    napari_scale : list of float, optional
        Pixel-to-meter scale for Napari in case a scale bar is required.
    size : int, optional
        Size of the points in the Napari point layer.
    opacity : float, optional
        Opacity of the points in the Napari point layer.
    symbol : str, optional
        Symbol used for the points in the Napari point layer.
    reset_position : bool, optional
        Whether to reset the viewer's position to the first frame of the highlighted cell.

    Returns
    ----------
    highlight : napari.layers.points.points.Points
        Napari layer with the cell highlighted at the final frame.
    """
    track = [track for track in tracks if track.ID == cell_ID][0]
    points = [[track.t[i], track.y[i] * scale_factor, track.x[i] * scale_factor]
              for i in range(len(track))]
    highlight = viewer.add_points(points, size=size,
                                  symbol=symbol,
                                  face_color='transparent',
                                  edge_color='white',
                                  edge_width=0.1,
                                  name=f'cell {cell_ID}',
                                  opacity=opacity
                                  # scale=napari_scale
                                  )
    if reset_position:
        viewer.dims.current_step = (points[0])

    return highlight


def scale_napari_tracks(napari_tracks, scale=scale_factor):
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


def scale_masks(masks_stack, final_image_size=(6048, 6048)):
    """
    Quick fix for upscaling masks to be original image size (after they were
    downscaled for tracking hacky fix)

    Parameters
    ----------
    masks_stack : array ()
        2D (XY) or 3D (TXY) array of instance segmentation for resizing
    final_image_size : tuple
        Desired final image size of a single time frame in the array

    Returns
    ----------
    mod_masks : array ()
        2D (XY) or 3D (TXY) array of instance segmentation that has been resized

    """
    # ensure that downsized masks have unique IDs for latter steps
    mod_masks = np.stack([label(masks)
                          for masks in tqdm(masks_stack,
                                            desc='Labelling')],
                         axis=0)
    # remove small objects
    mod_masks = np.stack([remove_small_objects(masks,
                                               min_size=64,
                                               connectivity=1)
                          for masks in tqdm(mod_masks,
                                            desc='Remove small objs')],
                         axis=0)
    # fill small holes
    mod_masks = np.stack([area_closing(masks,
                                       area_threshold=64)
                          for masks in tqdm(mod_masks,
                                            desc='Remove small holes')],
                         axis=0).astype(np.uint16)
    # upscale
    mod_masks = np.stack([cv2.resize(masks,
                                     final_image_size,
                                     interpolation=cv2.INTER_NEAREST)
                          for masks in tqdm(mod_masks,
                                            desc=f'Scaling to {final_image_size}')],
                         axis=0)

    return mod_masks


def create_glimpse_from_sc_df(sc_df, acq_ID, ID, images,
                              scale=6048 / 1200, size=500):
    """
    Takes a pandas dataframe containing information pertaining to a single cell
    and creates a glimpse set of images showing that single cell over time
    Parameters
    ----------
    sc_df : pd.DataFrame()
        A single-cell subset of the previously created pandas dataframe of many
        single-cell trajectories generated in compile_multi_track_df()
    acq_ID : tuple
        A tuple identifier for which experiment has been chosen
    ID : int
        An integer ID for which cell has been chosen
    images : dask.array() or numpy.array()
        The main corpus of images from which the glimpse will be created
    scale : int
        The tracking was originally performed on a rescaled image data set, so
        a scaling factor needs to be provided to match tracks to original images
    size : int
        Size of the final glimpse
    """

    # create empty list for stack of images
    glimpse_stack = list()
    # iterate over time points from single cell
    for row in tqdm(sc_df.iterrows(), total=len(sc_df),
                    desc=f'Creating glimpse ID: {acq_ID, ID}'):
        # get coords
        t, x, y = row[1]['Time (hours)'], row[1]['y'], row[1]['x']
        # select proper frame
        frame = images[t, ...]
        # scale as tracking was done on rescaled images
        x1, y1 = x * scale, y * scale
        # create window for glimpse
        x1, x2, y1, y2 = x1, x1 + size, y1, y1 + size
        # add padding for boundary cases
        frame = np.pad(frame, [(0, 0), (size / 2, size / 2), (size / 2, size / 2)],
                       'constant', constant_values=0)
        # create glimpse image by cropping original image
        glimpse = frame[..., int(x1): int(x2), int(y1): int(y2)]
        # append to glimpse stack
        glimpse_stack.append(glimpse)

    # stack glimpse together
    glimpse_stack = np.stack(glimpse_stack, axis=1)

    return glimpse_stack


def create_mask_glimpse_from_sc_df(sc_df, acq_ID, ID, masks,
                                   scale=6048 / 1200, size=500):
    """
    Takes a pandas dataframe containing information pertaining to a single cell
    and creates a MASK glimpse set of images showing that single cell mask over
    time.

    Parameters
    ----------
    sc_df : pd.DataFrame()
        A single-cell subset of the previously created pandas dataframe of many
        single-cell trajectories generated in compile_multi_track_df()
    acq_ID : tuple
        A tuple identifier for which experiment has been chosen
    ID : int
        An integer ID for which cell has been chosen
    masks : dask.array() or numpy.array()
        The main corpus of mask images from which the glimpse will be created
    scale : int
        The tracking was originally performed on a rescaled image data set, so
        a scaling factor needs to be provided to match tracks to original images
    size : int
        Size of the final glimpse
    """

    # create empty list for stack of images
    mask_glimpse_stack = list()
    coords_stack = list()
    # iterate over time points from single cell
    for row in tqdm(sc_df.iterrows(), total=len(sc_df),
                    desc=f'Creating mask glimpse ID: {acq_ID, ID}'):
        # get coords
        t, x, y = row[1]['Time (hours)'], row[1]['y'], row[1]['x']
        # select proper frame
        frame = masks[t, ...]
        # scale as tracking was done on rescaled images
        x1, y1 = x, y
        # create window for glimpse
        x1, x2, y1, y2 = x1, x1 + np.ceil(size / scale), y1, y1 + np.ceil(size / scale)
        # add padding for boundary casess
        frame = np.pad(frame, [(int(np.ceil(size / scale) / 2),
                       int(np.ceil(size / scale) / 2)),
                      (int(np.ceil(size / scale) / 2),
                       int(np.ceil(size / scale) / 2))],
                       'constant', constant_values=0)
        # create glimpse image by cropping original image
        mask_glimpse = frame[int(x1): int(x2), int(y1): int(y2)]
        # check to see if mask exists
        if mask_glimpse[int(np.ceil(size / scale) / 2), int(np.ceil(size / scale) / 2)]:
            # select cell of interest
            mask_glimpse = mask_glimpse == mask_glimpse[int(np.ceil(size
                                                        / scale) / 2),
                                                        int(np.ceil(size / scale) / 2)]
            # resize to image size
            mask_glimpse = cv2.resize(mask_glimpse.astype(np.uint8),
                                      (size , size))
            # get contour
            cnts = cv2.findContours(mask_glimpse,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            coords = np.asarray([[t, i[0][1], i[0][0]] for i in cnts[0]])
            for c in cnts:
                cv2.drawContours(mask_glimpse, [c], -1, (255, 255, 255),
                                 thickness=5)
            # change dtype of mask
            mask_glimpse = (mask_glimpse == 255).astype(np.uint16)
        # if mask doesnt exist then leave blank
        else:
            coords = [t, 0, 0]
            mask_glimpse = np.zeros((size, size))
        # append to mask glimpse stack
        mask_glimpse_stack.append(mask_glimpse)
        coords_stack.append(coords)
    # stack mask glimpse together
    mask_glimpse_stack = np.stack(mask_glimpse_stack, axis=0)
    # build coords into shape for napari
    mask_shapes = np.asarray(coords_stack, )

    return mask_glimpse_stack, mask_shapes


def compile_mp4(input_dir, output_fn, fps, fileformat : str = '.tiff'):
    """
    Take a series of images and compile mp4 video from them.

    Parameters
    ----------
    input_dir : PathLike
        The full path to a directory containing a series of .tiff images.
    output_fn : PathLike
        Filename for output mp4 video.
    fps : int
        The number of frames per second you would like to generate the video at.
    fileformat : str
        Optional input for different types of image fileformat
    """

    # Set the frame rate of the output video
    frame_rate = 3
    # Get the list of images in the directory
    frames = [img for img in os.listdir(input_dir) if img.endswith(fileformat)]
    # Sort the images in alphabetical order
    frames = natsorted(frames)
    # Get the first image to get the size of the video
    frame = cv2.imread(os.path.join(input_dir, frames[0]))
    height, width, channels = frame.shape
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_fn,
                                   fourcc,
                                   frame_rate,
                                   (width, height))
    # Loop through the images and add them to the video
    for image in frames:
        img_path = os.path.join(input_dir, image)
        frame = cv2.imread(img_path)
        # Write the frame to the video
        video_writer.write(frame)
    # Release the VideoWriter object
    video_writer.release()


def add_scale(viewer, font_size=24, text_colour='white', ticks=False):
    """
    Params for adding a scale bar to napari.Viewer()
    Actual scale must be defined in the viewer methods (i.e. viewer.add_image())
    """
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = 'm'
    viewer.scale_bar.font_size = font_size
    viewer.scale_bar.colored = True
    viewer.scale_bar.color = text_colour
    viewer.scale_bar.ticks = ticks


def add_time(viewer, frame_rate=1, font_size=24, text_colour='white',
             position='bottom_left'):
    """
    Params for adding a time counter to napari.Viewer().
    Used in conjunction with update_slider()
    """
    def update_slider(event):
        # only trigger if update comes from first axis (optional)
        # ind_lambda = viewer.dims.indices[0]
        time = viewer.dims.current_step[0]
        viewer.text_overlay.text = f"{time:1.1f} hours"

    viewer.text_overlay.visible = True
    viewer.text_overlay.color = text_colour
    viewer.text_overlay.position = 'bottom_left'
    viewer.text_overlay.font_size = font_size
    viewer.dims.events.current_step.connect(update_slider)


def tn_glimpse_maker(unique_ID, df, time, metadata=None, segmentation=None,
                     base_dir='desktop', crop_size=None,
                     track_scale_factor=5.04, mask_outline=True):
    """
    Create a cropped RGB image for a specific unique ID and time.

    Args:
        unique_ID (str): The unique identifier of the cell.
        df (pandas.DataFrame): The DataFrame containing the cell data.
        time (int, list, tuple, range): The time value(s) to extract the image(s) for.
        metadata (object, optional): The metadata object. Default is None. Best to supply beforehand otherwise
            it will result in a slow loading of the metadata (which is the same for all cells) for each cell.
        base_dir (str, optional): The base directory path or special string value ('desktop' or 'laptop')
            to determine the path. Default is 'desktop'.
        crop_size (int, optional): The desired size of the cropped image.
            If None, it's calculated based on the area. Default is None.
        track_scale_factor (float, optional): The scale factor to apply to the tracking coordinates. Default is 5.04.
        mask_outline (bool, optional): Whether to draw the mask outline on the image. Default is True.

    Returns:
        numpy.ndarray: The cropped RGB image.

    Raises:
        ValueError: If an invalid input type is provided for 'time'.
    """
    # extract row and column from unique_ID
    cell_ID, row, column = unique_ID.split('.')

    # Replace base_dir with appropriate string path if 'desktop' or 'laptop' is provided
    if base_dir == 'desktop':
        base_dir = '/mnt/DATA/macrohet/'
    elif base_dir == 'laptop':
        base_dir = '/Volumes/lab-gutierrezm/home/users/dayn/macrohet/'
    # check if metadata has been supplied, if not then load using basedir
    if metadata is None:
        # load metadata and preload images
        metadata_fn = os.path.join(base_dir, 'macrohet_images/PS0000/Index.idx.xml')
        metadata = dataio.read_harmony_metadata(metadata_fn)

    image_dir = os.path.join(base_dir, 'macrohet_images/PS0000/Images')
    images = tile.compile_mosaic(image_dir, metadata, row, column, set_plane='sum_proj')

    if segmentation is None:
        # load segmentation
        with btrack.io.HDF5FileHandler(os.path.join(base_dir,
                                                    f'labels/macrohet_seg_model/{int(row),int(column)}.h5'),
                                       'r',
                                       obj_type='obj_type_1') as reader:
            segmentation = reader.segmentation

    # extract single cell df
    sc_df = df[df['Unique ID'] == unique_ID]

    if isinstance(time, int):
        time_values = [time]
    elif isinstance(time, (list, tuple)):
        time_values = time
    elif isinstance(time, range):
        time_values = list(time)
    else:
        raise ValueError("Invalid input type for 'time'. Please provide an integer, a list of integers, or a range.")

    for t in time_values:
        if t == -1:
            t = sc_df['Time (hours)'].iloc[-1]

        sc_df_t = sc_df[sc_df['Time (hours)'] == t]

        # Extract xy coordinates and transpose for python and area from the cell information
        y_coord, x_coord, area, t = sc_df_t.loc[:, ['x', 'y', 'Area', 'Time (hours)']].values[0]

        # Scale according to tracking shrinkage
        y_coord, x_coord = y_coord * track_scale_factor, x_coord * track_scale_factor

        if not crop_size:
            # Calculate the side length for cropping based on the square root of the area
            side_length = int(np.sqrt(area)) * 2

        # Calculate the cropping boundaries
        x_start = int(x_coord - side_length / 2)
        x_end = int(x_coord + side_length / 2)
        y_start = int(y_coord - side_length / 2)
        y_end = int(y_coord + side_length / 2)

        # Pad the boundaries if they exceed the image dimensions
        if x_start < 0:
            x_pad = abs(x_start)
            x_start = 0
        else:
            x_pad = 0

        if x_end > images.shape[2]:
            x_pad_end = x_end - images.shape[2]
            x_end = images.shape[2]
        else:
            x_pad_end = 0

        if y_start < 0:
            y_pad = abs(y_start)
            y_start = 0
        else:
            y_pad = 0

        if y_end > images.shape[3]:
            y_pad_end = y_end - images.shape[3]
            y_end = images.shape[3]
        else:
            y_pad_end = 0

        # Crop the image
        cropped_image = images[t, :, x_start:x_end, y_start:y_end]

        # Pad the cropped image if necessary
        cropped_image = np.pad(cropped_image, ((0, 0), (x_pad, x_pad_end), (y_pad, y_pad_end)), mode='constant')

        # extract the gfp and rfp channels to apply some vis techn
        gfp = cropped_image[0, ...].compute().compute()
        rfp = cropped_image[1, ...].compute().compute()

        # clip the images so that the contrast is more apparent
        contrast_lim_gfp = np.clip(gfp, 358, 5886)
        contrast_lim_rfp = np.clip(rfp, 480, 1300)

        norm_gfp = cv2.normalize(contrast_lim_gfp, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)
        norm_rfp = cv2.normalize(contrast_lim_rfp, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)

        # put the modified gfp rfp back in place
        cropped_image[0, ...] = norm_gfp
        cropped_image[1, ...] = norm_rfp

        # Create an empty RGB image with the same shape as the input image
        rgb_image = np.zeros((cropped_image.shape[1], cropped_image.shape[2], 3), dtype=np.uint16)

        # Assign the first channel to the green channel of the RGB image
        rgb_image[:, :, 1] = cropped_image[0]

        # Assign the second channel to the red and blue channels of the RGB image to create magenta
        rgb_image[:, :, 0] = cropped_image[1]
        rgb_image[:, :, 2] = cropped_image[1]

        # scale down to 8bit
        rgb_image = np.uint8(rgb_image >> 8)

        if mask_outline:
            # load mask (singular)
            cropped_masks = segmentation[int(t), x_start:x_end, y_start:y_end]

            # Pad the cropped image if necessary
            cropped_masks = np.pad(cropped_masks, ((x_pad, x_pad_end), (y_pad, y_pad_end)), mode='constant')

            # extract only that segment
            seg_ID = cropped_masks[int(cropped_masks.shape[0] / 2), int(cropped_masks.shape[1] / 2)]
            instance_mask = (cropped_masks == seg_ID).astype(np.uint8)

            # draw outline
            contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb_image, contours, -1, (0, 2 ** 8, 2 ** 8), thickness=2)  # make 8bit

        # downsize image to reduce storage demands
        rgb_image = cv2.resize(rgb_image, (rgb_image.shape[1] // 2, rgb_image.shape[0] // 2))

        # Return the cropped RGB image
        return rgb_image
