import os

import btrack
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import napari
import numpy as np
import seaborn as sns
from magicgui import magicgui
from skimage.morphology import label
from skimage.transform import resize
from tqdm.auto import tqdm

from macrohet import dataio, tile
from macrohet.growth_model import euc_dist

from . import colours  # Import the colours from the macrohet module
from .colours import custom_colours

# default scale value taken from harmony metadata
napari_scale = [1.49e-05, 1.4949402023919043E-7, 1.4949402023919043E-7]
# default scale factor
# for datasets that have been tracked on scaled down images
scale_factor = 6048 / 1200


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
        """Highlights and displays a specific cell in Napari viewer with customizable visualization parameters.

        Parameters
        ----------
        cell_ID : int
            The identifier of the cell to be highlighted. Range: 0 to 1000.
        size : int
            Size of the symbol used to represent the cell. Range: 1 to 1000.
        opacity : float
            Opacity of the symbol. Range: 0.0 (fully transparent) to 1.0 (fully opaque).
        symbol : str
            Shape of the symbol representing the cell. Choices: 'o', 's', 't', '+', 'x'.
        edge_color : str
            Color of the symbol's edge. Choices include various common colors.
        edge_width : float
            Width of the symbol's edge. Range: 0.0 to 1.0.
        cell_property : str
            Specific property of the cell to display. If 'Show All' is selected, all properties are shown.

        Returns
        -------
        napari.types.LayerDataTuple
            A tuple containing the data for the Napari points layer. This includes the cell's coordinates,
            properties, and visual representation parameters like size, opacity, and color.

        Notes
        -----
        - The function uses a list of tracks to find the specific cell by its ID.
        - 'scale_factor' is applied to rescale the cell coordinates appropriately.
        - If 'Show All' is selected for cell_property, all properties of the cell are included.
          Otherwise, only the specified property is included. Special handling is applied
          if the property is 'mean_intensity'.

        """
        try:
            # Attempt to find the track with the given cell_ID
            track = next(track for track in tracks if track.ID == cell_ID)
        except StopIteration:
            # If no track is found with the given cell_ID, print an error message and return
            print(f"Error: No cell found with ID {cell_ID}")
            return

        # Get tracking position data
        data = np.array([[track.t[i], track.y[i] * tracking_scale_factor, track.x[i] * tracking_scale_factor]
                         for i in range(len(track))])

        # Display all track properties
        if cell_property == 'Show All':
            props = {cell_property: list(map(str, track.properties[cell_property]))
                     for cell_property in track.properties.keys()}
        # Or select a single track property to display
        else:
            props = {cell_property: list(map(str, track.properties[cell_property]))}

        # change position to where cell is
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


class ColorPalette:
    def __init__(self, color_map):
        self.colors = custom_colours[color_map]

    def replace(self, index, new_color):
        """Replace a color code at the specified index with a new color.

        Parameters
        ----------
            index (int): The index of the color code to replace.
            new_color (str): The new color code.

        Returns
        -------
            None

        """
        self.colors[index] = new_color


def color_palette(color_map):
    """Get the color palette of the specified color map.

    Parameters
    ----------
        color_map (str): The name of the color map.

    Returns
    -------
        ColorPalette: The color palette object.

    """
    return ColorPalette(color_map)




def upscale_labels_post_manual_annotation(labels, scale_factor):
    """Upscales labels after manual annotation to restore to original size.

    Parameters
    ----------
    labels (numpy.ndarray): The input labels to be upscaled.
    scale_factor (int): The scale factor for upscaling the labels.

    Returns
    -------
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
    """Downscale an image and its corresponding labels for manual annotation.

    Parameters
    ----------
        image (ndarray): The original image.
        labels (ndarray): The original labels.
        scale_factor (int): The scale factor for downsampling.

    Returns
    -------
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
    """Adds a rectangular grid overlay to a Napari viewer window.

    Parameters
    ----------
    viewer (napari.viewer.Viewer): A Napari viewer instance.
    N_rows_cols (int, optional): The number of divisions to divide the grid
        into. Default is 10.
    scale_factor (float, optional): A scaling factor applied to the grid size.
        Default is 1.
    edge_width (int, optional): The width of the grid lines. Default is 10.
    edge_color (str, optional): The color of the grid lines. Default is 'cyan'.

    Returns
    -------
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


def highlight_cell_fate(cell_ID, viewer, tracks,
                        scale_factor=scale_factor,
                        napari_scale=napari_scale):
    """Puts a napari point layer around the final frame of the cell of interest

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
    -------
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
    """Puts a Napari point layer around the cell of interest over all frames.

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
    -------
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