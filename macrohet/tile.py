import glob
import logging
import os
import warnings
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from typing import Any, Union

import dask
import dask.array as da
import numpy as np
import pandas as pd
import pkg_resources
from dask.array.core import normalize_chunks
from scipy.ndimage import affine_transform
from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon
from shapely.strtree import STRtree
from skimage.io import imread
from skimage.transform import AffineTransform

# ignore shapely depreciation warning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
pkg_resources.require("Shapely<2.0.0")
# ignore error message for pandas new col assignment
pd.options.mode.chained_assignment = None
FilePath = Path | str
ArrayLike = Union[np.ndarray, "dask.array.Array"]

logging.basicConfig(level=logging.INFO)


class FileNotFoundError(Exception):
    """Custom exception raised when a file is not found.
    """

    pass


def find_files_exist(fns: list[str], image_dir: str):
    """Check if the given list of filenames exist in the specified directory.
    logging.info('Entering function: find_files_exist')

    Parameters
    ----------
    fns : List[str]
        List of filenames to check.
    image_dir : str
        Directory where the files should exist.

    Raises
    ------
    FileNotFoundError
        If any of the files do not exist.

    """
    for fn in fns:
        file_path = os.path.join(image_dir, fn)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")


def compile_mosaic(
    image_dir: os.PathLike,
    metadata: pd.DataFrame,
    row: int,
    col: int,
    input_transforms: list[Callable[[np.ndarray], np.ndarray]] | None = None,
    set_plane: Any | None = None,  # Can be int or 'max_proj'/'sum_proj'
    set_channel: int | None = None,
    set_time: int | None = None,
    overlap_percentage: float = 0.1,
    subset_field_IDs: list[int] | None = None,
    n_tile_rows: int | None = None,
    n_tile_cols: int | None = None
) -> np.ndarray:
    """Compiles a mosaic of images from fragmented image files, typically exported
    from the Harmony software, and returns a dask array for on-the-fly stitching.

    This function is versatile, allowing for the stitching of any contiguous region
    of tiles, regardless of whether they form a square shape. It supports selective
    compilation based on planes, channels, timepoints, and applies optional image
    transformations. Handles Z-projections and efficient memory usage with dask.

    For non-square mosaics or when stitching specific regions within a well that
    may not include all image tiles, the `subset_field_IDs` parameter can be used
    to specify exactly which tiles to include. Additionally, for non-square shapes,
    `n_tile_rows` and `n_tile_cols` should be provided to define the mosaic's
    row and column dimensions.

    Parameters
    ----------
    image_dir : os.PathLike
        Directory containing fragmented images.
    metadata : pd.DataFrame
        DataFrame with image metadata.
    row : int
        Row index of the well in the multiwell plate.
    col : int
        Column index of the well.
    input_transforms : Optional[List[Callable]]
        List of functions for image preprocessing.
    set_plane : Optional[Any]
        Specific plane to compile, or 'max_proj'/'sum_proj' for Z-projections.
    set_channel : Optional[int]
        Specific channel to compile.
    set_time : Optional[int]
        Specific time frame to compile.
    overlap_percentage : float
        Overlap between tiles as a percentage.
    subset_field_IDs : Optional[List[int]]
        Specific field IDs to include in the compilation. Essential for stitching
        specific regions or non-square tile arrangements.
    n_tile_rows : Optional[int]
        Number of tile rows in the mosaic. Required for non-square mosaics.
    n_tile_cols : Optional[int]
        Number of tile columns in the mosaic. Required for non-square mosaics.

    Returns
    -------
    np.ndarray
        Dask array representing the stitched image.

    Raises
    ------
    ValueError
        If specified row or column is not in the metadata.
    TypeError
        If set_plane is not an integer, 'max_proj', or 'sum_proj'.
    ValueError
        If the number of tiles is not a perfect square (if n_tile_rows/cols not provided).

    Examples
    --------
    >>> image_dir = '/path/to/images'
    >>> metadata = pd.read_csv('/path/to/metadata.csv')
    >>> mosaic = compile_mosaic(image_dir, metadata, row=1, col=2)

    """
    # logging.info(f'Entering function: compile_mosaic\n Parameters: row, col, plane, channel, time {row, col, set_plane, set_channel, set_time} ')

    # check if specified row and column exists by checking metadata
    if str(row) not in metadata['Row'].unique():
        raise ValueError("Row not found in metadata.")
    if str(col) not in metadata['Col'].unique():
        raise ValueError("Column not found in metadata.")

    # check if projection is to be conducted over Z
    if isinstance(set_plane, str):
        # if set_plane is str, then specify which type of projection to
        # to conduct, also check that input type is accepted
        if set_plane not in ['max_proj', 'sum_proj']:
            raise TypeError("""Please specify either 'max_proj' or 'sum_proj'
                            if you want a projection over Z axis,
                            else specify 'set_plane' as an integer""")
        projection = set_plane
        set_plane = None
    else:
        projection = None
    # extract some necessary information from the metadata before tiling
    channel_IDs = (metadata['ChannelID'].unique()
                   if set_channel is None else [set_channel])
    plane_IDs = (metadata['PlaneID'].unique()
                 if set_plane is None else [set_plane])
    timepoint_IDs = (metadata['TimepointID'].unique()
                     if set_time is None else [set_time])

    # take a sample image to find dtype
    sample_fn = metadata['URL'][(metadata['Row'] == str(row))
                                & (metadata['Col'] == str(col))].iloc[0]
    dtype = imread(image_dir + f'/{sample_fn}').dtype

    # use metadata and overlap percentage to calculate the final expected size
    if subset_field_IDs:
        number_tiles = len(subset_field_IDs)
    else:
        number_tiles = int(metadata['FieldID'].max())

    # if n_tile_rows or cols are not supplied, then assumme mosaic is square
    if not n_tile_rows:
        n_tile_rows = n_tile_cols = np.sqrt(number_tiles)

    tile_size = int(metadata['ImageSizeX'].max())
    image_size = final_image_size(tile_size, overlap_percentage,
                                  n_tile_rows, n_tile_cols)

    load_transform_image = partial(load_image, transforms=input_transforms)

    # stitch the images together over all defined axis using dask delayed
    images = [dask.delayed(stitch)(load_transform_image,
                                   metadata,
                                   image_dir,
                                   time,
                                   plane,
                                   channel,
                                   str(row),
                                   str(col),
                                   n_tile_rows,
                                   n_tile_cols,
                                   subset_field_IDs)[0]
              for time in timepoint_IDs
              for channel in channel_IDs
              for plane in plane_IDs]

    # create a series of dask arrays out of the delayed funcs
    images = [da.from_delayed(frame,
                              shape=image_size,
                              dtype=dtype)
              # for frame in tqdm(images, desc='Stitching images together')]
              for frame in images]

    # rechunk so they are more managable along original image tile size
    images = [frame.rechunk(tile_size, tile_size) for frame in images]
    # stack them together and call compute so the it returns a single da and not a da of a da
    images = da.stack(images, axis=0)  # .compute()

    # reshape them according to TCZXY
    images = images.reshape((len(timepoint_IDs),
                             len(channel_IDs),
                             len(plane_IDs),
                             images.shape[-2], images.shape[-1]))
    # conduct projection according to specified type
    if projection == 'max_proj':
        images = np.max(images, axis=2)
    # sum projection requires image clipping in case px value exceeds dtype max
    elif projection == 'sum_proj':
        # Perform the summed projection along the z-axis
        summed_projection = np.sum(images, axis=2)

        # Determine the maximum value based on the data type
        max_value = np.iinfo(dtype).max

        # Clip the pixel values that exceed the maximum representable value
        images = np.clip(summed_projection, 0, max_value).astype(dtype)

    return images


def stitch(load_transform_image: partial,
           df: pd.DataFrame,
           image_dir: str,
           time: int,
           plane: int,
           channel: int,
           row: int,
           col: int,
           n_tile_rows: int,
           n_tile_cols: int,
           subset_field_IDs=None,) -> tuple[da.Array, list[tuple]]:
    """Function to stitch a single-frame/slice mosaic image together from individual image tiles.

    This function takes metadata for image tiles and their spatial coordinates, then
    stitches them together into a single large image. It supports selective stitching
    based on a subset of field IDs and handles non-square mosaics.

    Parameters
    ----------
    load_transform_image : partial function
        Function to load and apply transformations to an image.
    df : pd.DataFrame
        DataFrame containing all image metadata.
    image_dir : str
        Directory containing the images.
    time : int
        Time index for selecting the relevant images.
    plane : int
        Z-plane index for selecting the relevant images.
    channel : int
        Channel index for selecting the relevant images.
    row : int
        Row index of the Field of View (FOV).
    col : int
        Column index of the FOV.
    n_tile_rows : int
        Number of tile rows in the mosaic.
    n_tile_cols : int
        Number of tile columns in the mosaic.
    subset_field_IDs : list, optional
        List of field IDs to include in the stitching. If None, all fields are included.

    Returns
    -------
    frame : dask.Array
        Stitched mosaic image as a Dask array.
    tiles_shifted_shapely : List[Tuple]
        List of tuples containing chunk information and transformations for each tile.

    """
    # Filter metadata for the current mosaic
    conditions = (df['TimepointID'] == str(time)) & (df['PlaneID'] == str(plane)) & \
                 (df['ChannelID'] == str(channel)) & (df['Row'] == str(row)) & (df['Col'] == str(col))
    filtered_df = df[conditions]

    if subset_field_IDs:
        filtered_df = filtered_df[filtered_df['FieldID'].isin(subset_field_IDs)]

    # Extract filenames
    fns = filtered_df['URL']

    # Check if files exist
    find_files_exist(fns, image_dir)
    fns = [glob.glob(os.path.join(image_dir, fn))[0] for fn in fns]

    # Load and transform images
    sample = imread(fns[0])

    # Define function to fuse the image
    _fuse_func = partial(fuse_func, imload_fn=load_transform_image, dtype=sample.dtype)

    # Convert coordinates from standard units to pixels
    coords = filtered_df[["URL", "PositionX", "PositionY", "ImageResolutionX", "ImageResolutionY"]]
    coords['PositionXPix'] = coords['PositionX'].astype(float) / coords['ImageResolutionX'].astype(float)
    coords['PositionYPix'] = coords['PositionY'].astype(float) / coords['ImageResolutionY'].astype(float)

    norm_coords = list(zip(coords['PositionXPix'], coords['PositionYPix']))

    # Convert tile coordinates to transformation matrices and shift to the origin
    transforms = [AffineTransform(translation=stage_coord).params for stage_coord in norm_coords]
    tiles = [transform_tile_coord(sample.shape, transform) for transform in transforms]
    all_bboxes = np.vstack(tiles)
    stitched_shape = tuple(np.round(all_bboxes.max(axis=0) - all_bboxes.min(axis=0)).astype(int))
    shift_to_origin = AffineTransform(translation=-all_bboxes.min(axis=0))
    transforms_with_shift = [t @ shift_to_origin.params for t in transforms]
    shifted_tiles = [transform_tile_coord(sample.shape, t) for t in transforms_with_shift]

    # Determine chunk size and boundaries
    chunk_size = (int(stitched_shape[0] / n_tile_rows), int(stitched_shape[1] / n_tile_cols))
    chunks = normalize_chunks(chunk_size, shape=stitched_shape)
    assert np.all(np.array(stitched_shape) == np.array(list(map(sum, chunks)))), "Chunks do not fit into mosaic size"
    chunk_boundaries = list(get_chunk_coord(stitched_shape, chunk_size))

    # Use Shapely to find the intersection of the chunks
    tiles_shifted_shapely = [numpy_shape_to_shapely(s) for s in shifted_tiles]
    chunk_shapes = [get_rect_from_chunk_boundary(b) for b in chunk_boundaries]
    chunks_shapely = [numpy_shape_to_shapely(c) for c in chunk_shapes]

    # Build dictionary of chunk shape data with filenames and transformations
    for tile_shifted_shapely, file, transform in zip(tiles_shifted_shapely, fns, transforms_with_shift):
        tile_shifted_shapely.fuse_info = {'file': file, 'transform': transform}
    for chunk_shapely, chunk_boundary in zip(chunks_shapely, chunk_boundaries):
        chunk_shapely.fuse_info = {'chunk_boundary': chunk_boundary}

    chunk_tiles = find_chunk_tile_intersections(tiles_shifted_shapely, chunks_shapely)

    # Tile images together
    frame = da.map_blocks(func=_fuse_func, chunks=chunks, input_tile_info=chunk_tiles, dtype=sample.dtype)
    frame = da.rot90(frame)  # Need this to bridge cartesian coords with python image coords?

    return frame, tiles_shifted_shapely


def transform_tile_coord(shape: tuple[int, int], affine_matrix: np.ndarray) -> np.ndarray:
    """Returns the corner coordinates of a 2D array after applying a transformation.

    Parameters
    ----------
    shape : Tuple[int, int]
        The shape of the 2D array.
    affine_matrix : np.ndarray
        The affine transformation matrix.

    Returns
    -------
    np.ndarray
        Transformed corner coordinates.

    """
    h, w = shape
    baserect = np.array([[0, 0], [h, 0], [h, w], [0, w]])
    augmented_baserect = np.concatenate((baserect, np.ones((baserect.shape[0], 1))), axis=1)
    transformed_rect = (affine_matrix @ augmented_baserect.T).T[:, :-1]
    return transformed_rect


def get_chunk_coord(shape: tuple[int, int], chunk_size: tuple[int, int]) -> Iterator[tuple[tuple[int, int], tuple[int, int]]]:
    """Iterator that returns the bounding coordinates for the individual chunks of a dask array.

    Parameters
    ----------
    shape : Tuple[int, int]
        The shape of the dask array.
    chunk_size : Tuple[int, int]
        The size of the chunks.

    Returns
    -------
    Iterator[Tuple[Tuple[int, int], Tuple[int, int]]]
        Iterator over chunk boundaries.

    """
    chunksy, chunksx = normalize_chunks(chunk_size, shape=shape)
    y = 0
    for cy in chunksy:
        x = 0
        for cx in chunksx:
            yield ((y, y + cy), (x, x + cx))
            x += cx
        y += cy


def numpy_shape_to_shapely(coords: np.ndarray, shape_type: str = "polygon") -> BaseGeometry:
    """Convert a numpy array of coordinates to a shapely object.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates of the shape.
    shape_type : str, optional
        The type of the shape, by default "polygon".

    Returns
    -------
    BaseGeometry
        Shapely object.

    """
    _coords = coords[:, ::-1].copy()
    _coords[:, 1] *= -1
    if shape_type in ("rectangle", "polygon", "ellipse"):
        return Polygon(_coords)
    elif shape_type in ("line", "path"):
        return LineString(_coords)
    else:
        raise ValueError("Invalid shape type")


def get_rect_from_chunk_boundary(chunk_boundary: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    """Given a chunk boundary tuple, return a numpy array representing a rectangle.

    Parameters
    ----------
    chunk_boundary : Tuple[Tuple[int, int], Tuple[int, int]]
        Chunk boundary.

    Returns
    -------
    np.ndarray
        Rectangle coordinates.

    """
    ylim, xlim = chunk_boundary
    miny, maxy = ylim[0], ylim[1] - 1
    minx, maxx = xlim[0], xlim[1] - 1
    return np.array([[miny, minx], [maxy, minx], [maxy, maxx], [miny, maxx]])


def find_chunk_tile_intersections(
    tiles_shapely: list[BaseGeometry],
    chunks_shapely: list[BaseGeometry]
) -> dict[tuple[int, int], list[tuple[str | np.ndarray, np.ndarray]]]:
    """For each output array chunk, find the intersecting image tiles.

    Parameters
    ----------
    tiles_shapely : List[BaseGeometry]
        List of shapely objects corresponding to image tiles.
    chunks_shapely : List[BaseGeometry]
        List of shapely objects representing dask array chunks.

    Returns
    -------
    Dict[Tuple[int, int], List[Tuple[Union[str, np.ndarray], np.ndarray]]]
        Dictionary mapping chunk anchor points to tuples of image file names and their corresponding affine transform matrices.

    """
    chunk_to_tiles = {}
    tile_tree = STRtree(tiles_shapely)

    for chunk_shape in chunks_shapely:
        chunk_boundary = chunk_shape.fuse_info["chunk_boundary"]
        anchor_point = (chunk_boundary[0][0], chunk_boundary[1][0])
        intersecting_tiles = tile_tree.query(chunk_shape)
        chunk_to_tiles[anchor_point] = [
            ((t.fuse_info["file"], t.fuse_info["transform"]))
            for t in intersecting_tiles
        ]
    return chunk_to_tiles


def fuse_func(
    input_tile_info: dict[tuple[int, int], list[tuple[str | Path | np.ndarray, np.ndarray]]],
    imload_fn: Callable | None = imread,
    block_info=None,
    dtype=np.uint16,
) -> np.ndarray:
    """Fuses the tiles that intersect the current chunk of a dask array using maximum projection.

    Parameters
    ----------
    input_tile_info : Dict[Tuple[int, int], List[Tuple[Union[str, Path, np.ndarray], np.ndarray]]]
        Information about the input tiles.
    imload_fn : Optional[Callable], optional
        Function to load the images, by default imread.
    block_info : optional
        Information about the dask block, by default None.
    dtype : data-type, optional
        The desired data-type for the array, by default np.uint16.

    Returns
    -------
    np.ndarray
        Array of chunk-shape containing max projection of tiles falling into chunk.

    """
    array_location = block_info[None]["array-location"]
    anchor_point = (array_location[0][0], array_location[1][0])
    chunk_shape = block_info[None]["chunk-shape"]
    tiles_info = input_tile_info[anchor_point]
    fused = np.zeros(chunk_shape, dtype=dtype)

    for image_representation, tile_affine in tiles_info:
        if imload_fn is not None:
            tile_path = image_representation
            im = imload_fn(tile_path)
        else:
            im = image_representation

        shift = AffineTransform(translation=(-anchor_point[0], -anchor_point[1]))
        tile_shifted = affine_transform(
            im,
            matrix=np.linalg.inv(shift.params @ tile_affine),
            output_shape=chunk_shape,
            cval=0,
        )
        fused = np.maximum(fused, tile_shifted.astype(dtype))

    return fused


def load_image(file: str | Path, transforms: list[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
    """Load image from given filepath with optional transformation implementation.

    Parameters
    ----------
    file : Union[str, Path]
        Path to the image file.
    transforms : List[Callable[[np.ndarray], np.ndarray]], optional
        List of transformation functions to apply to the image, by default None.

    Returns
    -------
    np.ndarray
        Loaded and possibly transformed image.

    """
    try:
        img = imread(file)
    except Exception as e:
        raise Exception(f'{e} \n Could not load file: {file}') from e

    img = da.rot90(img, k=3)  # Need this to bridge cartesian coords with python image coords

    if transforms is not None:
        for transform in transforms:
            img = transform(img)

    return img


def final_image_size(size_of_tile, overlap_percentage, n_tile_rows, n_tile_cols):
    """Calculate the size of the final stitched image for a rectangular mosaic.

    Parameters
    ----------
    n_tile_rows (int): Number of tiles along the width.
    n_tile_cols (int): Number of tiles along the height.
    size_of_tile (int): Size of each tile in pixels.
    overlap_percentage (float): Overlap between the tiles as a percentage.

    Returns
    -------
    tuple: Size of the final stitched image in pixels (width, height).

    """
    # Calculate the actual overlap in pixels
    overlap = overlap_percentage * size_of_tile

    # Calculate the size of the final stitched image in width
    final_image_width = (n_tile_cols * size_of_tile) - ((n_tile_cols - 1) * overlap)

    # Calculate the size of the final stitched image in height
    final_image_height = (n_tile_rows * size_of_tile) - ((n_tile_rows - 1) * overlap)

    return (int(final_image_width), int(final_image_height))
