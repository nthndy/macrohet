import glob
import logging
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

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
from skimage.io import imread, imsave
from skimage.transform import AffineTransform
from tqdm.auto import tqdm

from .dataio import read_harmony_metadata

# ignore shapely depreciation warning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
pkg_resources.require("Shapely<2.0.0")
# ignore error message for pandas new col assignment
pd.options.mode.chained_assignment = None
FilePath = Union[Path, str]
ArrayLike = Union[np.ndarray, "dask.array.Array"]

logging.basicConfig(level=logging.INFO)


class FileNotFoundError(Exception):
    """
    Custom exception raised when a file is not found.
    """
    pass


def find_files_exist(fns: List[str], image_dir: str):
    """
    Check if the given list of filenames exist in the specified directory.
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
    input_transforms: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None,
    set_plane: Optional[Any] = None,  # Can be int or 'max_proj'/'sum_proj'
    set_channel: Optional[int] = None,
    set_time: Optional[int] = None,
    overlap_percentage: float = 0.1,
) -> np.ndarray:
    """
    Uses the stitch function to compile a mosaic set of images that have been
    exported and fragmented from the Harmony software and returns a dask array
    that can be lazily loaded and stitched together on the fly.

    The function supports a variety of options for selecting specific planes,
    channels, and timepoints, as well as applying image transformations and
    conducting Z-projections.

    Parameters
    ----------
    image_dir : os.PathLike
        Location of fragmented images, typically located in a folder named
        "/Experiment_ID/Images" that was exported form the Harmony software.
    metadata : pd.DataFrame
        pd.DataFrame representation of the experiment metadata file, typically
        located in a file called "/Experiment_ID/Index.idx.xml". This metadata
        can be extracted and formatted using the `read_harmony_metadata`
        function in `utils.py`.
    row : str
        Each experiment will be conducted over a multiwell plate, so the row and
        column of the desired well needs to be defined as a string input. This
        input defines the row of choice.
    col : str
        Corresponding column of choice.
    input_transforms : Optional[List[Callable[[np.ndarray], np.ndarray]]]
        Optional pre-processing transformations that can be applied to each
        image, such as a crop, a transpose or a background removal. Defaults to
        None.
    set_plane : Optional[Any]
        Optional input to define a single plane to compile. If left blank then
        mosaic will be compiled over all planes available. Must have same index
        as filename or will return error.
        If a string input of 'max_proj' or 'sum_proj' is provided then images
        will be either taken as a max pixel value projection or summed
        projection over the Z axis.
    set_channel : Optional[int]
        Optional input to define a single channel to compile. If left blank then
        mosaic will be compiled over all channels available. Must have same
        index as filename or will return error.
    set_time : Optional[int]
        Optional input to define a single frame to compile. If left blank then
        mosaic will be compiled over all frames available. Must have same index
        as filename or will return error.
    overlap_percentage : float
        The percentage of overlap between tiles in the mosaic. This is used to
        calculate the size of the final stitched image. Must be a value between
        0 and 1, where 0 means no overlap and 1 means 100% overlap. Defaults to
        0.1 (10% overlap).

    Returns
    -------
    np.ndarray
        A lazily-loaded dask array representing the stitched image. The array
        can be converted to a NumPy array or saved to disk using dask's
        functionality.

    Raises
    ------
    ValueError
        If the specified row or column is not found in the metadata.
    TypeError
        If set_plane is a string but not 'max_proj' or 'sum_proj'.
    ValueError
        If the number of tiles is not a perfect square when calculating the size
        of the final stitched image.

    Notes
    -----
    The function is designed to be flexible and handle a variety of use cases,
    from loading a single plane, channel, or timepoint, to compiling a complete
    mosaic with Z-projections and image transformations. The use of dask allows
    for efficient memory usage and lazy loading, making it suitable for large
    image datasets.

    Examples
    --------
    Compile a mosaic from the images in 'image_dir', using the metadata in
    'metadata', for the well at row 1 and column 2:

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

    # set final chunk fraction according to how many tiles there are
    chunk_fraction = int(metadata['FieldID'].max())
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
    number_tiles = int(metadata['FieldID'].max())
    tile_size = int(metadata['ImageSizeX'].max())
    image_size = final_image_size(number_tiles, tile_size, overlap_percentage)

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
                                   chunk_fraction)[0]
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

    # stack them together
    images = np.stack(images, axis=0)

    # # reshape them according to TCZXY
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
           chunk_fraction: int) -> Tuple[da.Array, List[Tuple]]:
    """
    Function to stitch a single-frame/slice mosaic image together.

    Parameters
    ----------
    load_transform_image : partial function
        Function to load and apply transformations to an image.
    df : pd.DataFrame
        DataFrame containing all image metadata.
    image_dir : str
        Directory containing the images.
    time : int
        Time index.
    plane : int
        Z index.
    channel : int
        Channel index.
    row : int
        Row index of the FOV.
    col : int
        Column index of the FOV.
    chunk_fraction : int
        Number of Dask array chunks to divide the mosaic image into.

    Returns
    -------
    frame : dask.Array
        Stitched mosaic image.
    tiles_shifted_shapely : List[Tuple]
        Chunk information.
    """
    # logging.info('Entering function: stitch')

    # Filter metadata for the current mosaic
    conditions = (df['TimepointID'] == str(time)) & (df['PlaneID'] == str(plane)) & \
                 (df['ChannelID'] == str(channel)) & (df['Row'] == str(row)) & (df['Col'] == str(col))
    filtered_df = df[conditions]

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
    stitched_shape = tuple(np.ceil(all_bboxes.max(axis=0) - all_bboxes.min(axis=0)).astype(int))
    shift_to_origin = AffineTransform(translation=-all_bboxes.min(axis=0))
    transforms_with_shift = [t @ shift_to_origin.params for t in transforms]
    shifted_tiles = [transform_tile_coord(sample.shape, t) for t in transforms_with_shift]

    # Determine chunk size and boundaries
    chunk_size = (stitched_shape[0] / np.sqrt(chunk_fraction),) * 2
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
    frame = da.rot90(frame)  # Need this to bridge cartesian coords with python image coords

    # logging.info('Exiting function: stitch')
    return frame, tiles_shifted_shapely


def transform_tile_coord(shape: Tuple[int, int], affine_matrix: np.ndarray) -> np.ndarray:
    """
    Returns the corner coordinates of a 2D array after applying a transformation.

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


def get_chunk_coord(shape: Tuple[int, int], chunk_size: Tuple[int, int]) -> Iterator[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Iterator that returns the bounding coordinates for the individual chunks of a dask array.

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
    """
    Convert a numpy array of coordinates to a shapely object.

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


def get_rect_from_chunk_boundary(chunk_boundary: Tuple[Tuple[int, int], Tuple[int, int]]) -> np.ndarray:
    """
    Given a chunk boundary tuple, return a numpy array representing a rectangle.

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
    tiles_shapely: List[BaseGeometry],
    chunks_shapely: List[BaseGeometry]
) -> Dict[Tuple[int, int], List[Tuple[Union[str, np.ndarray], np.ndarray]]]:
    """
    For each output array chunk, find the intersecting image tiles.

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
    input_tile_info: Dict[Tuple[int, int], List[Tuple[Union[str, Path, np.ndarray], np.ndarray]]],
    imload_fn: Optional[Callable] = imread,
    block_info=None,
    dtype=np.uint16,
) -> np.ndarray:
    """
    Fuses the tiles that intersect the current chunk of a dask array using maximum projection.

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


def load_image(file: Union[str, Path], transforms: List[Callable[[np.ndarray], np.ndarray]] = None) -> np.ndarray:
    """
    Load image from given filepath with optional transformation implementation.

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


def final_image_size(num_tiles, size_of_tile, overlap_percentage):
    """
    Calculate the size of the final stitched image.

    Parameters:
    num_tiles (int): Total number of tiles (must be a perfect square).
    size_of_tile (int): Size of each tile in pixels.
    overlap_percentage (float): Overlap between the tiles as a percentage (0.1 for 10%).

    Returns:
    int: Size of the final stitched image in pixels.
    """
    # Check if num_tiles is a perfect square
    if not np.sqrt(num_tiles).is_integer():
        raise ValueError("The number of tiles must be a perfect square.")

    # Calculate the number of tiles along one dimension
    tiles_per_side = int(np.sqrt(num_tiles))

    # Calculate the actual overlap in pixels
    overlap = overlap_percentage * size_of_tile

    # Calculate the size of the final stitched image
    final_image_size = (tiles_per_side * size_of_tile) - ((tiles_per_side - 1) * overlap)

    return (int(final_image_size), int(final_image_size))


def compile_and_export_mosaic(image_dir: str, metadata_file_path: str, chunk_fraction=9):
    """
    Uses various functions to compile a more user-friendly experience of tiling
    a set of images that have been exported from the Harmony software.
    """
    fns = glob.glob(os.path.join(image_dir, '*.tiff'))
    print(len(fns), 'image files found')
    df = read_harmony_metadata(metadata_file_path)
    # extract some necessary information from the metadata before tiling
    channel_IDs = df['ChannelID'].unique()
    plane_IDs = df['PlaneID'].unique()
    timepoint_IDs = df['TimepointID'].unique()
    # set a few parameters for the tiling approach

    load_transform_image = partial(load_image, transforms=[])
    row_col_list = list()
    for index, row in (df.iterrows()):
        row_col_list.append(tuple((int(row['Row']), int(row['Col']))))
    row_col_list = list(set(row_col_list))
    for n, i in enumerate(row_col_list):
        print('Position index and (row,column):', n, i)
    # get user input for desired row and column
    print('Enter the row number you want:')
    row = input()
    print('Enter the column number you want:')
    col = input()
    print('Enter the output directory, or enter for Desktop output')
    output_directory = input()
    if output_directory == '':
        from datetime import datetime
        now = datetime.now()  # current date and time
        date_time = now.strftime("%m_%d_%Y")
        output_directory = f'Images_{date_time}'
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
    else:
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
    for time in tqdm(timepoint_IDs, leave=False, desc='Timepoint progress'):
        for channel in tqdm(channel_IDs, leave=False, desc='Channel progress'):
            for plane in tqdm(plane_IDs, leave=False, desc='Z-slice progress'):
                frame, chunk_info = stitch(load_transform_image,
                                           df,
                                           image_dir,
                                           time,
                                           plane,
                                           channel,
                                           row,
                                           col,
                                           chunk_fraction)
                fn = f'image_t{str(time).zfill(6)}_c{str(channel).zfill(4)}_z{str(plane).zfill(4)}.tiff'
                output_path = os.path.join(output_directory, fn)
                imsave(output_path, frame)
