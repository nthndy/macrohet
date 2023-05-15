import glob
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import dask
import dask.array as da
import numpy as np
import pandas as pd
import pkg_resources
import shapely
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


class FileNotFoundError(Exception):
    """
    Custom exception raised when a file is not found.
    """
    pass


def find_files_exist(fns: List[str], image_dir: str):
    """
    Check if the given list of filenames exist in the specified directory.

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


def compile_mosaic(image_directory: os.PathLike,
                   metadata: pd.DataFrame,
                   row: int,
                   col: int,
                   input_transforms: List[Callable[[ArrayLike], ArrayLike]] = None,
                   set_plane=None,
                   set_channel=None,
                   set_time=None
                   ) -> dask.array:

    """
    Uses the stitch function to compile a mosaic set of images that have been
    exported and fragmented from the Harmony software and returns a dask array
    that can be lazily loaded and stitched together on the fly.
    Latest iteration is attempting to use dask delay to improve speed of
    precompilation (WIP),

    Parameters
    ----------
    image_directory : os.PathLike
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
    input_transforms : List[Callable[[ArrayLike], ArrayLike]]
        Optional pre-processing transformations that can be applied to each
        image, such as a crop, a transpose or a background removal.
    set_plane : int or str
        Optional input to define a single plane to compile. If left blank then
        mosaic will be compiled over all planes available. Must have same index
        as filename or will return error.
        If a string input of 'max_proj' or 'sum_proj' is provided then images
        will be either taken as a max pixel value projection or summed
        projection over the Z axis.
    set_channel : int
        Optional input to define a single channel to compile. If left blank then
        mosaic will be compiled over all channels available. Must have same
        index as filename or will return error.
    set_time : int
        Optional input to define a single frame to compile. If left blank then
        mosaic will be compiled over all frames available. Must have same index
        as filename or will return error.
    projection : str
        Specify a string input of 'max_proj' or 'sum_proj' if you want to
        conduct a projection along the
    """
    # check if projection is to be conducted over Z
    if type(set_plane) == str:
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
    # set a few parameters for the tiling approach, remove this hardcoded val
    chunk_fraction = 9
    load_transform_image = partial(load_image, transforms=input_transforms)

    # stitch the images together over all defined axis
    # but do so in using dask delayed
    images = [dask.delayed(stitch)(load_transform_image,
                                   metadata,
                                   image_directory,
                                   time,
                                   plane,
                                   channel,
                                   str(row),
                                   str(col),
                                   chunk_fraction,
                                   mask=False)[0]
              for time in timepoint_IDs
              for channel in channel_IDs
              for plane in plane_IDs]

    # need to remove hardcoded values, perhaps by returning shapely info from
    # stitch function
    # create a series of dask arrays out of the delayed funcs
    images = [da.from_delayed(frame,
                              shape=(6048, 6048),
                              dtype=np.uint16)
              for frame in images]
    # rechunk so they are more managable
    images = [frame.rechunk(2016, 2016) for frame in images]
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
    if projection == 'sum_proj':
        images = np.sum(images, axis=2)

    return images


def compile_and_export_mosaic(image_directory: str, metadata_file_path: str):
    """
    Uses various functions to compile a more user-friendly experience of tiling
    a set of images that have been exported from the Harmony software.
    """
    fns = glob.glob(os.path.join(image_directory, '*.tiff'))
    print(len(fns), 'image files found')
    df = read_harmony_metadata(metadata_file_path)
    # extract some necessary information from the metadata before tiling
    channel_IDs = df['ChannelID'].unique()
    plane_IDs = df['PlaneID'].unique()
    timepoint_IDs = df['TimepointID'].unique()
    # set a few parameters for the tiling approach
    chunk_fraction = 9
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
                                           image_directory,
                                           time,
                                           plane,
                                           channel,
                                           row,
                                           col,
                                           chunk_fraction,
                                           mask=False)
                fn = f'image_t{str(time).zfill(6)}_c{str(channel).zfill(4)}_z{str(plane).zfill(4)}.tiff'
                output_path = os.path.join(output_directory, fn)
                imsave(output_path, frame)


def compile_mask_mosaic(image_directory: str, metadata: pd.DataFrame, row: str,
                        col: str, input_transforms=None):
    """
    Uses various functions to compile a mosaic set of images that have been
    exported from the Harmony software and returns a dask array that can be
    lazily loaded and stitched together on the fly in napari
    """
    # extract some necessary information from the metadata before tiling
    plane_IDs = metadata['PlaneID'].unique()
    timepoint_IDs = metadata['TimepointID'].unique()
    # set a few parameters for the tiling approach
    chunk_fraction = 9
    load_transform_image = partial(load_image, transforms=input_transforms)
    # using a dummy input for channel as masks will be based off this atm
    channel = '1'
    # clear empty arrays for organsing into dask arrays
    t_stack = []
    for time in tqdm(timepoint_IDs, leave=False, desc='Timepoint progress'):
        z_stack = []
        for plane in tqdm(plane_IDs, leave=False, desc='Z-slice progress'):
            frame, chunk_info = stitch(load_transform_image,
                                       metadata,
                                       image_directory,
                                       time,
                                       plane,
                                       channel,
                                       str(row),
                                       str(col),
                                       chunk_fraction,
                                       mask=True)
            # collect stitched frames together into time stack
            z_stack.append(frame)
        # stack together timewise
        t_stack.append(z_stack)
    # stack stitched dask arrays together into multidim image volumes
    masks = da.stack([da.stack(c_stack, axis=0) for c_stack in t_stack])

    return masks


def transform_tile_coord(shape: Tuple[int, int],
                         affine_matrix: np.ndarray) -> np.ndarray:
    """
    returns the corner coordinates of a 2D array with shape shape
    after applying the transform represented by affine_matrix.
    From DaskFusion (https://github.com/VolkerH/DaskFusion/)
    """
    h, w = shape
    # create homogeneous coordinates for corner points
    baserect = np.array([[0, 0], [h, 0], [h, w], [0, w]])
    augmented_baserect = np.concatenate(
        (baserect, np.ones((baserect.shape[0], 1))), axis=1
    )
    # see where the corner points map to
    transformed_rect = (affine_matrix @ augmented_baserect.T).T[:, :-1]
    return transformed_rect


def get_chunk_coord(shape: Tuple[int, int], chunk_size: Tuple[int, int]):
    """Iterator that returns the bounding coordinates
    for the individual chunks of a dask array of size
    shape with chunk size chunk_size.

    return_np_slice determines the output format. If True,
    a numpy slice object is returned for each chunk, that can be used
    directly to slice a dask array to return the desired chunk region.
    If False, a Tuple of Tuples ((row_min, row_max+1),(col_min, col_max+1))
    is returned.
    From DaskFusion (https://github.com/VolkerH/DaskFusion/)

    """
    chunksy, chunksx = normalize_chunks(chunk_size, shape=shape)
    y = 0
    for cy in chunksy:
        x = 0
        for cx in chunksx:
            yield ((y, y + cy), (x, x + cx))
            x = x + cx
        y = y + cy


def numpy_shape_to_shapely(coords: np.ndarray, shape_type: str = "polygon") -> BaseGeometry:
    """
    Convert an individual shape represented as a numpy array of coordinates
    to a shapely object
    From DaskFusion (https://github.com/VolkerH/DaskFusion/)
    """
    _coords = coords[:, ::-1].copy()  # shapely has col,row order, numpy row,col
    _coords[:, 1] *= -1  # axis direction flipped between shapely and napari
    if shape_type in ("rectangle", "polygon", "ellipse"):
        return Polygon(_coords)
    elif shape_type in ("line", "path"):
        return LineString(_coords)
    else:
        raise ValueError


def get_rect_from_chunk_boundary(chunk_boundary):
    """given a chunk boundary tuple, return a numpy
    array that can be added as a shape to napari"
    From DaskFusion (https://github.com/VolkerH/DaskFusion/)
    """
    ylim, xlim = chunk_boundary
    miny, maxy = ylim[0], ylim[1] - 1
    minx, maxx = xlim[0], xlim[1] - 1
    return np.array([[miny, minx], [maxy, minx], [maxy, maxx], [miny, maxx]])


def find_chunk_tile_intersections(
    tiles_shapely: List["shapely.geometry.base.BaseGeometry"],
    chunks_shapely: List["shapely.geometry.base.BaseGeometry"],
) -> Dict[Tuple[int, int], Tuple[str, np.ndarray]]:
    """
    For each output array chunk, find the intersecting image tiles

    Args:
        tile_shapes: Contains the shapely objects corresponding to transformed
                    image outlines. Each shape in tile_shapes must have a
                    .fuse_info dictionary with keys "file" and "transform".
        chunk_shapes: Contains the shapely objects representing dask array
                    chunks. Each shape in chunk_shapes must have a .fuse_info
                    dictionary with key "chunk_boundary", containing a tuple of
                    chunk boundaries

    Returns:
         The chunk_to_tiles dictionary, which has the chunk anchor points as
         keys and tuples of image file names and their corresponding affine
         transform matrix as values.

    From DaskFusion (https://github.com/VolkerH/DaskFusion/)
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


def fuse_func(input_tile_info: Dict[
              Tuple[int, int], List[Tuple[Union[str, Path, np.ndarray], np.ndarray]]
              ],
              imload_fn: Optional[Callable] = imread,
              block_info=None,
              dtype=np.uint16,
              ) -> np.ndarray:

    """
    Fuses the tiles that intersect the current chunk of a dask array using
    maximum projection.

    Pass this function to dask.array.map_blocks, after partial evaluation of the
    required image_folder and (if needed) optional arguments.

    Returns:
        Array of chunk-shape containing max projection of tiles falling into
        chunk

    From DaskFusion (https://github.com/VolkerH/DaskFusion/)
    """
    array_location = block_info[None]["array-location"]
    # The anchor point is the key to the input_tile_info dictionary
    anchor_point = (array_location[0][0], array_location[1][0])
    chunk_shape = block_info[None]["chunk-shape"]
    tiles_info = input_tile_info[anchor_point]
    # print(f"Processing chunk at {anchor_point}")
    fused = np.zeros(chunk_shape, dtype=dtype)
    for image_representation, tile_affine in tiles_info:
        if imload_fn is not None:
            # When imload_fn is provided we assume we have been given strings representing files
            tile_path = image_representation
            im = imload_fn(tile_path)
        else:
            # Without imload function we assume images are passed
            im = image_representation
        shift = AffineTransform(translation=(-anchor_point[0], -anchor_point[1]))
        tile_shifted = affine_transform(
            im,
            matrix=np.linalg.inv(shift.params @ tile_affine),
            output_shape=chunk_shape,
            cval=0,
        )
        # note that the dtype comversion here happens without scaling
        # may want to use one of the skimage.img_as_* functions instead
        stack = np.stack([fused, tile_shifted.astype(dtype)])
        fused = np.max(stack, axis=0)
    return fused


def load_image(file: FilePath,
               transforms: List[Callable[[ArrayLike], ArrayLike]] = None
               ) -> np.ndarray:
    """
    Load image from given filepath with optional transformation implementation.
    """

    try:
        img = imread(file)
    except Exception as e:
        raise Exception(f'{e} \n Could not load file: {file}')
    # TO-DO remove this hacky fix
    # WHY IS THIS NECESSARY FOR PROPER TILING
    img = np.rot90(img, k=3)
    # if img.ndim == 2:
    #    img = np.expand_dims(img, axis=0)
    if transforms is not None:
        for t in transforms:
            img = t(img)
    return img


def stitch(load_transform_image: partial,
           df: pd.DataFrame,
           image_dir: os.PathLike,
           time: int,
           plane: int,
           channel: int,
           row: int,
           col: int,
           chunk_fraction: int,
           mask: bool) -> dask.array:
    """
    Function that takes DaskFusion core elements, defined above and uses them to
    stitch a single-frame/slice mosaic image together.

    Parameters
    ----------
    load_transform_image: partial function
        Partial function that loads the image along with any transformations
    df : pd.DataFrame
        Pandas DataFrame containing all of the image metadata, taken from
        utils.read_harmony_metadata
    time : int
        Time index, taken from utils.read_harmony_metadata dataframe output.
    plane : int
        Z index, taken from utils.read_harmony_metadata dataframe output.
    channel : int
        Channel index, taken from utils.read_harmony_metadata dataframe output.
    row : int
        Row index, taken from utils.read_harmony_metadata dataframe output.
        Encodes the row of the FOV that you want to tile into a mosaic.
    col : int
        Column index, taken from utils.read_harmony_metadata dataframe output.
        Encodes the row of the FOV that you want to tile into a mosaic.
    chunk_fraction : int
        How many Dask array chunks you want to divide the mosaic image into,
        must be a square number as the images are symmetric.
    mask : bool
        If true then the filenames are replaced with ch99 enumeration, which
        corresponds to a mask image for the set time and plane
        THIS IS THE WRONG APPROACH FOR MASKS...

    Returns
    -------
    frame : dask.array
        An stitched mosaic image for the given indices provided as the params,
        but only actually stitched together when the image is calculated.
    tiles_shifted_shapely: list of shapely.geometry.polygon.Polygon
        Chunk information can be extracted to save out individual mask tiles.
    """

    # extract metadata for this mosaic
    filtered_df = df[(df['TimepointID'] == str(time))
                     & (df['PlaneID'] == str(plane))
                     & (df['ChannelID'] == str(channel))
                     & (df['Row'] == str(row))
                     & (df['Col'] == str(col))]
    # extract filenames for subset
    fns = filtered_df['URL']
    # if you want to stitch the masks together then only extract the mask fns
    if mask:
        # check that all masks are present
        masks_exist = all([os.path.exists(os.path.join(image_dir, fn)) for fn in
                           fns.str.replace(r'ch(\d+)', 'ch99', regex=True)])
        assert masks_exist is True, "Cannot find all corresponding masks"
        # this is achieved by replacing the
        fns = fns.str.replace(r'ch(\d+)', 'ch99', regex=True)
    # check if files have actually be found
    find_files_exist(fns, image_dir)
    # build into full file path
    fns = [glob.glob(os.path.join(image_dir, fn))[0] for fn in fns]
    # stack single slice mosaic into lazy array
    sample = imread(fns[0])
    lazy_arrays = [dask.delayed(imread)(fn) for fn in fns]
    lazy_arrays = [da.from_delayed(x, shape=sample.shape, dtype=sample.dtype)
                   for x in lazy_arrays]

    # define the function to fuse the image
    _fuse_func = partial(fuse_func,
                         imload_fn=load_transform_image,
                         dtype=sample.dtype)

    # extract and convert coordinates from standard units into pixels
    coords = filtered_df[["URL", "PositionX", "PositionY", "PositionZ",
                          "ImageResolutionX", "ImageResolutionY"]]
    coords['PositionXPix'] = (coords['PositionX'].astype(float)) / (coords['ImageResolutionX']).astype(float)
    coords['PositionYPix'] = (coords['PositionY'].astype(float)) / (coords['ImageResolutionY']).astype(float)
    if mask:
        # needs more attn
        coords['PositionXPix'] = coords['PositionXPix'] * 1.1
        coords['PositionYPix'] = coords['PositionYPix'] * 1.1
    norm_coords = list(zip(coords['PositionXPix'], coords['PositionYPix']))
    # convert tile coordinates into transformation matrices
    transforms = [AffineTransform(translation=stage_coord).params
                  for stage_coord in norm_coords]
    tiles = [transform_tile_coord(sample.shape, transform)
             for transform in transforms]
    # shift the tile coordinates to the origin
    all_bboxes = np.vstack(tiles)
    all_min = all_bboxes.min(axis=0)
    all_max = all_bboxes.max(axis=0)
    stitched_shape = tuple(np.ceil(all_max - all_min).astype(int))
    shift_to_origin = AffineTransform(translation=-all_min)
    transforms_with_shift = [t @ shift_to_origin.params for t in transforms]
    shifted_tiles = [transform_tile_coord(sample.shape, t)
                     for t in transforms_with_shift]
    # decide on chunk size as a fraction of total slice size
    # TODO: auto size, assuming symmetric atm
    chunk_size = (stitched_shape[0] / np.sqrt(chunk_fraction),
                  stitched_shape[0] / np.sqrt(chunk_fraction))
    chunks = normalize_chunks(chunk_size, shape=tuple(stitched_shape))
    # check the maths adds up correctly (chunks fit into mosaic)
    computed_shape = np.array(list(map(sum, chunks)))
    assert np.all(np.array(stitched_shape) == computed_shape)
    # get boundary coords of chunks
    chunk_boundaries = list(get_chunk_coord(stitched_shape, chunk_size))
    # use shapely to find the intersection of the chunks
    tiles_shifted_shapely = [numpy_shape_to_shapely(s) for s in shifted_tiles]
    chunk_shapes = list(map(get_rect_from_chunk_boundary, chunk_boundaries))
    chunks_shapely = [numpy_shape_to_shapely(c) for c in chunk_shapes]
    # build dictionary of chunk shape data with filenames and transformations
    for tile_shifted_shapely, file, transform in zip(tiles_shifted_shapely,
                                                     fns,
                                                     transforms_with_shift):
        tile_shifted_shapely.fuse_info = {'file': file,
                                          'transform': transform}
    for chunk_shapely, chunk_boundary in zip(chunks_shapely,
                                             chunk_boundaries):
        chunk_shapely.fuse_info = {'chunk_boundary': chunk_boundary}
    chunk_tiles = find_chunk_tile_intersections(tiles_shifted_shapely,
                                                chunks_shapely)
    # tile images together
    frame = da.map_blocks(func=_fuse_func,
                          chunks=chunks,
                          input_tile_info=chunk_tiles,
                          dtype=sample.dtype)

    # TO-DO remove this hacky fix
    # WHY IS THIS NECESSARY FOR PROPER TILING
    frame = np.rot90(frame)

    return frame, tiles_shifted_shapely


def legacy_compile_mosaic(image_directory: os.PathLike,
                          metadata: pd.DataFrame,
                          row: int,
                          col: int,
                          input_transforms: List[Callable[[ArrayLike], ArrayLike]] = None,
                          set_plane=None,
                          set_channel=None,
                          set_time=None
                          ) -> dask.array:

    """
    Uses the LEGACY/OLDER stitch function to compile a mosaic set of images that have been
    exported and fragmented from the Harmony software and returns a dask array
    that can be lazily loaded and stitched together on the fly.
    Latest iteration is attempting to use dask delay to improve speed of
    precompilation (WIP),

    Parameters
    ----------
    image_directory : os.PathLike
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
    input_transforms : List[Callable[[ArrayLike], ArrayLike]]
        Optional pre-processing transformations that can be applied to each
        image, such as a crop, a transpose or a background removal.
    set_plane : int
        Optional input to define a single plane to compile. If left blank then
        mosaic will be compiled over all planes available.
    set_channel : int
        Optional input to define a single channel to compile. If left blank then
        mosaic will be compiled over all channels available.
    set_time : int
        Optional input to define a single frame to compile. If left blank then
        mosaic will be compiled over all frames available.
    """
    # extract some necessary information from the metadata before tiling
    channel_IDs = (metadata['ChannelID'].unique()
                   if set_channel is None else [set_channel])
    plane_IDs = (metadata['PlaneID'].unique()
                 if set_plane is None else [set_plane])
    timepoint_IDs = (metadata['TimepointID'].unique()
                     if set_time is None else [set_time])
    # set a few parameters for the tiling approach
    chunk_fraction = 9
    load_transform_image = partial(load_image, transforms=input_transforms)
    # stitch the images together over all defined axis
    images = [stitch(load_transform_image,
                     metadata,
                     image_directory,
                     time,
                     plane,
                     channel,
                     str(row),
                     str(col),
                     chunk_fraction,
                     mask=False)[0]
              for plane in tqdm(plane_IDs, leave=False)
              for channel in tqdm(channel_IDs, leave=False)
              for time in tqdm(timepoint_IDs, leave=False)]
    # stack them together
    images = da.stack(images, axis=0)
    # reshape them according to TCZXY
    images = images.reshape((len(timepoint_IDs),
                             len(channel_IDs),
                             len(plane_IDs),
                             images.shape[-2], images.shape[-1]))

    return images
