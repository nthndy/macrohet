import os

import cv2
import numpy as np
from natsort import natsorted
from skimage.morphology import area_closing, label, remove_small_objects
from tqdm.auto import tqdm

# default scale value taken from harmony metadata
napari_scale = [1.49e-05, 1.4949402023919043E-7, 1.4949402023919043E-7]
# default scale factor
# for datasets that have been tracked on scaled down images
scale_factor = 6048 / 1200


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
                   napari_scale=napari_scale):
    """
    Puts a napari point layer around the cell of interest over all frames

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
    points = [[track.t[i], track.y[i] * scale_factor, track.x[i] * scale_factor]
              for i in range(len(track))]
    highlight = viewer.add_points(points, size=300,
                                  face_color='transparent',
                                  edge_color='white',
                                  edge_width=0.1,
                                  name=f'cell {cell_ID}',
                                  scale=napari_scale)
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
