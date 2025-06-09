import btrack


def segment(
    frame,
    model,
    channels=(0, 0),
    diameter=350,
    min_size=5000,
    model_type='pretrained'
):
    """
    Segment a single image frame using a pretrained or custom model.

    Parameters:
        frame (ndarray): The input image.
        model (Cellpose model): A cellpose-compatible model object.
        channels (tuple): Input chan indices, e.g. (0, 0) for single-channel.
        diameter (int): Approximate object diameter.
        min_size (int): Minimum object size to retain.
        model_type (str): 'pretrained' or 'custom'.

    Returns:
        masks (ndarray): Label image of segmented objects.
    """
    if model_type == 'pretrained':
        masks, _, _, _ = model.eval(
            frame,
            channels=channels,
            diameter=diameter,
            min_size=min_size
        )
    else:
        masks, _, _ = model.eval(
            frame,
            channels=channels,
            diameter=diameter,
            min_size=min_size
        )
    return masks


def localise(
    masks,
    intensity_image,
    properties=('area', 'mean_intensity', 'orientation'),
    scale_factor=1.0,
    use_weighted_centroid=False
):
    """
    Extract single-cell objects and their properties from segmentation masks.

    Parameters:
        masks (ndarray): Label image of segmented objects.
        intensity_image (ndarray): Multichan intensity image for measurements.
        properties (tuple): Properties to extract (e.g. area, mean_intensity).
        scale_factor (float): Micron-per-pixel scale for spatial measurements.
        use_weighted_centroid (bool): Use intensity-weighted centroid if True.

    Returns:
        objects (list): List of btrack-like object instances.
    """
    objects = btrack.utils.segmentation_to_objects(
        segmentation=masks,
        intensity_image=intensity_image,
        properties=properties,
        scale=(scale_factor, scale_factor),
        use_weighted_centroid=use_weighted_centroid
    )
    return objects


def track(
    objects,
    masks,
    config_fn,
    scale_factor=1.0,
    search_radius=20
):
    """
    Run Bayesian tracking on localised single-cell objects.

    Parameters:
        objects (list): List of segmented objects with properties.
        masks (ndarray): Original label mask image for setting volume.
        config_fn (str): Path to btrack XML configuration file.
        scale_factor (float): Micron-per-pixel scale.
        search_radius (int): Maximum search radius between frames.

    Returns:
        tracks (list): List of Tracklet objects with time-resolved info.
    """
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
