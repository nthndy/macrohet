import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from lxml import etree as ET_iter
from tqdm.auto import tqdm


def generate_url(row):
    """
    Generate a properly formatted local file address for the 'URL' column in Harmony metadata.
    This function replaces remote addresses, ensuring consistency when metadata is exported separately from the images.

    Parameters:
    row (pd.Series): A row of Harmony metadata containing 'Row', 'Col', 'FieldID', 'PlaneID', 'ChannelID', 'TimepointID', and 'FlimID' columns.

    Returns:
    str: The formatted local file address.
    """
    m_row = row['Row'].zfill(2)
    m_col = row['Col'].zfill(2)
    m_field = row['FieldID'].zfill(2)
    m_plane = row['PlaneID'].zfill(2)
    m_ch = row['ChannelID']
    m_time = int(row['TimepointID']) + 1
    m_flim = row['FlimID']
    return f'r{m_row}c{m_col}f{m_field}p{m_plane}-ch{m_ch}sk{m_time}fk1fl{m_flim}.tiff'


def load_macrohet_metadata(location='desktop'):
    """
    Lazy function for loading a couple of bits of info that usually take the
    first couple of cells to load
    """

    if location == 'desktop':
        base_dir = '/mnt/DATA/sandbox/pierre_live_cell_data/outputs/Replication_IPSDM_GFP/'
    else:
        base_dir = '/Volumes/lab-gutierrezm/home/users/dayn/macrohet/'
    metadata_fn = os.path.join(base_dir, 'macrohet_images/Index.idx.xml')
    metadata = read_harmony_metadata(metadata_fn)
    metadata_path = os.path.join(base_dir, 'macrohet_images/Assaylayout/20210602_Live_cell_IPSDMGFP_ATB.xml')
    assay_layout = read_harmony_metadata(metadata_path,
                                         assay_layout=True,)

    return metadata, assay_layout


def track_to_df(track):
    """
    Quick hack to return a single track as a dataframe for output into excel
    """

    return pd.DataFrame(track.to_dict(), columns=list(track.to_dict().keys()))


def read_harmony_metadata(metadata_path: os.PathLike, assay_layout=False,
                          mask_exist=False, image_dir=None, image_metadata=None
                          ) -> pd.DataFrame:
    """
    Read the metadata from the Harmony software for the Opera Phenix microscope.
    Takes an input of the path to the metadata .xml file.
    Returns the metadata in a pandas dataframe format.
    If assay_layout is True then alternate xml format is anticipated, returning
    information about the assay layout of the experiment rather than the general
    organisation of image volume.
    If mask_exist is True then the existence of masks will be checked, which the
    image directory (image_dir) is required with the image metadata
    (image_metadata)
    """
    # read xml metadata file
    print('Reading metadata XML file...')

    # extraction procedure for image volume metadata
    if not assay_layout:
        metadata = []

        # for the large image metadata file using iterative reading of metadata
        for event, elem in tqdm(ET_iter.iterparse(metadata_path, events=("end",))):
            # Check for the 'Images' tag in the element
            if event == "end" and "Images" in elem.tag:
                for image_metadata in elem:
                    single_image_dict = {}
                    for item in image_metadata:
                        # Extract column name, removing namespace if necessary
                        col = item.tag.split('}')[-1]  # This splits the tag by '}' and takes the last part
                        # Get metadata value
                        entry = item.text
                        # Store in dictionary
                        single_image_dict[col] = entry

                    # Append to list
                    metadata.append(single_image_dict)

                # Clear processed element to free memory
                elem.clear()

    # extraction procedure for assay layout metadata
    if assay_layout:
        xml_data = open(metadata_path, 'r', encoding="utf-8-sig").read()
        root = ET.XML(xml_data)
        metadata = dict()
        for branch in root:
            for subbranch in branch:
                if subbranch.text.strip() and subbranch.text.strip() != 'string':
                    col_name = subbranch.text
                    metadata[col_name] = dict()
                for subsubbranch in subbranch:
                    if 'Row' in subsubbranch.tag:
                        row = int(subsubbranch.text)
                    elif 'Col' in subsubbranch.tag and 'Color' not in subsubbranch.tag:
                        col = int(subsubbranch.text)
                    if 'Value' in subsubbranch.tag and subsubbranch.text is not None:
                        val = subsubbranch.text
                        metadata[col_name][int(row), int(col)] = val

    # create a dataframe out of all metadata
    df = pd.DataFrame(metadata)

    if assay_layout and mask_exist:
        df['Missing masks'] = np.nan
        for index, row in df.iterrows():
            row, col = index
            missing_mask_dict = do_masks_exist(image_dir, image_metadata,
                                               row=row, col=col,
                                               print_output=False)
            df.at[(row, col), 'Missing masks'] = missing_mask_dict[row, col]
            df = df.where(pd.notnull(df), None)

    # final few aesthetic touches to assay layout
    if assay_layout:
        # add names to assay layout indexing
        df.index.set_names(['Row', 'Column'], inplace=True)
        # clearing few hacky errors in some recent assay layout
        if 'Cell Count' in df.columns:
            if pd.isna(df['Cell Count']).any():
                df.drop(columns='Cell Count', inplace=True)
        if 'double' in df.columns:
            df.rename(columns={'double': 'Cell Count'}, inplace=True)

    print('Extracting metadata complete!')
    return df


def do_masks_exist(image_dir, metadata, row=None, col=None, print_output=True):
    """
    Iterates over all positions in experiment and checks if masks have been
    created for each individual tiled image, returns missing mask info as dict()
    If row and col are not defined then iterates over all found instances
    """
    missing_mask_dict = dict()
    if None in [row, col]:
        row_col_list = list()
        for index, row in metadata.iterrows():
            row_col_list.append(tuple((int(row['Row']), int(row['Col']))))
        row_col_list = list(set(row_col_list))
        for row, col in row_col_list:
            channel = '1'
            input_img_fns = metadata[(metadata['Row'] == str(row))
                                     & (metadata['Col'] == str(col))
                                     & (metadata['ChannelID'] == channel)]['URL']
            corresponding_mask_fns = input_img_fns.str.replace(r'ch(\d+)', 'ch99')
            # input_paths = [os.path.join(image_dir, fn) for fn in input_img_fns]
            mask_paths = [os.path.join(image_dir, fn) for fn in corresponding_mask_fns]
            masks_exist = all([os.path.exists(fn) for fn in mask_paths])
            if not masks_exist:
                missing_masks = [fn for fn in mask_paths if not os.path.exists(fn)]
                print(f'{len(missing_masks)} masks are missing for row, col {row, col}')
                missing_mask_dict[row, col] = len(missing_masks), missing_masks
            else:
                print(f'All masks present and correct for row, col {row, col}')
                missing_mask_dict[row, col] = None
        return missing_mask_dict
    else:
        channel = '1'
        input_img_fns = metadata[(metadata['Row'] == str(row))
                                 & (metadata['Col'] == str(col))
                                 & (metadata['ChannelID'] == channel)]['URL']
        corresponding_mask_fns = input_img_fns.str.replace(r'ch(\d+)', 'ch99')
        # input_paths = [os.path.join(image_dir, fn) for fn in input_img_fns]
        mask_paths = [os.path.join(image_dir, fn) for fn in corresponding_mask_fns]
        masks_exist = all([os.path.exists(fn) for fn in mask_paths])
        if not masks_exist:
            missing_masks = [fn for fn in mask_paths if not os.path.exists(fn)]
            if print_output is True:
                print(f'{len(missing_masks)} masks are missing for row, col {row, col}')
            missing_mask_dict[row, col] = len(missing_masks), missing_masks
        else:
            if print_output is True:
                print(f'All masks present and correct for row, col {row, col}')
            missing_mask_dict[row, col] = None
        return missing_mask_dict
