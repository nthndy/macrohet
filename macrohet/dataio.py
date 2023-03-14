import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


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
    xml_data = open(metadata_path, 'r', encoding="utf-8-sig").read()
    root = ET.XML(xml_data)
    # extraction procedure for image volume metadata
    if not assay_layout:
        # extract the metadata from the xml file
        images_metadata = [child for child in root if "Images" in child.tag][0]
        # create an empty list for storing individual image metadata
        metadata = list()
        # iterate over every image entry extracting the metadata
        for image_metadata in tqdm(images_metadata, total=len(images_metadata),
                                   desc='Extracting HarmonyV5 metadata'):
            # create empty dict to store single image metadata
            single_image_dict = dict()
            # iterate over every metadata item in that image metadata
            for item in image_metadata:
                # get column names from metadata
                col = item.tag.replace('{http://www.perkinelmer.com/PEHH/HarmonyV5}', '')
                # get metadata
                entry = item.text
                # make dictionary out of metadata
                single_image_dict[col] = entry
            # append that image metadata to list of all images
            metadata.append(single_image_dict)
    # extraction procedure for assay layout metadata
    if assay_layout:
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

    # add names to assay layout indexing
    # df.index.set_names(['Row', 'Column'], inplace = True)

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
