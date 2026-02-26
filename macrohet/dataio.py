import io
import json
import os
import xml.etree.ElementTree as ET
import zipfile
import h5py
import zarr
import btrack
from pathlib import Path
import numpy as np
import pandas as pd
from lxml import etree as ET_iter
from tqdm.auto import tqdm


def export_tracks_to_h5(tracks, output_fn, obj_type='obj_type_1'):
    """Writes btrack tracks to HDF5 and injects placeholder dummies to prevent reload bugs."""
    with btrack.io.HDF5FileHandler(output_fn, 'w', obj_type=obj_type) as writer:
        writer.write_tracks(tracks)

    with h5py.File(output_fn, 'a') as f:
        tracks_grp = f[f'tracks/{obj_type}']
        if 'dummies' not in tracks_grp:
            refs = tracks_grp['tracks'][:]
            min_ref = int(np.min(refs))
            if min_ref < 0:
                num_dummies = abs(min_ref)
                dummy_data = np.zeros((num_dummies, 5), dtype=np.float32)
                for i in range(num_dummies):
                    dummy_data[i, 0] = -(i + 1)
                tracks_grp.create_dataset('dummies', data=dummy_data)

def export_tracks_to_zarr(tracks, zarr_path, component="tracks/0"):
    """Packs btrack tracks and features into an OME-NGFF compatible Zarr store."""
    track_data, area_list, gfp_list, rfp_list, mtb_area_list, infected_list = [], [], [], [], [], []
    
    for track_id, track in enumerate(tracks):
        n_frames = len(track)
        ids = np.full(n_frames, track_id)
        coords = np.column_stack((ids, track.t, track.y, track.x))
        track_data.append(coords)
        
        area_list.extend(track.properties["area"])
        gfp_list.extend(track.properties["mean_intensity"][0])
        rfp_list.extend(track.properties["mean_intensity"][1])
        mtb_area_list.extend(track.properties["Mtb area px"])
        infected_list.extend(track.properties["Infected"])

    track_array = np.vstack(track_data).astype(np.float32)
    features = {
        "area": np.array(area_list, dtype=np.float32),
        "gfp_intensity": np.array(gfp_list, dtype=np.float32),
        "rfp_intensity": np.array(rfp_list, dtype=np.float32),
        "mtb_area_px": np.array(mtb_area_list, dtype=np.float32),
        "infected": np.array(infected_list, dtype=bool)
    }

    store = zarr.open(zarr_path, mode="a")
    tracks_grp = store.require_group(component)
    tracks_grp.create_dataset("track_data", data=track_array, compressor=zarr.Blosc(), overwrite=True)
    
    feat_grp = tracks_grp.require_group("features")
    for key, arr in features.items():
        feat_grp.create_dataset(key, data=arr, compressor=zarr.Blosc(), overwrite=True)

    tracks_grp.attrs["tracks_metadata"] = {
        "format_version": "0.1",
        "type": "napari_tracks",
        "columns": ["track_id", "time", "y", "x"]
    }

def load_prism_file(file_path):
    tables = []

    with zipfile.ZipFile(file_path, "r") as prism_zip:
        file_list = prism_zip.namelist()
        print("Files in the Prism archive:", file_list)  # Debugging step

        # Find CSV and JSON table files
        csv_files = [f for f in file_list if f.endswith("data.csv")]
        json_files = [f for f in file_list if f.endswith("content.json")]

        # Try loading CSV files first
        for csv_file in csv_files:
            with prism_zip.open(csv_file) as f:
                df = pd.read_csv(io.StringIO(f.read().decode("utf-8")))
                tables.append((csv_file, df))

        # If no CSV files, fallback to JSON
        if not tables and json_files:
            for json_file in json_files:
                with prism_zip.open(json_file) as f:
                    json_data = json.load(f)
                    df = pd.DataFrame(json_data)  # Convert JSON data to DataFrame
                    tables.append((json_file, df))

    if not tables:
        raise ValueError("No valid data files (CSV or JSON) found in Prism file!")

    return tables


def get_folder_size(folder):
    """ByteSize Class
    ==============

    This class represents a byte size value and provides utility methods for
    formatting and manipulating byte sizes.

    Usage:
    ------
    1. Create a ByteSize object:
       bs = ByteSize(1024)  # Initialize with bytes (e.g., 1024 bytes)

    2. Access byte sizes in different units:
       bs.bytes          # Get size in bytes
       bs.kilobytes      # Get size in kilobytes
       bs.megabytes      # Get size in megabytes
       bs.gigabytes      # Get size in gigabytes
       bs.petabytes      # Get size in petabytes

    3. Get a human-readable representation of the byte size:
       str(bs)           # Get a formatted string (e.g., '1.00 KB')

    4. Perform arithmetic operations with ByteSize objects:
       addition, subtraction, and multiplication are supported.

    Example:
    -------
    bs1 = ByteSize(2048)
    bs2 = ByteSize(4096)

    # Perform arithmetic operations
    result = bs1 + bs2    # Addition
    result = bs2 - bs1    # Subtraction
    result = bs1 * 2      # Multiplication

    Attributes:
    ----------
    - bytes: Size in bytes.
    - kilobytes: Size in kilobytes.
    - megabytes: Size in megabytes.
    - gigabytes: Size in gigabytes.
    - petabytes: Size in petabytes.
    - readable: A tuple with the unit suffix and the corresponding value (e.g., ('KB', 2.0)).

    Methods:
    -------
    - __str__: Return a formatted string representation of the byte size.
    - __repr__: Return a string representation suitable for object inspection.
    - __format__: Format the byte size according to a specified format.
    - __add__, __sub__, __mul__: Perform arithmetic operations with ByteSize objects.
    - __radd__, __rsub__, __rmul__: Perform reverse arithmetic operations with ByteSize objects.

    """
    return ByteSize(sum(file.stat().st_size for file in Path(folder).rglob('*')))


class ByteSize(int):

    _KB = 1024
    _suffixes = 'B', 'KB', 'MB', 'GB', 'PB'

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        self.bytes = self.B = int(self)
        self.kilobytes = self.KB = self / self._KB**1
        self.megabytes = self.MB = self / self._KB**2
        self.gigabytes = self.GB = self / self._KB**3
        self.petabytes = self.PB = self / self._KB**4
        *suffixes, last = self._suffixes
        suffix = next((
            suffix
            for suffix in suffixes
            if 1 < getattr(self, suffix) < self._KB
        ), last)
        self.readable = suffix, getattr(self, suffix)

        super().__init__()

    def __str__(self):
        return self.__format__('.2f')

    def __repr__(self):
        return f'{self.__class__.__name__}({super().__repr__()})'

    def __format__(self, format_spec):
        suffix, val = self.readable
        return '{val:{fmt}} {suf}'.format(val=val, fmt=format_spec, suf=suffix)

    def __sub__(self, other):
        return self.__class__(super().__sub__(other))

    def __add__(self, other):
        return self.__class__(super().__add__(other))

    def __mul__(self, other):
        return self.__class__(super().__mul__(other))

    def __rsub__(self, other):
        return self.__class__(super().__sub__(other))

    def __radd__(self, other):
        return self.__class__(super().__add__(other))

    def __rmul__(self, other):
        return self.__class__(super().__rmul__(other))


def generate_url(row):
    """Generate a properly formatted local file address for the 'URL' column in Harmony metadata.
    This function replaces remote addresses, ensuring consistency when metadata is exported separately from the images.

    Parameters
    ----------
    row (pd.Series): A row of Harmony metadata containing 'Row', 'Col', 'FieldID', 'PlaneID', 'ChannelID', 'TimepointID', and 'FlimID' columns.

    Returns
    -------
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


def track_to_df(track):
    """Quick hack to return a single track as a dataframe for output into excel
    """
    return pd.DataFrame(track.to_dict(), columns=list(track.to_dict().keys()))


def read_harmony_metadata(metadata_path: os.PathLike, assay_layout=False,
                        mask_exist=False, image_dir=None, image_metadata=None,
                        replicate_number=True, iter=True
                        ) -> pd.DataFrame:
    """Read the metadata from the Harmony software for the Opera Phenix microscope.
    Takes an input of the path to the metadata .xml file.
    Returns the metadata in a pandas dataframe format.
    """
    # extraction procedure for image volume metadata
    metadata = []

    # Handle the iteration mode with iterparse (iter=True)
    if not assay_layout and iter:
        # Get the total size of the XML file for a finite progress bar
        file_size = os.path.getsize(metadata_path)
        
        # Open the file and wrap iterparse in tqdm using file size as the limit
        with open(metadata_path, 'rb') as f:
            pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Parsing Harmony Metadata")
            
            # Use 'end' events to clear elements from memory after processing
            for event, elem in ET_iter.iterparse(f, events=("end",)):
                # Update the progress bar based on the current file pointer position
                pbar.update(f.tell() - pbar.n)
                
                if event == "end" and "Images" in elem.tag:
                    for image_metadata_elem in elem:
                        # Professional dictionary comprehension for extraction
                        single_image_dict = {
                            item.tag.split('}')[-1]: item.text 
                            for item in image_metadata_elem
                        }
                        metadata.append(single_image_dict)

                    # CRITICAL: Clear processed element to free memory on the workstation
                    elem.clear()
            
            pbar.close()

    # Handle the non-iterative method (iter=False)
    elif not assay_layout and not iter:
        print('Reading metadata XML file (Non-iterative)...')
        try:
            tree = ET_iter.parse(metadata_path)
            root = tree.getroot()

            # Find the 'Images' tag with the specific Harmony namespace
            for images in root.iter('{http://www.perkinelmer.com/PEHH/HarmonyV5}Images'):
                for image_metadata_elem in images:
                    single_image_dict = {
                        item.tag.split('}')[-1]: item.text 
                        for item in image_metadata_elem
                    }
                    metadata.append(single_image_dict)

        except ET_iter.XMLSyntaxError as e:
            print(f"XML Syntax Error: {e}")
            raise
        except OSError as e:
            print(f"Error parsing file: {e}")
            raise

    # extraction procedure for assay layout metadata
    if assay_layout:
        print('Try the newer dataio.read_harmony_assaylayout function for added compatibility')
        with open(metadata_path, 'rb') as f:
            xml_data = f.read()
        root = ET_iter.XML(xml_data)
        metadata_dict = dict()
        for branch in root:
            for subbranch in branch:
                if subbranch.text.strip() and subbranch.text.strip() != 'string':
                    col_name = subbranch.text
                    metadata_dict[col_name] = dict()
                for subsubbranch in subbranch:
                    if 'Row' in subsubbranch.tag:
                        row = int(subsubbranch.text)
                    elif 'Col' in subsubbranch.tag and 'Color' not in subsubbranch.tag:
                        col = int(subsubbranch.text)
                    if 'Value' in subsubbranch.tag and subsubbranch.text is not None:
                        val = subsubbranch.text
                        metadata_dict[col_name][int(row), int(col)] = val
        metadata = metadata_dict

    # Create a dataframe out of all metadata
    df = pd.DataFrame(metadata)

    # Aesthetics and secondary processing for assay layout
    if assay_layout:
        df.index.set_names(['Row', 'Column'], inplace=True)
        if 'Cell Count' in df.columns:
            if pd.isna(df['Cell Count']).any():
                df.drop(columns='Cell Count', inplace=True)
        if 'double' in df.columns:
            df.rename(columns={'double': 'Cell Count'}, inplace=True)
        if replicate_number:
            df['Replicate #'] = df.groupby(['Strain', 'Compound', 'Concentration', 'ConcentrationEC']).cumcount() + 1

    print('Extracting metadata complete!')
    return df

def do_masks_exist(image_dir, metadata, row=None, col=None, print_output=True):
    """Iterates over all positions in experiment and checks if masks have been
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


def read_harmony_assaylayout(xml_path: str | Path, replicate_number: bool = False) -> pd.DataFrame:
    """
    Parse PerkinElmer/Revvity Harmony assay layout XML (V5 or V6) into a DataFrame.

    Parameters
    ----------
    xml_path : str or Path
        Path to the Harmony assay layout XML file.
    replicate_number : bool, optional
        If True, add a 'Replicate #' column when the columns 'Strain', 'Compound',
        and at least one of ['Concentration', 'ConcentrationEC'] exist.

    Returns
    -------
    pd.DataFrame
        Index = MultiIndex (Row, Column), columns = each Layer <Name>,
        values coerced according to <ValueType> where possible.
    """
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()

    layers = []
    for layer in root.findall(".//{*}Layer"):
        name_el = layer.find("./{*}Name")
        vtype_el = layer.find("./{*}ValueType")
        lname = (name_el.text or "").strip() if name_el is not None else f"Layer_{len(layers)+1}"
        vtype = (vtype_el.text or "").strip() if vtype_el is not None else None

        # V5 puts <Well> under <Wells>; V6 may put <Well> directly under <Layer>
        wells_parent = layer.find("./{*}Wells")
        if wells_parent is not None:
            well_nodes = wells_parent.findall("./{*}Well")
        else:
            well_nodes = layer.findall("./{*}Well")

        wells = []
        for w in well_nodes:
            r_el = w.find("./{*}Row")
            c_el = w.find("./{*}Col")
            val_el = w.find("./{*}Value")
            if r_el is None or c_el is None:
                continue
            r = int(r_el.text)
            c = int(c_el.text)
            v = _coerce(val_el.text if val_el is not None else None, vtype)
            wells.append((r, c, v))
        layers.append((lname, vtype, wells))

    # Collect coordinates and assemble table
    coords = sorted({(r, c) for _, _, ws in layers for (r, c, _) in ws})
    idx = pd.MultiIndex.from_tuples(coords, names=["Row", "Column"])
    rows = {coord: {} for coord in idx}
    for lname, _, ws in layers:
        for r, c, v in ws:
            rows[(r, c)][lname] = v

    df = pd.DataFrame([rows[k] for k in idx], index=idx).where(pd.notnull, None)

    # Optional replicate numbering
    if replicate_number:
        # Require Strain + Compound + (Concentration and/or ConcentrationEC)
        options = [
            ["Strain", "Compound", "Concentration", "ConcentrationEC"],
            ["Strain", "Compound", "Concentration"],
            ["Strain", "Compound", "ConcentrationEC"],
        ]
        have = set(df.columns)
        for group in options:
            if set(group).issubset(have):
                df = df.copy()
                df["Replicate #"] = df.groupby(group, dropna=False).cumcount() + 1
                break

    # drop nananaaa values from df
    df = df.dropna()

    return df


def _strip_ns(tag: str) -> str:
    return tag.split('}', 1)[-1] if '}' in tag else tag


def _coerce(val_text: str | None, value_type: str | None):
    if val_text is None:
        return None
    s = val_text.strip()
    if s == "":
        return None
    vt = (value_type or "").strip().lower()
    if vt in {"double", "float"}:
        try:
            return float(s)
        except ValueError:
            return s
    if vt in {"int", "integer"}:
        try:
            return int(s)
        except ValueError:
            try:
                return int(float(s))
            except ValueError:
                return s
    if vt in {"bool", "boolean"}:
        return s.lower() in {"true", "1", "yes"}
    return s
