import os
import shutil
import warnings
from glob import glob
from typing import Union, List

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
from osgeo import gdal, osr, ogr
from rasterio import features
from rasterio.windows import Window
from shapely.geometry import polygon
from torchvision.transforms import ToTensor
from tqdm.auto import tqdm
from PIL import Image
import mgrs
from geopy import Point
from geopy.distance import distance
import pyproj
from pyproj import CRS

from src.utilities.coords import tiff_to_bbox, bridge_in_bbox
from file_types import OpticalComposite, MultiVariateComposite, Tile, PyTorch, Sentinel2Tile, OpticalCompositeSlice, SingleRegionTileMatch

BANDS_TO_IX = {
    'B02': 3,  # Blue
    'B03': 2,  # Green
    'B04': 1,  # Red,
    'B08': 4  # IR
}

MAX_RGB_VAL = 4000  # counts
MAX_IR_VAL = 4000  # counts (double check this by looking at composites)
MAX_ELEV = 8000  # meters
MIN_ELEV = 0  # meters


# the max_pixel_val was set to 2500? this probably should match with the max allowed value for clouds or it shouldn't be
# included...


def scale(
        x: Union[float, np.array],
        max_rgb: float = MAX_RGB_VAL,
        max_ir: float = MAX_IR_VAL,
        min_elev: float = MIN_ELEV,
        max_elev: float = MAX_ELEV
) -> Union[float, np.array]:
    if type(x) == float:
        return x / max_rgb
    else:
        if x.shape[2] == 3:
            return x / max_rgb
        else:  # multivariate case
            assert x.shape[2] == 8  # B04, B03, B02 (RGB), B08(IR), OSM Water, OSM Boundary, Elevation, Slope
            normalized_rgb = np.clip(x[:, :, 0:3] / max_rgb, 0, 1)
            normalized_ir = np.clip(x[:, :, 3] / max_ir, 0, 1)
            normalized_osm = x[:, :, 4:6]
            assert np.all(normalized_osm <= 1) and np.all(normalized_osm >= 0), \
                'OSM water and boundary should be binary images (only 0s and 1s)'
            normalized_elevation = np.clip((x[:, :, 6] - min_elev) / (max_elev - min_elev), 0, 1)  # in meter
            normalized_slope = np.clip(x[:, :, 7] / 90, 0, 1)  # in deg
        return np.concatenate([
            normalized_rgb,
            normalized_ir.reshape((*normalized_ir.shape, 1)),
            normalized_osm,
            normalized_elevation.reshape((*normalized_elevation.shape, 1)),
            normalized_slope.reshape((*normalized_slope.shape, 1))],
            axis=2)


def get_utm_epsg(lat, lon):
    utm_zone = int((lon + 180) // 6) + 1
    epsg_code = 32600 if lat >= 0 else 32700
    epsg_code += utm_zone
    return epsg_code


def get_img_from_file(img_path, g_ncols, dtype, row_bound=None):
    img = rasterio.open(img_path, driver='JP2OpenJPEG')
    ncols, nrows = img.meta['width'], img.meta['height']
    assert g_ncols == ncols, f'Imgs have different size ncols: {ncols} neq {g_ncols}'
    if row_bound is None:
        pixels = img.read(1).astype(np.float32)
    else:
        pixels = img.read(
            1,
            window=Window.from_slices(
                slice(row_bound[0], row_bound[1]),
                slice(0, ncols)
            )
        ).astype(dtype)
    return pixels


def get_cloud_mask_from_file(cloud_path, crs, transform, shape, row_bound=None):
    # filter out RuntimeWarnings, due to geopandas/fiona read file spam
    # https://stackoverflow.com/questions/64995369/geopandas-warning-on-read-file
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    try:
        cloud_file = gpd.read_file(cloud_path)
        cloud_file.crs = (str(crs))
        # convert the cloud mask data to a raster that has the same shape and transformation as the
        # img raster data
        cloud_img = features.rasterize(
            (
                (g['geometry'], 1) for v, g in cloud_file.iterrows()
            ),
            out_shape=shape,
            transform=transform,
            all_touched=True
        )
        if row_bound is None:
            return np.where(cloud_img == 0, 1, 0)
        return np.where(cloud_img[row_bound[0]:row_bound[1], :] == 0, 1, 0)
    except Exception as e:
        return None


def nan_clouds(pixels, cloud_channels, max_pixel_val: float = MAX_RGB_VAL):
    cp = pixels * cloud_channels
    mask = np.where(np.logical_or(cp == 0, cp > max_pixel_val))
    cp[mask] = np.nan
    return cp


def create_optical_composite_from_s2(region: str, district: str, coord: str, bands: list, dtype: type,
                                     num_slices: int = 12, pbar: bool = True) -> str:
    """
    Creates a cloud cleaned optical composite from a set of sentinel 2 files. Can combine multiple optical bands
    Args:
        region (str): The region for which to create the composite for
        district (str): The district within the region for which to create the composite for
        coord (str): The military grid coordinate (35MGR, 36MTV, etc.) within the district to create the composite for 
        bands (list): The optical bands for which to create the composite for (B02, B03, B08) etc.
        dtype (type): The data type to save the band data as
        num_slices (int): The amount of slices to split the composite up into while building it. More slices will use less RAM
        pbar (bool): If true then a progress bar will be shown
    Returns:
        optical_composite_file.archive_path (str): Path to the output optical composite 
    """
    print('Creating composite for', coord)
    optical_composite_file = OpticalComposite(region, district, coord, bands)
    if optical_composite_file.exists:
        return optical_composite_file.archive_path
    optical_composite_file.create_archive_dir()

    # Loop through each band, getting a median estimate for each
    crs = None
    transform = None
    optical_composite_band_files: List[OpticalComposite] = []
    for band in tqdm(bands, desc=f'Processing {coord}', leave=True, position=1, total=len(bands), disable=pbar):
        band_files = Sentinel2Tile.find_files(region=region, district=district, band=band, mgrs=coord)
        if not band_files:
            raise LookupError(f'No sentinel2 files found for region: {region} district: {district} band: {band} '
                              f'mgrs: {mgrs}')

        crs = None
        transform = None
        g_nrows = None
        g_ncols = None
        for file in band_files:
            with rasterio.open(file, 'r', driver='JP2OpenJPEG') as rf:
                g_nrows = rf.meta['width'] if g_nrows is None else g_nrows
                g_ncols = rf.meta['height'] if g_ncols is None else g_ncols
                crs = rf.crs if crs is None else crs
                transform = rf.transform if transform is None else transform
            if crs is not None and transform is not None and g_nrows is not None and g_ncols is not None:
                break
        
        if crs is None or transform is None or g_nrows is None or g_ncols is None:
            raise LookupError(f'Could not determine the following projection attributes from the available '
                              f'sentinel2 files in {os.path.dirname(os.path.dirname(file))}: \n' \
                              f'{"CRS" if crs is None else ""} '
                              f'{"Transform" if transform is None else ""} '
                              f'{"Number of rows" if g_nrows is None else ""} '
                              f'{"Number of columns" if g_ncols is None else ""}')

        slice_width = g_nrows / num_slices
        slice_end_pts = [int(i) for i in np.arange(0, g_nrows + slice_width, slice_width)]
        slice_bounds = [(slice_end_pts[i], slice_end_pts[i + 1] - 1) for i in range(num_slices - 1)]
        slice_bounds.append((slice_end_pts[-2], slice_end_pts[-1]))   

        single_band_composite = OpticalComposite(region, district, coord, [band])
        if single_band_composite.exists:
            optical_composite_band_files.append(single_band_composite)
            continue
        single_band_composite.create_archive_dir()

        # Median across time, slicing if necessary
        slice_files: List[OpticalCompositeSlice] = []
        for k, row_bound in tqdm(enumerate(slice_bounds), desc=f'band={band}', total=num_slices, position=2,
                                 disable=pbar):
            slice_file = OpticalCompositeSlice(region, district, coord, band, row_bound[0], row_bound[1])
            slice_file.create_archive_dir()
            if slice_file.exists:
                slice_files.append(slice_file)
                continue

            cloud_correct_imgs = []
            for img_path in tqdm(band_files, desc=f'slice {k + 1}', leave=False, position=3, disable=pbar):
                # Get data from files
                pixels = get_img_from_file(img_path, g_ncols, dtype, row_bound)
                date_dir = os.path.dirname(img_path)
                cloud_files = glob(os.path.join(date_dir, "*.gml"))
                assert len(cloud_files) == 1, f'Cloud path does not exist for {date_dir}'

                cloud_channels = get_cloud_mask_from_file(cloud_files[0], crs, transform, (g_nrows, g_ncols), row_bound)
                if cloud_channels is None:
                    continue
                # add to list to do median filter later
                cloud_correct_imgs.append(nan_clouds(pixels, cloud_channels))
                del pixels
            corrected_stack = np.vstack([img.ravel() for img in cloud_correct_imgs])
            median_corrected = np.nanmedian(corrected_stack, axis=0, overwrite_input=True)
            median_corrected = median_corrected.reshape(cloud_correct_imgs[0].shape)

            with rasterio.open(slice_file.archive_path, 'w', driver='Gtiff', width=g_ncols, height=g_nrows, count=1,
                               crs=crs, transform=transform, dtype=dtype) as wf:
                wf.write(median_corrected.astype(dtype), 1)
                slice_files.append(slice_file)
            # release mem
            median_corrected = []
            del median_corrected
            corrected_stack = []
            del corrected_stack

        # Combine slices
        with rasterio.open(single_band_composite.archive_path, 'w', driver='GTiff', width=g_ncols, height=g_nrows,
                           count=1, crs=crs, transform=transform, dtype=dtype) as wf:
            for slice_file in slice_files:
                with rasterio.open(slice_file.archive_path, 'r', driver='GTiff') as rf:
                    wf.write(
                        rf.read(1),
                        window=Window.from_slices(
                            slice(slice_file.left_bound, slice_file.right_bound),
                            slice(0, g_ncols)
                        ),
                        indexes=1
                    )
                os.remove(slice_file.archive_path)

    # Combine Bands
    n_bands = len(optical_composite_band_files)
    with rasterio.open(
            optical_composite_file.archive_path,
            'w',
            driver='GTiff',
            width=g_ncols,
            height=g_nrows,
            count=n_bands,
            crs=crs,
            transform=transform,
            dtype=dtype
    ) as wf:
        for band_file in tqdm(optical_composite_band_files, total=n_bands, desc='Combining bands...', leave=False,
                              position=1, disable=pbar):
            j = BANDS_TO_IX[band_file.bands[0]]
            with rasterio.open(band_file.archive_path, 'r', driver='GTiff') as rf:
                wf.write(rf.read(1), indexes=j)
            os.remove(band_file.archive_path)

    shutil.rmtree(os.path.dirname(slice_file.archive_path))

    return optical_composite_file.archive_path


def scale_multiband_composite(multiband_tiff: str):
    assert os.path.isfile(multiband_tiff)
    scaled_tiff = multiband_tiff.split('.')[0] + '_scaled.tiff'

    with gdal.Open(multiband_tiff, 'r') as rf:
        with rasterio.open(
                str(scaled_tiff),
                'w',
                driver='Gtiff',
                width=rf.width, height=rf.height,
                count=3,
                crs=rf.crs,
                transform=rf.transform,
                dtype='uint8'
        ) as wf:
            wf.write((scale(rf.read(0)) * 255).astype('uint8'), 0)
            wf.write((scale(rf.read(1)) * 255).astype('uint8'), 1)
            wf.write((scale(rf.read(2)) * 255).astype('uint8'), 2)
    return scaled_tiff


def composite_to_tiles(composite: MultiVariateComposite, bridge_locations: Union[None, gpd.array.GeometryArray], tqdm_pos: int = None, tqdm_update_rate: float = None, remove_tiff: bool = True, tile_size: int = 300) -> pd.DataFrame:
    """
    Splits up the input composite file into small chunks called tiles. The size of the tiles in meters is controlled by the tile_size input parameter.
    If there is ground truth data for the input composite, each tile will be classified as either "bridge" or "no_bridge". A dataframe with this information as well 
    as the geographic and file location of each tile will be output. Each tile will be saved as a PyTorch pt file which contains its normalized features as a tensor.
    These pt files will be what is used to train models / run inference over.
    Args:
        composite (MultiVariateComposite): File object of the composite to split up into tiles
        bridge_locations (gpd.array.GeometryArray): Contains the location of each bridge that there is ground truth data for
        tqdm_pos (int): Progress pointer
        tqdm_update_rate (float): Progress update rate
        remove_tiff (bool): Defaults to true. If true then tiff files used to determine tile bounding boxes will be removed. These are only useful for visually inspecting tiles for validation
        tile_size (int): Size that each tile should be in meters. For example, if tile_size is 300 then each tile will be 300x300 meters
    Returns:
        df (pd.DataFrame): Dataframe with file and geographic location of each tile, its bounding box, and if there was ground truth data, whether the tile contains a bridge or not
    """
    grid_geoloc_file = SingleRegionTileMatch(tile_size)

    grid_geoloc_path = grid_geoloc_file.archive_path(composite.region, composite.district, composite.mgrs, create_dir=True)
    if os.path.exists(grid_geoloc_path):
        df = pd.read_csv(grid_geoloc_path)
        return df

    rf = gdal.Open(composite.archive_path())
    _, xres, _, _, _, yres = rf.GetGeoTransform()
    nxpix = int(tile_size / abs(xres))
    nypix = int(tile_size / abs(yres))
    xsteps = np.arange(0, rf.RasterXSize, nxpix).astype(np.int64).tolist()
    ysteps = np.arange(0, rf.RasterYSize, nypix).astype(np.int64).tolist()

    if bridge_locations is not None:
        bbox = tiff_to_bbox(composite.archive_path())
        this_bridge_locs = []
        p = polygon.Polygon(bbox)
        for loc in bridge_locations:
            if p.contains(loc):
                this_bridge_locs.append(loc)

    numTiles = len(xsteps) * len(ysteps)
    torch_transformer = ToTensor()
    df = pd.DataFrame(
        columns=['tile', 'bbox', 'is_bridge', 'bridge_loc'] if bridge_locations is not None else ['tile', 'bbox'],
        index=range(numTiles)
    )
    if tqdm_update_rate is None:
        tqdm_update_rate = int(round(numTiles / 100))
    else:
        assert type(tqdm_update_rate) == int

    with tqdm(
            position=tqdm_pos,
            total=numTiles,
            desc=composite.mgrs,
            miniters=tqdm_update_rate,
            disable=(tqdm_pos is None)
    ) as pbar:
        k = 0
        for xmin in xsteps:
            for ymin in ysteps:
                tile_tiff = Tile(x_min=xmin, y_min=ymin)
                tile_tiff_path = tile_tiff.archive_path(composite.region, composite.district, composite.mgrs, tile_size=tile_size, create_dir=True)
                pt_file = PyTorch(x_min=xmin, y_min=ymin)
                pt_file_path = pt_file.archive_path(composite.region, composite.district, composite.mgrs, tile_size=tile_size)
                if not os.path.isfile(tile_tiff_path):
                    gdal.Translate(
                        tile_tiff_path,
                        rf,
                        srcWin=(xmin, ymin, nxpix, nypix),
                    )
                bbox = tiff_to_bbox(tile_tiff_path)
                df.at[k, 'tile'] = pt_file_path
                df.at[k, 'bbox'] = bbox
                if bridge_locations is not None:
                    df.at[k, 'is_bridge'], df.at[k, 'bridge_loc'], ix = bridge_in_bbox(bbox, this_bridge_locs)
                    if ix is not None:
                        this_bridge_locs.pop(ix)
                if not os.path.exists(pt_file_path):
                    with rasterio.open(tile_tiff_path, 'r') as tmp:
                        scale_img = tmp.read()
                        scale_img = np.moveaxis(scale_img, 0, -1)  # make dims be c, w, h
                        scale_img = scale(scale_img)
                        tensor = torch_transformer(scale_img)
                        torch.save(tensor, pt_file_path)
                if remove_tiff:
                    os.remove(tile_tiff_path)
                k += 1
                if k % tqdm_update_rate == 0:
                    pbar.update(tqdm_update_rate)
                    pbar.refresh()
                if k % int(round(numTiles / 4)) == 0 and k < numTiles - 1:
                    percent = int(round(k / int(round(numTiles)) * 100))
                    pbar.set_description(f'Saving {composite.mgrs} {percent}%')
                    df.to_csv(grid_geoloc_path, index=False)
        pbar.set_description(f'Saving to file {grid_geoloc_file}')
    df.to_csv(grid_geoloc_path, index=False)
    return df


def transform_point(src_wkt, dst_wkt, x, y):
    src_crs = CRS.from_wkt(src_wkt)
    dst_crs = CRS.from_wkt(dst_wkt)
    print(x, y)
    # Convert source UTM coordinates to latitude and longitude
    transformer_latlon = pyproj.Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = transformer_latlon.transform(x, y)
    print(lon, lat, 'latlontransform')
    # Convert latitude and longitude to destination UTM zone
    transformer_dest = pyproj.Transformer.from_crs(CRS.from_epsg(4326), dst_crs, always_xy=True)
    transformed_point = transformer_dest.transform(lon, lat)
    return transformed_point


def subsample_geo_tiff(low_resolution_path: str, high_resolution_path: str):
    low_res = gdal.Open(low_resolution_path)
    high_res = gdal.Open(high_resolution_path)

    print(low_resolution_path)
    # Access the data
    low_res_band = low_res.GetRasterBand(1)
    low_res_data = low_res_band.ReadAsArray()

    low_res_geo_transform = low_res.GetGeoTransform()
    low_res_projection = low_res.GetProjection()

    high_res_geo_transform = high_res.GetGeoTransform()
    high_res_projection = high_res.GetProjection()

    if low_res_projection != high_res_projection:

        dst_point = transform_point(low_res_projection, high_res_projection, low_res_geo_transform[0],
                                    low_res_geo_transform[3])
        low_res_geo_transform = [dst_point[0], low_res_geo_transform[1], 0, dst_point[1], 0, low_res_geo_transform[5]]

    low_res_lons, low_res_lats = get_geo_locations_from_tif(low_res_geo_transform, low_res.RasterXSize,
                                                            low_res.RasterYSize)

    def lookup_nearest_lon(lon: int):
        yi = np.abs(np.array(low_res_lons) - lon).argmin()
        return yi

    def lookup_nearest_lat(lat: int):
        yi = np.abs(np.array(low_res_lats) - lat).argmin()
        return yi

    high_res_lons, high_res_lats = get_geo_locations_from_tif(high_res_geo_transform, high_res.RasterXSize,
                                                              high_res.RasterYSize)

    high_res_data = np.zeros((len(high_res_lats), len(high_res_lons)))

    closest_lats = [lookup_nearest_lat(lat) for lat in high_res_lats]
    closest_lons = [lookup_nearest_lon(lon) for lon in high_res_lons]

    for i, y in enumerate(closest_lons):
        for j, x in enumerate(closest_lats):
            high_res_data[j, i] = low_res_data[x, y]

    return high_res_data


def get_geo_locations_from_tif(geo_transform: List[float], x_size: int, y_size: int):
    # Get geolocation information
    dx = geo_transform[1]
    dy = geo_transform[5]
    x_origin = geo_transform[0]
    y_origin = geo_transform[3]

    print(dx, dy, x_origin, y_origin, 'geotransform', x_size, y_size)

    # Get geolocation of each data point
    lats = []
    for row in range(y_size):
        lats.append(y_origin + (row * dy))

    lons = []
    for col in range(x_size):
        lons.append(x_origin + (col * dx))

    return lons, lats


def _create_raster(output_path: str, columns: int, rows: int, n_band: int = 1, gdal_data_type: int = gdal.GDT_UInt16,
                   driver: str = r'GTiff'):
    """
    Credit:
    https://gis.stackexchange.com/questions/290776/how-to-create-a-tiff-file-using-gdal-from-a-numpy-array-and-
    specifying-nodata-va

    Creates a blank raster for data to be written to
    Args:
        output_path (str): Path where the output tif file will be written to
        columns (int): Number of columns in raster
        rows (int): Number of rows in raster
        n_band (int): Number of bands in raster
        gdal_data_type (int): Data type for data written to raster
        driver (str): Driver for conversion
    """
    # create driver
    driver = gdal.GetDriverByName(driver)

    output_raster = driver.Create(output_path, columns, rows, n_band, eType=gdal_data_type)
    return output_raster


def elevation_to_slope(elevation_file: str, slope_outfile: str):
    image = Image.open(elevation_file)
    elevation_data = np.array(image)  # Measured in meters
    dx, dy = 30.87, 30.87  # 1 arc second in meters
    x_slope, y_slope = np.gradient(elevation_data, dx, dy)
    slope = np.sqrt(x_slope ** 2 + y_slope ** 2)

    # Calculate in degrees
    slope_deg = np.rad2deg(np.arctan(slope))

    dataset = gdal.Open(elevation_file)
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    numpy_array_to_raster(slope_outfile, slope_deg, geo_transform, projection)


def numpy_array_to_raster(output_path: str, numpy_array: np.array, geo_transform,
                          projection: str, n_band: int = 1, no_data: int = -1,
                          gdal_data_type: int = gdal.GDT_UInt16):
    """
    Returns a gdal raster data source
    Args:
        output_path (str): Full path to the raster to be written to disk
        numpy_array (np.array): Numpy array containing data to write to raster
        geo_transform (gdal GeoTransform): tuple of six values that represent the top left corner coordinates, the
        pixel size in x and y directions, and the rotation of the image
        n_band (int): The band to write to in the output raster
        no_data (int): Value in numpy array that should be treated as no data
        gdal_data_type (int): Gdal data type of raster (see gdal documentation for list of values)
    """
    rows, columns = numpy_array.shape

    # create output raster
    output_raster = _create_raster(output_path, int(columns), int(rows), n_band, gdal_data_type)
    output_raster.SetProjection(projection)
    output_raster.SetGeoTransform(geo_transform)
    output_band = output_raster.GetRasterBand(1)
    output_band.SetNoDataValue(no_data)
    output_band.WriteArray(numpy_array)
    output_band.FlushCache()
    output_band.ComputeStatistics(False)

    if not os.path.exists(output_path):
        raise Exception('Failed to create raster: %s' % output_path)

    return output_raster
