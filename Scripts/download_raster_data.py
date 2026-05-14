import ee
import geopandas
import pickle
from tqdm import tqdm
import requests
import os
import glob
from dotenv import load_dotenv

import rasterio as r
from rasterio.errors import RasterioIOError

import tifffile

import time
import numpy as np

import subprocess

import argparse

##############################################
# Aux. functions
t = 0

def tic():
    global t
    t = time.time()

def toc():
    global t
    now = time.time()
    delta_t = now-t
    t = now
    return delta_t

def translate_image(file:str):
    original_file = file.replace('.tif', '_.tif')
    os.rename(file, original_file)
    command = f'gdal_translate -ot Float32 -co COMPRESS=DEFLATE -co NBITS=16 "{original_file}" "{file}"'
    subprocess.call(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(original_file)

##############################################
# Workflow functions
def create_folders(folder_output:str, image_ids:list[str]):
    for image_id in image_ids:
        os.makedirs(os.path.join(folder_output, image_id), exist_ok=True)

def get_chips(folder_input:str):
    if os.path.exists(folder_input+'/Samples.gpkg'):
        file_path = folder_input+'/Samples.gpkg'
    elif os.path.exists(folder_input+'/samples.gpkg'):
        file_path = folder_input+'/samples.gpkg'
    else:
        files = glob.glob(folder_input+'/*.gpkg')
        if len(files)>0:
            file_path = files[0]
        else:
            return None
    
    chips = geopandas.read_file(file_path)
    chips_reproj = chips.to_crs('EPSG:4326')
    return chips_reproj

def get_metadata(folder_input:str):
    with open(f'{folder_input}/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return metadata

##############################################
# EE functions
def addDateBandS1(image):
        days = ee.Date(image.get('system:time_start')).difference(ee.Date('2000-01-01'), 'days')
        return image.addBands(ee.Image(days).rename('pixel_date').mask(image.select('VV').mask()).toUint16())

def addRatio(image):
        return (image.addBands(image.select('VV')
                    .divide(image.select('VH'))
                    .rename('ratio')))

def addDateBandS2(image):
        days = ee.Date(image.get('system:time_start')).difference(ee.Date('2000-01-01'), 'days')
        return image.addBands(ee.Image(days).rename('pixel_date').mask(image.select('B2').mask()).toUint16())

# # Make a clear median composite.
# def updatemaks(image, QA_BAND, CLEAR_THRESHOLD):
#     return image.updateMask(image.select(QA_BAND).gte(CLEAR_THRESHOLD))

def get_s1_mosaics(aoi:ee.Geometry.Polygon, date1:str, date2:str, date3:str, date4:str):
    '''
    return Sentinel-1 mosaics of images for the flooded and before flood periods. 

    ---
    aoi: ee.Geometry.Polygon
        EE geometry object to consider as AOI.
    date1: str
        Date of the start of flooded period. YYYY-MM-DD.
    date2: str
        Date of the end of the flooded period. YYYY-MM-DD.
    date3: str
        Date of the start of the before flood period. YYYY-MM-DD.
    date4: str
        Date of the end of the before flood period. YYYY-MM-DD.
    
    returns:
        s1_flooded: ee.Image
            Sentinel-1 mosaic of images for the flooded period.
        s2_before_flood: ee.Image
            Sentinel-1 mosaic of images for the before flood period.
    '''
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi)
                                                 .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                                                 .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                                                 .filter(ee.Filter.eq('orbitProperties_pass', orbit_pass))
                                                 .select(['VV', 'VH']))
    s1_flooded = s1.filterDate(date1, date2).sort('system:time_start', False).map(addRatio).map(addDateBandS1).mosaic()
    s1_before_flood = (s1.filterDate(date3, date4)).sort('system:time_start', False).map(addRatio).map(addDateBandS1).mosaic()
    return s1_flooded, s1_before_flood

def get_s2_mosaics(aoi:ee.Geometry.Polygon, date1:str, date2:str, date3:str, date4:str):
    '''
    return Sentinel-2 mosaics of images for the flooded and before flood periods. 

    ---
    aoi: ee.Geometry.Polygon
        EE geometry object to consider as AOI.
    date1: str
        Date of the start of flooded period. YYYY-MM-DD.
    date2: str
        Date of the end of the flooded period. YYYY-MM-DD.
    date3: str
        Date of the start of the before flood period. YYYY-MM-DD.
    date4: str
        Date of the end of the before flood period. YYYY-MM-DD.
    
    returns:
        s2_flooded: ee.Image
            Sentinel-2 mosaic of images for the flooded period.
        s2_before_flood: ee.Image
            Sentinel-2 mosaic of images for the before flood period.
    '''
    # Sentinel-2 Data
    # Harmonized Sentinel-2 Level 2A collection.
    s2 = ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterBounds(aoi)

    # Use 'cs' or 'cs_cdf', depending on your use case; see docs for guidance.
    QA_BAND = 'cs_cdf'

    # Cloud Score+ image collection. Note Cloud Score+ is produced from Sentinel-2
    # Level 1C data and can be applied to either L1C or L2A collections.
    def scale_cloud_mask(image):
        return image.select(QA_BAND).multiply(ee.Image(1000))
    csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED').filterBounds(aoi).map(scale_cloud_mask)

    # The threshold for masking; values between 0.50 and 0.65 generally work well.
    # Higher values will remove thin clouds, haze & cirrus shadows.
    CLEAR_THRESHOLD = 600

    s2_flooded = (s2.filterDate(date1, date2)
                    .linkCollection(csPlus, [QA_BAND])
                    .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12', QA_BAND])
                    .sort('system:time_start', False)
                    .map(addDateBandS2)
                    .mosaic()
                    .toUint16())

    s2_before_flood = (s2.filterDate(date3, date4)
                        .linkCollection(csPlus, [QA_BAND])
                        .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12', QA_BAND])
                        .sort('CLOUDY_PIXEL_PERCENTAGE', False)
                        .map(addDateBandS2)
                        .mosaic()
                        .toUint16())
    
    return s2_flooded, s2_before_flood


def create_flood_mask(s1_flooded, s1_before_flood, threshold=1.25, smoothing_radius=50):
    ########
    # Automatic simple flood mask

    # filtering images
    s1_flooded_filtered = s1_flooded.select('VH').focal_mean(smoothing_radius, 'circle', 'meters')
    s1_before_flood_filtered = s1_before_flood.select('VH').focal_mean(smoothing_radius, 'circle', 'meters')

    # calculate difference between images
    difference = s1_flooded_filtered.divide(s1_before_flood_filtered)

    # thresholding
    difference_binary = difference.gt(threshold)

    # refining result
    # water mask
    swater = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('seasonality')
    swater_mask = swater.gte(10).updateMask(swater.gte(10))

    # mask permanent water bodies
    flooded_mask = difference_binary.where(swater_mask,2)
    flooded = flooded_mask.updateMask(flooded_mask)
        
    # filter badly connected pixels
    connections = flooded.connectedPixelCount()
    flooded = flooded.updateMask(connections.gte(8))

    return flooded

def set_nodata_value(file_path, nodata_val):
    command = f'gdal_edit.py -a_nodata {nodata_val} {file_path}'
    subprocess.call(command, shell=True, stdout=None)

def download_chip(folder_output, aoi_geometry,
                  s1_before_flood, s1_flooded, s2_before_flood, s2_flooded,
                  flooded, dem, lulc, gsw, ind, image_id_to_download:list[str]=None):
    dict_time = {}
    region = ee.Geometry.BBox(aoi_geometry.bounds[0],
                              aoi_geometry.bounds[1],
                              aoi_geometry.bounds[2],
                              aoi_geometry.bounds[3])
    valid_list = ['s1_before_flood', 's1_during_flood', 's2_before_flood', 's2_during_flood', 'flood_mask', 'terrain', 'LULC', 'global_surface_water']
    if image_id_to_download is None:
        image_id_to_download = valid_list
    else:
        for image_id in image_id_to_download:
            if not image_id in valid_list:
                raise TypeError(f"'{image_id}' is not a valid image_id. Valid values are {valid_list}.")

    tic()
    images = {'s1_before_flood': s1_before_flood,
              's1_during_flood': s1_flooded,
              's2_before_flood': s2_before_flood,
              's2_during_flood': s2_flooded,
              'flood_mask': flooded,
              'terrain': dem,
              'LULC': lulc,
              'global_surface_water': gsw}

    for image_id in image_id_to_download:
        filename = f'{folder_output}/{image_id}/{ind:06d}_{image_id}.tif'
        if os.path.exists(filename):
            try:
                # dataset = r.open(filename)
                data = tifffile.imread(filename)
                data = r.open(filename).read()
                continue
            except KeyboardInterrupt:
                print('Manually interrupted!')
                exit()
            except:
                os.remove(filename)
                print(filename)
        url = images[image_id].getDownloadURL(
            {'region': region,
            'dimensions': '512x512',
            'crs': epsg,
            'format': 'GEO_TIFF'
            }
        )
        data_ = None
        count = 0
        while data_ is None:
            try:
                with open(filename, 'wb') as f:
                    response = requests.get(url, stream=True)
                    if not response.ok:
                        print(response)
                    for block in response.iter_content(1024):
                        if not block:
                            break
                        f.write(block)
                        
                dict_time[image_id]=toc()

                # opens the dile to check if it is ok
                # remove no data tags because they're causing trouble later
                with r.open(filename, 'r+') as data_:
                    data_.nodata = None
            except:
                os.remove(filename)
                count += 1
                if count>=10:
                    raise IOError(f'Some error is happening during download, chip id {ind} {image_id} was tried 10 times already.')
                time.sleep(5)
        
        # change bits for SAR data
        if image_id in ['s1_before_flood', 's1_during_flood', 'terrain']:
            translate_image(filename)
        if image_id == 'global_surface_water':
            set_nodata_value(filename, -128)

    return dict_time

##############################################
# MAIN
if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog='ChipsDownloader',
                    description='Download the chips of a flood event previously selected.',
                    epilog='bruno')
    parser.add_argument('-i', '--folder_input', required=True, type=str, help='Path to the folder containing the GPKG file and the metadata information.')
    parser.add_argument('-o', '--folder_output', type=str, help='Path to the folder where the samples are to be saved.')
    parser.add_argument('-id', '--image-ids', default=['s1_before_flood', 's1_during_flood', 's2_before_flood', 's2_during_flood', 'flood_mask', 'terrain', 'LULC'], 
                        nargs='+', help="List of image-id to donwload. Valid values are 's1_before_flood', 's1_during_flood', 's2_before_flood', 's2_during_flood', 'flood_mask', 'terrain', 'LULC', and 'global_surface_water'.")
    parser.add_argument('-m', '--max-samples', default=None, type=int, help='Maximum number of samples to download.')
    args = parser.parse_args()

    # check if image ids were inserted correctly
    valid_list = ['s1_before_flood', 's1_during_flood', 's2_before_flood', 's2_during_flood', 'flood_mask', 'terrain', 'LULC', 'global_surface_water']
    if args.image_ids is None:
        image_id_to_download = valid_list
    else:
        for image_id in args.image_ids:
            if not image_id in valid_list:
                raise TypeError(f"'{image_id}' is not a valid image_id. Valid values are {valid_list}.")

    folder_input = args.folder_input
    folder_output = args.folder_output
    if folder_output is None:
        folder_output=folder_input

    load_dotenv()

    ee.Initialize(project=os.getenv('EE_PROJECT'))

    create_folders(folder_output=folder_output, image_ids=args.image_ids)

    chips_reproj = get_chips(folder_input=folder_input)

    if not chips_reproj is None:
        if len(chips_reproj)>0:

            metadata = get_metadata(folder_input=folder_input)

            aoi = ee.Geometry.Polygon([[[chips_reproj.bounds['minx'].min(), chips_reproj.bounds['miny'].min()],
                                        [chips_reproj.bounds['maxx'].max(), chips_reproj.bounds['miny'].min()],
                                        [chips_reproj.bounds['maxx'].max(), chips_reproj.bounds['maxy'].max()],
                                        [chips_reproj.bounds['minx'].min(), chips_reproj.bounds['maxy'].max()]]])
            
            date1 = metadata['date_during_flood_start_S1']
            date2 = metadata['date_during_flood_end_S1']
            date3 = metadata['date_before_flood_start_S1']
            date4 = metadata['date_before_flood_end_S1']
            orbit_pass = metadata['orbit_pass']
            
            s1_flooded, s1_before_flood = get_s1_mosaics(aoi=aoi, date1=date1, date2=date2, date3=date3, date4=date4)
            s2_flooded, s2_before_flood = get_s2_mosaics(aoi=aoi, date1=date1, date2=date2, date3=date3, date4=date4)
            
            flooded = create_flood_mask(s1_flooded=s1_flooded, s1_before_flood=s1_before_flood, threshold=metadata['threshold'])
            
            # Copernicus DEM
            proj = ee.ImageCollection("COPERNICUS/DEM/GLO30").filterBounds(aoi).first().projection()
            dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").filterBounds(aoi).select('DEM').mosaic().setDefaultProjection(proj)
            dem = dem.addBands(ee.Terrain.slope(dem).rename('slope'))

            # Mask out areas with more than 5 percent slope using a Digital Elevation Model
            flooded = flooded.updateMask(dem.select('slope').lt(5))

            # ESA World Cover
            lulc = ee.ImageCollection("ESA/WorldCover/v200").first()

            # global surface water
            gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select(['occurrence', 'seasonality', 'recurrence'])

            epsg = 'EPSG:3857'

            dict_time = {'s1_before_flood': [],
                         's1_during_flood': [],
                         's2_before_flood': [],
                         's2_during_flood': [],
                         'flood_mask': [],
                         'terrain': [],
                         'LULC': [],
                         'global_surface_water': []}
            
            # shuffle samples
            chips_reproj = chips_reproj.sample(frac=1, random_state=len(chips_reproj))
            if not args.max_samples is None:
                if args.max_samples>0 and args.max_samples<len(chips_reproj):
                    chips_reproj = chips_reproj.head(args.max_samples)

            for ind, row in tqdm(chips_reproj.iterrows(), total=len(chips_reproj), ncols=100):
                dt = download_chip(folder_output=folder_output,
                                aoi_geometry=row.geometry,
                                s1_before_flood=s1_before_flood,
                                s1_flooded=s1_flooded,
                                s2_before_flood=s2_before_flood,
                                s2_flooded=s2_flooded,
                                flooded=flooded,
                                dem=dem,
                                lulc=lulc,
                                gsw=gsw,
                                ind=ind,
                                image_id_to_download=args.image_ids)
                for key in dict_time.keys():
                    if key in dt.keys():
                        dict_time[key].append(dt[key])
            
            for key in dict_time.keys():
                if len(dict_time[key])>0:
                    print(f'Mean time: {np.mean(dict_time[key]):0.3f}s n={len(dict_time[key])} {key}')
        else:
            print('DFO has no chip available in the GPKG file.')
    else:
        print('There is no GPKG file for this folder.')