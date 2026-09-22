import ee
import geopandas
import glob
import rasterio as r
from tqdm.auto import tqdm
from shapely.geometry import Point, box
import datetime
import os
import pickle
import time
import requests

def return_centroid_in_wgs84_and_date(dataset:str):
    dataset = r.open(dataset)

    # getting centroid of the sample
    df = geopandas.GeoDataFrame(
        geometry=[Point((dataset.bounds.right+dataset.bounds.left)/2, (dataset.bounds.top+dataset.bounds.bottom)/2)],
        crs='EPSG:3857' # dataset.crs
    )

    # reprojecting it to WGS84
    df.to_crs('EPSG:4326', inplace=True)

    # getting the date
    date = dataset.read(dataset.count, window=r.windows.Window(dataset.width/2, dataset.height/2, 1, 1))[0][0]
    date = datetime.datetime.strptime('2000-01-01', '%Y-%m-%d')+datetime.timedelta(days=int(date))

    return df.geometry.values[0], date

def return_region_in_wgs84_and_date(path_ref_file:str):
    dataset = r.open(path_ref_file)

    # getting centroid of the sample
    df = geopandas.GeoDataFrame(
        geometry=[box(
            dataset.bounds.left, 
            dataset.bounds.bottom, 
            dataset.bounds.right, 
            dataset.bounds.top
        )],
        crs='EPSG:3857' # dataset.crs
    )

    # reprojecting it to WGS84
    df.to_crs('EPSG:4326', inplace=True)

    # getting the date
    date = dataset.read(dataset.count, window=r.windows.Window(dataset.width/2, dataset.height/2, 1, 1))[0][0]
    date = datetime.datetime.strptime('2000-01-01', '%Y-%m-%d')+datetime.timedelta(days=int(date))

    return df.geometry.values[0], date

def clip_in_collection(image):
    return image.clip(region)
    
if __name__=='__main__':
    ee.Authenticate()
    ee.Initialize(project='adept-fountain-392019')

    # getting files
    files = glob.glob('../../datasets/SenForFlood/*/*/*/s1_during_flood/*.tif')
    files.extend(glob.glob('../../datasets/SenForFlood/*/*/s1_during_flood/*.tif'))
    files.sort()

    # open FeatureCollection from GEE
    MERIT_hydro = ee.Image('MERIT/Hydro/v1_0_1').select(['upa', 'hnd'])

    # iterate over samples to get time series data
    for sample_path in tqdm(files):
        # create folder
        folder_path = sample_path.rsplit('\\', 2)[0]+'\\MERIT_hydro'
        data_path = folder_path+'\\'+sample_path.rsplit('\\',1)[-1].replace('s1_during_flood.tif', 'MERIT_hydro.tif')
        
        if os.path.isfile(data_path):
            try:
                data = r.open(data_path)
                continue
            except:
                os.remove(data_path)
        
        download_pending = True
        while download_pending:
            try:
                os.makedirs(folder_path, exist_ok=True)
                if os.path.isfile(data_path):
                    os.remove(data_path)

                region, image_date = return_region_in_wgs84_and_date(sample_path)
                region = ee.Geometry.Polygon(list(region.exterior.coords))

                url = MERIT_hydro.getDownloadURL(
                    {'region': region,
                    'dimensions': '512x512',
                    'crs': 'EPSG:3857',
                    'format': 'GEO_TIFF'
                    }
                )
                
                response = requests.get(url)
                response.raise_for_status()

                with open(data_path, 'wb') as f:
                    f.write(response.content)
                download_pending = False
            except Exception as e:
                print(e)
                print('\nDownload failed, retrying in 60 seconds...')
                time.sleep(60)