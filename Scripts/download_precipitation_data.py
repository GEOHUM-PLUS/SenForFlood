import ee
import geopandas
import glob
import rasterio as r
from tqdm.auto import tqdm
from shapely.geometry import Point
import datetime
import os
import pickle
import time

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
    
if __name__=='__main__':
    ee.Authenticate()
    ee.Initialize(project='adept-fountain-392019')

    # getting files
    files = glob.glob('../../SenForFlood/*/*/*/s1_during_flood/*.tif')
    files.extend(glob.glob('../../SenForFlood/*/*/s1_during_flood/*.tif'))
    files.sort()

    # open FeatureCollection from GEE
    daily_precipitation = ee.ImageCollection('UCSB-CHC/CHIRPS/V3/DAILY_SAT')

    # iterate over samples to get time series data
    for sample_path in tqdm(files):
        # create folder
        folder_path = sample_path.rsplit('/', 2)[0]+'/precipitation_30d_local'
        data_path = folder_path+'/'+sample_path.rsplit('/',1)[-1].replace('s1_during_flood.tif', 'precipitation_30d_local.pkl')

        if not os.path.isfile(data_path):
            os.makedirs(folder_path, exist_ok=True)

            centroid, image_date = return_centroid_in_wgs84_and_date(sample_path)

            daily_precipitation_filtered = daily_precipitation.filterDate(
                (image_date-datetime.timedelta(days=32)).strftime('%Y-%m-%d'),
                image_date.strftime('%Y-%m-%d')
            )

            def get_time_series(image):
                stats = image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=ee.Geometry.Point([centroid.x, centroid.y]),
                    scale=10,
                    maxPixels=1e9
                )

                return ee.Feature(None, {
                    'date': image.date().format('YYYY-MM-dd'),
                    'precipitation': stats.get('precipitation')
                })

            timeSeries_info = daily_precipitation_filtered.map(get_time_series).getInfo()

            time_series = {
                'precipitation': [timeSeries_info['features'][i]['properties']['precipitation'] for i in range(len(timeSeries_info['features']))] if 'precipitation' in timeSeries_info['features'][0]['properties'].keys() else [50]*len(timeSeries_info['features']),
                'date': [timeSeries_info['features'][i]['properties']['date'] for i in range(len(timeSeries_info['features']))],
            }

            with open(data_path, 'wb') as handle:
                pickle.dump(time_series, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            # let's no anger the google cloud API gods
            # I can do up t0 1200 queries per minute. Currently, I am doing 3 per second, which is around 180.
            # time.sleep(1)