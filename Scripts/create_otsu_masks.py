import rasterio as r
import glob
import os
from tqdm.auto import tqdm
from skimage.filters import threshold_otsu
from skimage.morphology import area_opening, area_closing

import sys
sys.path.insert(1, '/media/bruno/Matosak/repos/SenForFlood')

from SenForFlood import SenForFlood

sff = SenForFlood(
    dataset_folder='/media/bruno/Matosak/SenForFlood',
    chip_size=512,
    # countries=['Bangladesh'],
    source='CEMS',
    data_to_include=['s1_during_flood', 'flood_mask_v1.1'],
    use_data_augmentation=True,
    normalize=True
)

files = glob.glob('/media/bruno/Matosak/SenForFlood/DFO/*/*/s1_during_flood/*.tif')
files.extend(glob.glob('/media/bruno/Matosak/SenForFlood/CEMS/*/s1_during_flood/*.tif'))
print(f'{len(files):,d} Samples.')

for file in tqdm(files):
    dataset = r.open(file)
    data = dataset.read()
    data = sff.normalize_data('s1_during_flood', data)[1]
    threshold_value = min(max(-0.75, threshold_otsu(data)), 0.75)
    
    mask = area_closing(area_opening(data<threshold_value))

    with r.Env():
        profile = dataset.profile
        profile.update(
            dtype=r.uint8,
            count=1
        )

        os.makedirs(file.replace('s1_during_flood', 'otsu_water_during_flood').rsplit('/',1)[0], exist_ok=True)

        with r.open(file.replace('s1_during_flood', 'otsu_water_during_flood'), 'w', **profile) as dst:
            dst.write(mask.astype(r.uint8), 1)