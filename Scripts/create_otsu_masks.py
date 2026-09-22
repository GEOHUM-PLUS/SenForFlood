import rasterio as r
import glob
import os
from tqdm.auto import tqdm
from skimage.filters import threshold_otsu
from skimage.morphology import area_opening, area_closing
import tifffile
import numpy as np

def create_otsu_mask_for_folder(path_folder):
    files = glob.glob(os.path.join(path_folder, 's1_during_flood/*.tif'))
    print(len(files))

    # load data first
    data = np.zeros([len(files), 512, 512], dtype=np.float32)
    for i,file in tqdm(zip(range(len(files)),files), ncols=100, total=len(files)):
        data[i] = np.squeeze(tifffile.imread(file, selection=(slice(0,512), slice(0,512), slice(1,2))))

    threshold_value = threshold_otsu(data)
    for file in tqdm(files, ncols=100):
        dataset = r.open(file)
        data = dataset.read(2)
        
        mask = area_closing(area_opening(data<threshold_value))

        with r.Env():
            profile = dataset.profile
            profile.update(
                dtype=r.uint8,
                count=1
            )

            os.makedirs(file.replace('s1_during_flood', 'otsu_water_during_flood').rsplit('\\',1)[0], exist_ok=True)
            with r.open(file.replace('s1_during_flood', 'otsu_water_during_flood'), 'w', **profile) as dst:
                dst.write(mask.astype(r.uint8), 1)

if __name__ == '__main__':
    folders_dfo = glob.glob(r"F:\datasets\SenForFlood\DFO\*\*")

    for i,folder in zip(range(len(folders_dfo)),folders_dfo):
        if os.path.isdir(folder):
            print('------------------------------')
            print(f'{(i+1)}/{len(folders_dfo)}', folder.split('\\')[-1])
            create_otsu_mask_for_folder(folder)
