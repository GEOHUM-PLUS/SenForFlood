import torch
import torchvision
from tifffile import imread
import glob
import os
import random
import warnings
import numpy as np
import torchvision.transforms.functional
# warnings.filterwarnings("ignore")

class SenForFlood(torch.utils.data.Dataset):
    def __init__(self, dataset_folder:str, source:str='DFO', shuffle_seed:int=0, chip_size:int=512, events:list[str]=None, countries:list[str]=None,
                 data_to_include:list[str]=['s1_before_flood', 's1_during_flood', 's2_before_flood', 's2_during_flood', 'flood_mask_v1.1', 'terrain', 'LULC', 'global_surface_water', 'SatCLIP_embedding'],
                 use_data_augmentation:bool=False, scale_0_1:bool=True, percentile_scale_bttm:int=1, percentile_scale_top:int=99):
        '''
        Dataset reader for SenForFlood.

        Parameters
        ---
        dataset_folder: str
            Path to folder containing the dataset. The folder that contains CEMS
            and DFO folders.
        source: str (default 'DFO')
            source from which to get samples. Either DFO or CEMS.
        shuffle_seed: int (default 0)
            Seed for shuffling the dataset.
        chip_size: int (default 512)
            Chip size for tiling the samples. Valid values are 32, 64, 128, 256, 
            and 512.
        events: list[str] (default None)
            List of events to be included when returning samples. If different than None,
            parameters 'source' and 'countries' will be ignored and 'events' will have
            preference.
        countries: list[str] (default None)
            List of countries to be included when returning samples. Name of 
            countries should follow the name of folders inside "Data" folder.
            When None is given, all countries are included. Irrelevant when
            using 'CEMS' as source.
        data_to_include: list[str]
            List of data names to include when returning the samples. Should
            follow the name of the last folders inside the DFO or CEMS. Valid
            values are 's1_before_flood', 's1_during_flood', 's2_before_flood', 
            's2_during_flood', 'flood_mask_v1.1', 'terrain', 'LULC', 
            'global_surface_water' and 'SatCLIP_embedding'.
        use_data_augmentation: bool (default False)
            Wheter or not to do data augmentation.
        scale_0_1: bool (default True)
            Wheter or not to scale samples between 0 and 1. If not, samples are
            returned as their original raster values.
        percentile_scale_bttm: int (default 1)
            Percentile to use as bottom value when scaling samples. Valid values
            are 0, 1, 2, 5, 90, 95, 98, 99, and 100. Percentiles are pre-loaded
            and are not calculated in real time.
        percentile_scale_top: int (default 99)
            Percentile to use as top value when scaling samples. Valid values
            are 0, 1, 2, 5, 90, 95, 98, 99, and 100. Percentiles are pre-loaded
            and are not calculated in real time.
        '''
        super().__init__()

        if not os.path.isdir(dataset_folder):
            raise ValueError(f'"{dataset_folder}" is not a folder.')
        if not percentile_scale_top in [0, 1, 2, 5, 90, 95, 98, 99, 100] or not percentile_scale_bttm in [0, 1, 2, 5, 90, 95, 98, 99, 100]:
            raise ValueError('Invalid percentile for scaling data, valid values are 0, 1, 2, 5, 90, 95, 98, 99, and 100.')
        if not int(chip_size) in [32, 64, 128, 256, 512]:
            raise ValueError('Invalid value encountered for chip_size, value must be 32, 64, 128, 256 or 512.')
        if not source in ['CEMS', 'DFO']:
            raise ValueError('Invalid source. Valid values are "CDSE" or "DFO".')
        for d in data_to_include:
            if not d in ['s1_before_flood', 's1_during_flood', 's2_before_flood', 's2_during_flood', 'flood_mask_v1.1', 'terrain', 'LULC', 'global_surface_water', 'SatCLIP_embedding']:
                raise ValueError(f'Invalid value encountered for data_to_include. Valid values are "s1_before_flood", "s1_during_flood", "s2_before_flood", "s2_during_flood", "flood_mask_v1.1", "terrain", "LULC", "global_surface_water" and "SatCLIP_embedding".')
        
        self.percentile_top = percentile_scale_top
        self.percentile_bttm = percentile_scale_bttm
        if events is None:
            if source == 'DFO':
                if countries is None:
                    self.samples_ids = glob.glob(os.path.join(dataset_folder, 'DFO/*/DFO_*_*/flood_mask_v1.1/*_flood_mask_v1.1.tif'))
                else:
                    self.samples_ids = []
                    for country in countries:
                        self.samples_ids.extend(glob.glob(os.path.join(dataset_folder, f'DFO/{country}/DFO_*_*/flood_mask_v1.1/*_flood_mask_v1.1.tif')))
            elif source == 'CEMS':
                self.samples_ids = glob.glob(os.path.join(dataset_folder, f'CEMS/*/flood_mask_v1.1/*_flood_mask_v1.1.tif'))
            else:
                raise ValueError(f'{source} not valid as source. DFO or CEMS.')
        else:
            self.samples_ids = []
            for event in events:
                if event.split('_')[0]=='DFO':
                    country = event.split('_',2)[-1]
                    if os.path.isdir(os.path.join(dataset_folder, f'DFO/{country}/{event}')):
                        self.samples_ids.extend(glob.glob(os.path.join(dataset_folder, f'DFO/{country}/{event}/flood_mask_v1.1/*_flood_mask_v1.1.tif')))
                    else:
                        print(f'Event "{event}" not found in "{os.path.join(dataset_folder, f"DFO/{country}/{event}")}".')
                else:
                    if os.path.isdir(os.path.join(dataset_folder, f'CEMS/{event}')):
                        self.samples_ids.extend(glob.glob(os.path.join(dataset_folder, f'CEMS/{event}/flood_mask_v1.1/*_flood_mask_v1.1.tif')))
                    else:
                        print(f'Event "{event}" not found in "{os.path.join(dataset_folder, f"CEMS/{event}")}".')

        self.samples_ids.sort()
        self.samples_ids = [[sample_id,i] for i in range(int(512/int(chip_size))**2) for sample_id in self.samples_ids]
        if len(self.samples_ids)==0:
            print('No samples found with the options given. Please check the dataset location, source, events or crountries given.')
        random.Random(shuffle_seed).shuffle(self.samples_ids)
        self.data_to_include = data_to_include
        self.use_data_augmentation = use_data_augmentation
        self.chip_size=int(chip_size)
        self.scale_0_1 = scale_0_1

        # loading limits
        if self.scale_0_1:
            import pickle
            from pathlib import Path
            with open(Path(Path(__file__).parent / 'percentile_limits.pickle'), 'rb') as f:
                self.STRETCH_LIMITS = pickle.load(f)

    def __len__(self):
        return len(self.samples_ids)

    def __getitem__(self, index):
        result = []
        sample_id = self.samples_ids[index]

        # iterates over data to include
        for dti in self.data_to_include:
            if dti=='SatCLIP_embedding':
                data = np.load(sample_id[0].replace('flood_mask_v1.1', dti).replace('.tif', '.npy'))
            else:
                # tifffile reads data faster than rasterio when the whole file is needed (not windowed)
                data = imread(sample_id[0].replace('flood_mask_v1.1', dti), selection=(slice(int((sample_id[1]%(512/self.chip_size))*self.chip_size),int((sample_id[1]%(512/self.chip_size))*self.chip_size+self.chip_size)),
                                                                                    slice(int(int(sample_id[1]/(512/self.chip_size))*self.chip_size),int(int(sample_id[1]/(512/self.chip_size))*self.chip_size+self.chip_size))
                                                                                    )).astype(np.float32)
                # add dimention to datasets with one band
                if dti == 'flood_mask_v1.1' or dti == 'LULC':
                    data = np.expand_dims(data, -1)
                # make shape pytorch-like
                data = np.moveaxis(data, -1, 0)
                    
                # scales data between 0 and 1
                if self.scale_0_1:
                    data = self.scale_data(dti, data)

            # store to return later with others
            result.append(data)
        
        # in case data augmentation is needed
        if self.use_data_augmentation:
            augment_flip_h = bool(random.randint(0,1))
            augment_flip_v = bool(random.randint(0,1))
            augment_rotation = random.randint(0,3)*90
            return tuple(self.augment(torch.tensor(arg), augment_flip_h, augment_flip_v, augment_rotation) for arg in result)
        
        # final data return
        return tuple(arg for arg in result)
    
    def augment(self, data, flip_h:bool, flip_v:bool, rotation:int):
        if len(data.shape)==1:
            return data
        if flip_h:
            data = torchvision.transforms.functional.hflip(data)
        if flip_v:
            data = torchvision.transforms.functional.vflip(data)
        if rotation!=0:
            data = torchvision.transforms.functional.rotate(data, rotation)
        return data
    
    def scale_data(self, data_type, data):
        # follows scales only if needed
        if data_type in ['s1_before_flood', 's1_during_flood', 's2_before_flood', 's2_during_flood', 'terrain']:
            for i in range(data.shape[0]):
                data[i,:,:] = (data[i,:,:]-self.STRETCH_LIMITS[data_type][i][str(int(self.percentile_bttm))])/(self.STRETCH_LIMITS[data_type][i][str(int(self.percentile_top))]-self.STRETCH_LIMITS[data_type][i][str(int(self.percentile_bttm))])
        else:
            for i in range(data.shape[0]):
                data[i,:,:] = (data[i,:,:]-self.STRETCH_LIMITS[data_type][i]['0'])/(self.STRETCH_LIMITS[data_type][i]['100']-self.STRETCH_LIMITS[data_type][i]['0'])
        
        # clipping to 0 1
        data = np.clip(data, a_min=0, a_max=1)

        return data

if __name__=='__main__':
    senforflood = SenForFlood('/media/bruno/Matosak/SenForFlood', events=['EMSR352'], chip_size=256)

    print('Total Samples:', len(senforflood))

    for ind, samples in enumerate(senforflood):
        for s in samples:
            print(s.shape)
        break