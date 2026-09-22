import torch
import torchvision
from tifffile import imread
import glob
import os
import random
import warnings
import numpy as np
import torchvision.transforms.functional
import pickle
from pathlib import Path
# warnings.filterwarnings("ignore")

class SenForFlood(torch.utils.data.Dataset):
    def __init__(
        self, 
        dataset_folder:str, 
        source:str='DFO', 
        shuffle_seed:int=0, 
        chip_size:int=512, 
        events:list[str]=None, 
        countries:list[str]=None,
        data_to_include:list[str]=[
            's1_before_flood', 's1_during_flood', 's2_before_flood', 's2_during_flood', 'flood_mask_v1.1', 'terrain', 
            'LULC', 'global_surface_water', 'SatCLIP_embedding', 'otsu_water_during_flood', 'precipitation_30d_local',
            'MERIT_hydro'],
        use_data_augmentation:bool=False,
        normalize:bool=False,
        normalize_s1_to_match_terramesh:bool=False
    ):
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
        normalize: bool (default True)
            Wheter or not to normalize the samples.
        '''
        super().__init__()

        if not os.path.isdir(dataset_folder):
            raise ValueError(f'"{dataset_folder}" is not a folder.')
        if not int(chip_size) in [8, 16, 32, 64, 128, 256, 512]:
            raise ValueError('Invalid value encountered for chip_size, value must be 32, 64, 128, 256 or 512.')
        if not source in ['CEMS', 'DFO']:
            raise ValueError('Invalid source. Valid values are "CDSE" or "DFO".')
        for d in data_to_include:
            if not d in ['s1_before_flood', 's1_during_flood', 's2_before_flood', 's2_during_flood', 'flood_mask_v1.1', 'terrain', 'LULC', 'global_surface_water', 'otsu_water_during_flood', 'precipitation_30d_local', 'MERIT_hydro']:
                raise ValueError(f'Invalid value encountered for data_to_include. Valid values are "s1_before_flood", "s1_during_flood", "s2_before_flood", "s2_during_flood", "flood_mask_v1.1", "terrain", "LULC", "global_surface_water", "otsu_water_during_flood", "precipitation_30d_local", and "MERIT_hydro".')
        
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
        self.normalize = normalize
        self.normalize_s1_to_match_terramesh = normalize_s1_to_match_terramesh
        self.terramesh_sentinel1_norm_statistics = {'mean': [-10.793, -17.198], 'std': [4.278, 4.346]}

        # loading limits
        with open(Path(Path(__file__).parent / 'percentile_limits.pickle'), 'rb') as f:
            self.STRETCH_LIMITS = pickle.load(f)

    def __len__(self):
        return len(self.samples_ids)

    def __getitem__(self, index):
        result = []
        sample_id = self.samples_ids[index]

        # iterates over data to include
        for dti in self.data_to_include:
            if dti=='precipitation_30d_local':
                # with open(sample_id[0].replace('flood_mask_v1.1', dti).replace('.tif', '.pkl'), "rb") as input_file:
                #     data = pickle.load(input_file)
                data = imread(sample_id[0].replace('flood_mask_v1.1', dti)).astype(np.float32) # np.asarray(data['precipitation'], dtype=np.float32)[:,None]
                data[data==-9999] = 0
                data = np.moveaxis(data, -1, 0)[:,None,...]
            else:
                # tifffile reads data faster than rasterio when the whole file is needed (not windowed)
                data = imread(
                    sample_id[0].replace('flood_mask_v1.1', dti), 
                    selection=(
                        slice(int((sample_id[1]%(512/self.chip_size))*self.chip_size),int((sample_id[1]%(512/self.chip_size))*self.chip_size+self.chip_size)),
                        slice(int(int(sample_id[1]/(512/self.chip_size))*self.chip_size),int(int(sample_id[1]/(512/self.chip_size))*self.chip_size+self.chip_size))
                    )
                ).astype(np.float32)
                
                # add dimention to datasets with one band
                if dti == 'flood_mask_v1.1' or dti == 'LULC' or dti == 'otsu_water_during_flood':
                    data = np.expand_dims(data, -1)
                
                # correcting nodata in MERIT_hydro
                if dti=='MERIT_hydro':
                    data[data==-9999] = 0
                
                # make shape pytorch-like
                data = np.moveaxis(data, -1, 0)
                    
            # normalizes data
            if self.normalize:
                data = self.normalize_data(dti, data, clip_std=3.0)

            # store to return later with others
            data = torch.Tensor(data).to(torch.float32)
            result.append(data)
        
        # in case data augmentation is needed
        if self.use_data_augmentation:
            augment_flip_h = bool(random.randint(0,1))
            augment_flip_v = bool(random.randint(0,1))
            augment_rotation = random.randint(0,3)*90
            return tuple(self.augment(arg, augment_flip_h, augment_flip_v, augment_rotation) for arg in result)
        
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
    
    def normalize_data(self, data_type, data, clip_std=None):
        # follows scales only if needed
        if data_type == 'LULC':
            # data = np.moveaxis(get_one_hot((data[0]/10).astype(np.byte), 11), -1,0)
            data = torch.nn.functional.one_hot(torch.Tensor((data[0]/10)-1).to(torch.long), num_classes=10).moveaxis(-1,0).numpy()
        elif data_type == 'precipitation_30d_local':
            data = np.log1p(np.clip(data, min=0))
            data = (data-self.STRETCH_LIMITS[data_type]['log_mean'])/self.STRETCH_LIMITS[data_type]['log_std']
        elif data_type in ['terrain', 'MERIT_hydro']:
            data = np.log1p(np.clip(data, min=0))
            for i in range(data.shape[0]):
                data[i,:,:] = (data[i,:,:]-self.STRETCH_LIMITS[data_type][i]['log_mean'])/self.STRETCH_LIMITS[data_type][i]['log_std']
        elif data_type in ['s1_before_flood', 's1_during_flood'] and self.normalize_s1_to_match_terramesh:
            for i in range(2):
                data[i] = (data[i]-self.terramesh_sentinel1_norm_statistics['mean'][i])/(self.terramesh_sentinel1_norm_statistics['std'][i])
        else:
            for i in range(data.shape[0]):
                data[i,:,:] = (data[i,:,:]-self.STRETCH_LIMITS[data_type][i]['mean'])/self.STRETCH_LIMITS[data_type][i]['std']
        
        if clip_std and not data_type in ['s1_before_flood', 's1_during_flood']:
            return np.clip(data, min=-clip_std, max=clip_std)
        return data
    
    def unnormalize_data(self, data_type, data):
        for i in range(data.shape[0]):
            data[i,:,:] = (data[i,:,:]*self.STRETCH_LIMITS[data_type][i]['std'])+self.STRETCH_LIMITS[data_type][i]['mean']
        
        return data

# Source - https://stackoverflow.com/a
# Posted by Martin Thoma, modified by community. See post 'Timeline' for change history
# Retrieved 2025-12-13, License - CC BY-SA 4.0
def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

if __name__=='__main__':
    # creates pickle file with the parameters for normalization
    from tqdm.auto import tqdm
    import pickle

    for dti in ['precipitation_30d_local']:
        print(dti)

        senforflood = SenForFlood(
            '../../datasets/SenForFlood',
            # countries=['Brazil'],
            data_to_include=[dti],
            chip_size=512,
            use_data_augmentation=False,
            normalize=False
        )

        bs = 256
        dataloader = torch.utils.data.DataLoader(senforflood, batch_size=bs, drop_last=False, num_workers=4)

        _ = senforflood[0]

        with open('percentile_limits.pickle', 'rb') as f:
            statistics = pickle.load(f)
        # statistics[dti] = [{}]*_[0].shape[0]
        statistics[dti] = {}

        for b in range(_[0].shape[0]):
            data = np.zeros([len(senforflood), 512, 512], dtype=np.float32)
            for ind, [samples] in tqdm(enumerate(dataloader), total=len(dataloader), ncols=100):
                data[ind*bs:ind*bs+samples.shape[0]] = samples[:,b].numpy()

            if dti not in ['terrain', 'LULC', 'global_surface_water', 'precipitation_30d_local', 'MERIT_hydro']:
                data[data==0] = None

            data = data[np.isfinite(data)]
            
            statistics[dti][b]['mean'] = np.mean(data)
            statistics[dti][b]['std'] = np.std(data)

            data = np.log1p(np.clip(data, min=0))

            statistics[dti][b]['log_mean'] = np.mean(data)
            statistics[dti][b]['log_std'] = np.std(data)

            print('Mean:', statistics[dti][b]['mean'])
            print('STD:', statistics[dti][b]['std'])

        # data = np.zeros([len(senforflood), 32, 1, 64, 64], dtype=np.float32)
        # for ind, [samples] in tqdm(enumerate(dataloader), total=len(dataloader), ncols=100):
        #     data[ind*bs:ind*bs+samples.shape[0]] = samples.numpy()

        # data = data[np.isfinite(data)]
        
        # statistics[dti]['mean'] = np.mean(data)
        # statistics[dti]['std'] = np.std(data)

        # data = np.log1p(np.clip(data, min=0))

        # statistics[dti]['log_mean'] = np.mean(data)
        # statistics[dti]['log_std'] = np.std(data)

        # print('Mean:', statistics[dti]['mean'])
        # print('STD:', statistics[dti]['std'])
    
        with open('percentile_limits.pickle', 'wb') as handle:
            pickle.dump(statistics, handle, protocol=pickle.HIGHEST_PROTOCOL)