from pathlib import Path
import sys
sys.path.insert(1, str(Path(__file__).parent.parent))

from SenForFlood import SenForFlood
from plots import plot_curves
from unet import UNet
import torch
import torch.nn as nn
import numpy
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print('Device:', DEVICE)

def train(model_id):
    model = UNet(in_channels=5, out_channels=2, dropout_val=0.2).to(DEVICE)
    dataset = SenForFlood(dataset_folder='/media/bruno/Matosak/SenForFlood', chip_size=256,
                        data_to_include=['s1_during_flood', 'terrain', 'flood_mask_v1.1'],
                        percentile_scale_bttm=5, percentile_scale_top=95,
                        countries=['Sudan'], #['Bangladesh', 'India', 'Pakistan', 'Sri Lanka', 'Afghanistan', 'Nepal', 'Buthan'],
                        # events = ['EMSR352', 'DFO_4463_Iran', 'DFO_4734_Iran', 'DFO_4816_Iran', 
                        #           'DFO_4883_Iran', 'DFO_4909_Iran', 'DFO_5066_Iran'],
                        use_data_augmentation=True)

    print(f'Total Chips: {len(dataset)}')

    cross_entropy = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))

    training_curves = {'Loss': {'train': [], 'eval': []},
                   'Accuracy': {'train': [], 'eval': []},
                   }
    
    model_folder = f'Models/{model_id}'
    os.makedirs(model_folder, exist_ok=True)

    def model_train(input_train, output_train):
        model.train()
        output_ = model(input_train)
        loss_train = cross_entropy(output_.type(torch.float32), output_train)
        model.zero_grad()
        loss_train.backward()
        opt.step()

        acc_train = (output_.argmax(1) == output_train.argmax(1)).type(torch.float).sum().item()/(output_train.shape[0]*output_train.shape[2]*output_train.shape[3])

        training_curves['Loss']['train'].append(loss_train.item())
        training_curves['Accuracy']['train'].append(acc_train)

    def model_val(input_eval, output_eval, plot_now=False):
        model.eval()
        with torch.no_grad():
            output_ = model(input_eval)
            acc_eval = (output_.argmax(1) == output_eval.argmax(1)).type(torch.float).sum().item()/(output_eval.shape[0]*output_eval.shape[2]*output_eval.shape[3])
            
            if plot_now:
                f, ax = plt.subplots(5,15, figsize=(15,5))
                o1 = output_.argmax(1).detach().cpu()
                o2 = output_eval.argmax(1).detach().cpu()
                for i in range(5):
                    for j in range(5):
                        ax[i,j*3].imshow(o1[i*5+j])
                        ax[i,j*3+1].imshow(o2[i*5+j])
                        ax[i,j*3+2].imshow(input_eval[i*5+j,:3].detach().cpu().moveaxis(0,-1))
                        ax[i,j*3].axis('off')
                        ax[i,j*3+1].axis('off')
                        ax[i,j*3+2].axis('off')
                plt.tight_layout()
                plt.savefig(f'{model_folder}/imgs/{step:05d}.png')
                plt.close()

            loss_eval = cross_entropy(output_.type(torch.float32), output_eval)
            training_curves['Loss']['eval'].append(loss_eval.item())
            training_curves['Accuracy']['eval'].append(acc_eval)

            return acc_eval

    max_epoch = 50
    train_size = 64
    batch_size = 64+32

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True)

    step = 0
    best_acc = 0
    for epoch in range(max_epoch):
        epoch_acc = []
        for ind, (s1b, te, fm) in enumerate(dataloader):
            # train
            input_train = torch.cat([s1b[:train_size,:3,:,:], te[:train_size,:,:,:]], dim=1).type(torch.float32).to(DEVICE)
            output_train = torch.nn.functional.one_hot((fm[:train_size,:,94:-94,94:-94]).to(torch.int64), 2).squeeze().moveaxis(-1,1).type(torch.float32).to(DEVICE)
            
            model_train(input_train, output_train)

            # validate
            input_eval = torch.cat([s1b[train_size:,:3,:,:], te[train_size:,:,:,:]], dim=1).type(torch.float32).to(DEVICE)
            output_eval = torch.nn.functional.one_hot((fm[train_size:,:,94:-94,94:-94]).to(torch.int64), 2).squeeze().moveaxis(-1,1).type(torch.float32).to(DEVICE)

            val_acc = model_val(input_eval, output_eval, ind+1==len(dataloader))
            epoch_acc.append(val_acc)

            print(f'Epoch: {epoch+1}, Step: {step}, val_acc: {training_curves["Accuracy"]["eval"][-1]:0.5f}, val_loss: {training_curves["Loss"]["eval"][-1]:0.5f}', end='\r' if ind<len(dataloader)-1 else '\n')
            
            step += 1

            if step%10==0:
                plot_curves(training_curves, [i for i in range(step)], 
                            smoothed=True,
                            save_path=f'{model_folder}/training_curves.png',
                            show=False,
                            title=f'Epochs: {epoch}')
            
        if numpy.mean(epoch_acc)>best_acc:
            model.eval()
            best_acc = numpy.mean(epoch_acc)
            torch.save(model.state_dict(), f'{model_folder}/model_e{epoch:02d}.pt')
            print(f"Best model ever stored (acc. {best_acc:.4f}).")

def predic_on_dataset():
    model_path = '/home/bruno/dataset_SenForFlood/models/UNet_SouthAsia.pt'
    model = UNet(in_channels=6, out_channels=2, dropout_val=0.2)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(DEVICE)

    model.eval()

    batch_size = 64+32
    train_size = 64

    dataset = SenForFlood(dataset_folder='/media/bruno/Matosak/SenForFlood', chip_size=256,
                        data_to_include=['s1_during_flood', 'global_surface_water', 'flood_mask'],
                        percentile_scale_bttm=5, percentile_scale_top=95,
                        countries=['Bangladesh', 'India', 'Pakistan', 'Sri Lanka', 'Afghanistan', 'Nepal', 'Buthan'],
                        use_data_augmentation=True)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True)
    
    for ind, (s1b, te, fm) in enumerate(dataloader):
        input_eval = torch.cat([s1b[train_size:,:3,:,:], te[train_size:,:,:,:]], dim=1).type(torch.float32).to(DEVICE)
        output_eval = torch.nn.functional.one_hot((fm[train_size:,:,94:-94,94:-94]*2).to(torch.int64), 3).squeeze().moveaxis(-1,1)[:,:2].type(torch.float32).to(DEVICE)
        with torch.no_grad():
            output_ = model(input_eval)
        
        out_ =    output_.detach().cpu().numpy()
        inp =  input_eval.detach().cpu().numpy()
        out = output_eval.detach().cpu().numpy()

        for i in range(16):
            f, ax = plt.subplots(1,4)
            ax[0].imshow(numpy.moveaxis(inp[i,:3,94:-94,94:-94], 0, -1))
            ax[1].imshow(numpy.moveaxis(inp[i,4,94:-94,94:-94], 0, -1))
            ax[2].imshow(numpy.argmax(out[i,:,:,:], axis=0), vmin=0, vmax=1)
            ax[3].imshow(numpy.argmax(out_[i,:,:,:], axis=0), vmin=0, vmax=1)
            for k in range(4):
                ax[k].axis('off')
            plt.tight_layout()
            plt.savefig(f'/home/bruno/dataset_SenForFlood/models/{ind:03d}-{i:02d}.png')
            plt.close()

def predict_on_image(image_path:str, output_path:str):
    import rasterio as r

    # load model
    model_path = '/media/bruno/Matosak/repos/SenForFlood/Examples/Models/test_Iran/model_e02.pt'
    model = UNet(in_channels=5, out_channels=2, dropout_val=0.2)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(DEVICE)
    model.eval()

    # load SenForFlood for getting scalling function
    dataset = SenForFlood(dataset_folder='/media/bruno/Matosak/SenForFlood', chip_size=256,
                        data_to_include=['s1_during_flood', 'terrain', 'flood_mask_v1.1'],
                        percentile_scale_bttm=5, percentile_scale_top=95,
                        countries=['Bangladesh', 'India', 'Pakistan', 'Sri_Lanka', 'Afghanistan', 'Nepal', 'Buthan'],
                        use_data_augmentation=True)
    
    input_dataset = r.open(image_path)
    result = numpy.zeros([input_dataset.height, input_dataset.width], dtype=numpy.uint8)+255

    # DO THE PREDICTION
    # collect batches with each sample coordinates
    batch_size = 64
    batches = []
    batch = []
    for i in range(0,input_dataset.height-256,256-94-94):
        for j in range(0,input_dataset.width-256,256-94-94):
            batch.append([i,j])
            if len(batch)==batch_size:
                batches.append(batch)
                batch = []
    if len(batch)>0:
        batches.append(batch)
    
    # iterates and predicts over each batch
    for batch in tqdm(batches, ncols=100):
        data = numpy.zeros([batch_size,5,256,256], dtype=numpy.float32)
        for i in range(len(batch)):
            sample = numpy.zeros([5,256,256], dtype=numpy.float32)
            d = input_dataset.read(window=r.windows.Window(batch[i][1], batch[i][0], 256, 256))
            sample[:,:d.shape[1],:d.shape[2]] = d
            data[i,:3] = dataset.scale_data('s1_during_flood', sample[:3,:,:])
            data[i,3:] = dataset.scale_data('terrain', sample[3:,:,:])

        data[numpy.isnan(data)] = 0
        pred = model(torch.from_numpy(data).to(DEVICE))
        for i in range(len(batch)):
            result[94+batch[i][0]:256-94+batch[i][0], 94+batch[i][1]:256-94+batch[i][1]] = numpy.argmax(pred[i].detach().cpu().numpy(), 0)

    # save image
    with r.Env():

        # Write an array as a raster band to a new 8-bit file. For
        # the new file's profile, we start with the profile of the source
        profile = input_dataset.profile

        # And then change the band count to 1, set the
        # dtype to uint8, and specify LZW compression.
        profile.update(
            dtype=r.uint8,
            nodata=255,
            count=1,
            compress='lzw')

        with r.open(output_path, 'w', **profile) as dst:
            dst.write(result.astype(r.uint8), 1)

def predict_on_image_bayesian_dropout(input_path:str, output_path:str, n_iterations:int=10):
    import rasterio as r

    # load model
    model_path = '/home/bruno/Documents/dataset_Sen2Flood/models/UNet_SouthAsia.pt'
    model = UNet(in_channels=5, out_channels=2, dropout_val=0.2)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

    # load SenForFlood for getting scalling function
    dataset = SenForFlood(dataset_folder='/media/bruno/Matosak/SenForFlood', chip_size=256,
                        data_to_include=['s1_during_flood', 'terrain', 'flood_mask'],
                        percentile_scale_bttm=5, percentile_scale_top=95,
                        countries=['Bangladesh', 'India', 'Pakistan', 'Sri_Lanka', 'Afghanistan', 'Nepal', 'Buthan'],
                        use_data_augmentation=True)
    
    input_dataset = r.open(input_path)
    result_class = numpy.zeros([input_dataset.height, input_dataset.width], dtype=numpy.uint8)+255
    result_eu = numpy.zeros([input_dataset.height, input_dataset.width], dtype=numpy.float32)-1

    # DO THE PREDICTION
    # collect batches with each sample coordinates
    batch_size = 64
    batches = []
    batch = []
    for i in range(0,input_dataset.height-256,256-94-94):
        for j in range(0,input_dataset.width-256,256-94-94):
            batch.append([i,j])
            if len(batch)==batch_size:
                batches.append(batch)
                batch = []
    if len(batch)>0:
        batches.append(batch)
    
    # iterates and predicts over each batch
    for batch in tqdm(batches, ncols=100):
        data = numpy.zeros([batch_size,5,256,256], dtype=numpy.float32)
        for i in range(len(batch)):
            sample = numpy.zeros([5,256,256], dtype=numpy.float32)
            d = input_dataset.read(window=r.windows.Window(batch[i][1], batch[i][0], 256, 256))
            sample[:,:d.shape[1],:d.shape[2]] = d
            data[i,:3] = dataset.scale_data('s1_during_flood', sample[:3,:,:])
            data[i,3:] = dataset.scale_data('terrain', sample[3:,:,:])

        data[numpy.isnan(data)] = 0
        
        # execute the MC iterations
        pred = []
        for iteration in range(n_iterations):
            result = model(torch.from_numpy(data).to(DEVICE))
            pred.append(result.detach().cpu().numpy())
        pred = numpy.mean(numpy.asarray(pred),0)
        for i in range(len(batch)):
            result_class[94+batch[i][0]:256-94+batch[i][0], 94+batch[i][1]:256-94+batch[i][1]] = numpy.argmax(pred[i], 0)
            result_eu[94+batch[i][0]:256-94+batch[i][0], 94+batch[i][1]:256-94+batch[i][1]] = -1*numpy.sum(numpy.log(pred[i])*pred[i], 0)
    
    result_eu[~numpy.isfinite(result_eu)] = -1

    # save image
    with r.Env():

        # Write an array as a raster band to a new 8-bit file. For
        # the new file's profile, we start with the profile of the source
        profile = input_dataset.profile

        # And then change the band count to 1, set the
        # dtype to uint8, and specify LZW compression.
        profile.update(
            dtype=r.uint8,
            nodata=255,
            count=1,
            compress='lzw')

        with r.open(output_path+'-class.tif', 'w', **profile) as dst:
            dst.write(result_class.astype(r.uint8), 1)
        
        profile.update(
            dtype=r.float32,
            nodata=-1,
            count=1,
            compress='lzw')

        with r.open(output_path+'-uncertainty.tif', 'w', **profile) as dst:
            dst.write(result_eu.astype(r.float32), 1)

if __name__=='__main__':
    train(model_id='test_Sudan')
    # predict_on_image(
    #     '/media/bruno/Matosak/repos/bayesian-UNET-dropout/images/Bangladesh_2024-06_mosaic_2.tif', 
    #     '/media/bruno/Matosak/repos/SenForFlood/Examples/Models/test_Iran/model_e02.tif')
    # pass