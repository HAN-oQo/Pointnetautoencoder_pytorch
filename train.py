import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import argparse
import os
import random
import h5py
import easydict

from tqdm import tqdm

from model.model import PointNetAE
from model.model1 import PointNetAE1
from dataset.dataset import Dataset
from dataset.dataset import create_datasets_and_dataloaders

import pytorch3d
from pytorch3d.loss import chamfer_distance

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--train', default= True, action='store_true')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size')
parser.add_argument('--n_epochs', type=int, default=50,
                    help='number of epochs')
parser.add_argument('--n_workers', type=int, default=4,
                    help='number of data loading workers')

parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='beta 1')
parser.add_argument('--beta2', type=float, default=0.999, help='beta 2')
parser.add_argument('--step_size', type=int, default=20, help='step size')
parser.add_argument('--gamma', type=float, default=0.5, help='gamma')

parser.add_argument('--in_data_file', type=str,
                    default='data/ModelNet/modelnet_classification.h5',
                    help="data directory")
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--model_type', type = str, default = 'hankyu1', 
                    choices= ['hankyu', 'hankyu1'], help = 'model type')
# parser.add_argument('--out_dir', type=str, default='outputs',
#                     help='output directory')
args = parser.parse_args()

def run_train(epoch, autoencoder, train_dataset, train_dataloader, args, writer):
    best_loss = 1e20
    n_data = len(train_dataset)
  
    total_loss = 0.0
    mode = 'Train'
    # Create a progress bar. 
    pbar = tqdm(total = n_data, leave = False)
    epoch_str = '' if epoch is None else '[Epoch {}/{}]'.format(
        str(epoch).zfill(len(str(args.n_epochs))), args.n_epochs)
    

    for i, data in enumerate(train_dataloader):

        points, gt_classes = data
        points = points.to(device)
        
        # Reset Gradient
        optimizer.zero_grad()

        reconstructed_points = autoencoder.train()(points)
        reconstructed_points = reconstructed_points.transpose(1,2)
        
        loss_chamfer, _ = chamfer_distance(points, reconstructed_points)
        train_loss = loss_chamfer
        
        train_loss.backward()
        optimizer.step()
        if writer is not None:
            assert(epoch is not None)
            step = epoch * len(train_dataloader) + i
            writer.add_scalar('Loss/Train', train_loss, step)

        total_loss += train_loss * args.batch_size
        pbar.set_description('{} {} Loss: {:f}'.format(
        epoch_str, mode, train_loss))

        pbar.update(args.batch_size)
    
    pbar.close()       
    epoch_loss = total_loss / float(n_data)  

              
    return epoch_loss


def run_test(autoencoder, test_dataset, test_dataloader, args, writer):
    total_loss = 0.0
    n_data = len(test_dataset)
    # Create a progress bar. 
    pbar = tqdm(total = n_data, leave = False)
    mode = 'Test'


    for i, data in enumerate(test_dataloader):

        points, gt_classes = data
        points = points.to(device)
 
        with torch.no_grad():
            reconstructed_points = autoencoder.eval()(points)
            reconstructed_points = reconstructed_points.transpose(1,2)
            loss_chamfer, _ = chamfer_distance(points, reconstructed_points)
            test_loss = loss_chamfer
            
        
        epoch_str = ''
        total_loss += test_loss * args.batch_size
        pbar.set_description('{} {} Loss: {:f}'.format(
        epoch_str, mode, test_loss))

        pbar.update(args.batch_size)

    pbar.close()       
    mean_loss = total_loss / float(n_data) 

    return mean_loss



if __name__ == "__main__":
    
    print(args)

    # Model loading
    in_dim = 3
    num_points = 2048
    
    if args.model_type =='hankyu':
        autoencoder = PointNetAE(in_dim, num_points)
    elif args.model_type == 'hankyu1':
        autoencoder = PointNetAE1(in_dim, num_points)

    if args.model != '':
        autoencoder.load_state_dict(torch.load(args.model))

    autoencoder.to(device)

    # Create instance of SummaryWriter
    writer = SummaryWriter('runs/' + args.model_type)

    # Create dataset and data loader
    train_dataset, train_dataloader ,test_dataset, test_dataloader, n_classes = create_datasets_and_dataloaders(args)


    # Setting up an optimizer and a scheduler
    optimizer = torch.optim.Adam(
        autoencoder.parameters(), lr=args.learning_rate, 
        betas=(args.beta1, args.beta2))

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size = args.step_size, gamma = args.gamma
    )


    # Create the ouput directory
    os.makedirs('saved_models', exist_ok=True)
    
    best_loss = 1e20
    
    if args.train:
        for epoch in range(args.n_epochs):
      
            epoch_loss = run_train(epoch, autoencoder, train_dataset, train_dataloader, args, writer)
            mean_loss = run_test( autoencoder, test_dataset, test_dataloader, args, writer) 
            epoch_str = '' if epoch is None else '[Epoch {}/{}]'.format(
                str(epoch).zfill(len(str(args.n_epochs))), args.n_epochs)

            log = epoch_str + ' '
            log += 'Train Loss: {:f}, '.format(epoch_loss)

            log += 'Test Loss: {:f}, '.format(mean_loss)
            print(log)
            if (epoch + 1) % 10 == 0:
                model_file = os.path.join(
                       'saved_models', 'autoencoder_{:d}.pth'.format(epoch + 1))
                torch.save(autoencoder.state_dict(), model_file)
                print("Saved '{}'.".format(model_file))
                mean_loss = run_test( autoencoder, test_dataset, test_dataloader, args, writer) 
                epoch_str = ''
                log = epoch_str + 'Test Loss: {:f}, '.format(mean_loss)
                print(log)
            
            scheduler.step()
            
        writer.close()
    else:
        mean_loss = run_test( autoencoder, test_dataset, test_dataloader, args, writer)
        log = 'Test Loss: {:f}, '.format(mean_loss)
        print(log)


