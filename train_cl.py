from utils import VoxelDataset
from cl_orion import Encoder
from cl_orion import HeadContrastive
from cl_orion import NTXentLoss
from utils import RandomRotateZ
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import os
import sys
from time import time
from tqdm import tqdm
import numpy as np


def finish_training(train_loss_log, test_loss_log, best_enc_state, best_head_state, params_dir, best_accuracy):
    # Plot losses
    plt.figure(figsize=(12, 8))
    plt.semilogy(train_loss_log, label='Train loss')
    plt.semilogy(test_loss_log, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(params_dir + 'loss_overtime.png', dpi=300)

    ### Save the network state
    # The state dictionary includes all the parameters of the network

    torch.save(best_enc_state, params_dir + 'encoder_parameters.torch')
    torch.save(best_head_state, params_dir + 'head_parameters.torch')
    
    print("Parameters saved at " + params_dir + 'net_parameters.torch')
    print(f"Best validation accuracy: {best_accuracy}")
    sys.exit()


if __name__ =="__main__":
    csv_path  = "../ModelNet10/new_metadata_modelnet10.csv"  # or wherever you put it
    data_root = "../ModelNet10"        # root folder for the folders in object_path
    # Parameters
    RNG_seed = 1
    ##############Change these
    params_dir = '../ModelNet10/'
    # Enable cuDNN to select the fastest convolution algorithms for your hardware
    torch.backends.cudnn.benchmark = True

    params = {
        'num_epochs': 1501,
        'learning_rate': 1e-3,
        'batch_size': 200,
        'optimizer': torch.optim.Adam,
        'grid_size': 28,
        'num_orientations': 4,
        }
    print(f"Model parameters: {params}")
    num_epochs = params['num_epochs']
    train_batch_size = params['batch_size']
    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "No CUDA device available")
    num_print = 10
    # build a single transform pipeline:
    rotation = RandomRotateZ(num_orientations=4, augmentation=True)


    # Create train & test sets, sharing the same class→idx map:
    train_ds = VoxelDataset(csv_path, data_root, split="train", transform=rotation)
    print(f"Dataset length: {len(train_ds)}")
    test_ds  = VoxelDataset(csv_path, data_root,
                            split="test",
                            class_to_idx=train_ds.class_to_idx, transform=rotation)
    
    num_classes = len(train_ds.class_to_idx)
    print(f"Found {num_classes} classes: {train_ds.class_to_idx}")
    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True,
                        num_workers=os.cpu_count(),        # spawn 3 worker processes
                        pin_memory=True,      # helps when transferring to GPU
                        prefetch_factor=4)     # how many batches each worker preloads)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False, num_workers=0)     # how many batches each worker preloads)
    # Define the loss function
    encoder = Encoder()
    encoder.to(device)
    transform_head = HeadContrastive(input_dim=8000)
    transform_head.to(device)
    # Define the optimizer
    optimizer_enc = params['optimizer'](encoder.parameters(), lr=params['learning_rate'])
    optimizer_head = params['optimizer'](transform_head.parameters(), lr=params['learning_rate'])

    loss_fn = NTXentLoss(0.5)

    file_name = params_dir + 'params.txt'
    with open(file_name, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}={value}\n")
    print("Loading data from " + csv_path)

    # Initialize the network
    torch.manual_seed(RNG_seed)
    print("Starting learning...")
    train_loss_log = []
    test_loss_log = []
    start_time = time()

    patience = 200  # Number of epochs to wait for improvement
    lr_reduction_milestone = 75
    min_val_loss = 20  # Min loss after which early stopping occurs
    best_loss = float('inf')
    epochs_no_improve = 0
    best_enc_state = None
    best_head_state = None
    try:
        for epoch_num in range(num_epochs):

            ### TRAIN
            train_loss = []
            encoder.train()  # Training mode (e.g. enable dropout, batchnorm updates,...)
            transform_head.train()
            for sample_batched in tqdm(train_loader,desc=f"Epoch {epoch_num+1}/{num_epochs} - train", leave=False):
                # Move data to device
                query_voxels = sample_batched['voxel_rotated1'].unsqueeze(1).to(device)     # shape (B,1,28,28,28)
                
                pos_voxels = sample_batched['voxel_rotated2'].unsqueeze(1).to(device)

                query_representations = transform_head(encoder(query_voxels))
                pos_representations = transform_head(encoder(pos_voxels))
                loss = loss_fn(query_representations, pos_representations)
                encoder.zero_grad()
                transform_head.zero_grad()
                
                ###########################################

                # Backpropagation
                loss.backward()

                # Update the weights
                optimizer_enc.step()
                optimizer_head.step()
                ###########################################

                # Save train loss for this batch
                loss_batch = loss.detach().cpu().numpy()
                train_loss.append(loss_batch)

            # Save average train loss
            train_loss = np.mean(train_loss)
            train_loss_log.append(train_loss)

            ### VALIDATION
            test_loss = []
            test_loss_unscaled = []  # List to store unscaled test losses
            accuracy = []
            encoder.eval()  # Evaluation mode (e.g. disable dropout, batchnorm,...)
            transform_head.eval()
            with torch.no_grad():  # Disable gradient tracking
                for sample_batched in test_loader:
                    # Move data to device
                    
                    query_voxels = sample_batched['voxel_rotated1'].unsqueeze(1).to(device)     # shape (B,1,28,28,28)
                
                    pos_voxels = sample_batched['voxel_rotated2'].unsqueeze(1).to(device)

                    query_representations = transform_head(encoder(query_voxels))
                    pos_representations = transform_head(encoder(pos_voxels))

                    loss = loss_fn(query_representations, pos_representations)
                    ###########################################
                    # Save val loss for this batch (scaled)
                    loss_batch = loss.detach().cpu().numpy()
                    test_loss.append(loss_batch)

                # Save average validation loss (scaled)
                test_loss = np.mean(test_loss)
                test_loss_log.append(test_loss)

                # Early stopping logic
                if test_loss < best_loss:
                    best_loss = test_loss
                    epochs_no_improve = 0
                    best_enc_state = encoder.state_dict()  # Save best model
                    best_head_state = transform_head.state_dict()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == lr_reduction_milestone:
                        print("Reducing learning rate by 10x")
                        for pg in optimizer_enc.param_groups:
                            pg['lr'] *= 0.1
                        for pg in optimizer_head.param_groups:
                            pg['lr'] *= 0.1
                    if epochs_no_improve >= patience:
                        print(f"Early stopping at epoch {epoch_num}. Best validation accuracy: {accuracy}")
                        break

                if epoch_num % num_print == 0:
                    print('#################')
                    print(f'# EPOCH {epoch_num}')
                    print('#################')
                    print(f"AVERAGE TRAIN LOSS: {train_loss}")
                    print(f"AVERAGE VAL LOSS: {test_loss}")
                    print(f"EPOCHS WITHOUT IMPROVEMENT {epochs_no_improve}")
                    print('#################')
                    print(f"ELAPSED TIME: {time() - start_time}")
    except KeyboardInterrupt:
        print("\n⏸ Training interrupted by user. Saving current state…")
        finish_training(train_loss_log, test_loss_log, best_enc_state, best_head_state, params_dir, best_loss)
        

    finish_training(train_loss_log, test_loss_log, best_enc_state, best_head_state, params_dir, best_loss)