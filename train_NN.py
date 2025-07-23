import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from time import time
from utils.dataset import PandaDataset
from utils.neural_network import StudentNN


# Parameters
RNG_seed = 1
##############Change these
file_dir = '/home/borgherini/Desktop/data/logs/'
params_dir = '/home/borgherini/Desktop/data/models/'
if not os.path.exists(params_dir):
	os.makedirs(params_dir)
# Enable cuDNN to select the fastest convolution algorithms for your hardware
torch.backends.cudnn.benchmark = True

# Load the dataset
dataset = PandaDataset(file_dir + 'controller_log.csv')

print(f"Dataset length: {len(dataset)}")

params = {
	'num_epochs': 2001,
	'learning_rate': 1e-3,
	'batch_size': 2000,
	'optimizer': torch.optim.Adam,
	'Ni': len(dataset.input_indices),
	'No': 6,
	'Nh1': 128,
	'Nh2': 128,
	'activation': torch.nn.LeakyReLU(),
	'loss': torch.nn.MSELoss(),
	'p_dropout': 0.2,
	'normalize': False,
	'with_wrench': dataset.wrench_used,
	}
print(f"Model parameters: {params}")
num_epochs = params['num_epochs']
train_batch_size = params['batch_size']
train_percentage = 0.7
# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(torch.cuda.get_device_name() if torch.cuda.is_available() else "No CUDA device available")

num_print = 10
# Saving hyperparameters on a txt file
#act_matrix = np.load(file_dir + 'act_matrix.npy')
#np.save(params_dir + 'act_matrix.npy', act_matrix)

file_name = params_dir + 'params.txt'
with open(file_name, 'w') as f:
	for key, value in params.items():
		f.write(f"{key}={value}\n")
print("Loading data from " + file_dir + "controller_log.csv")

# Initialize the network
torch.manual_seed(RNG_seed)

net = StudentNN(params['Ni'], params['Nh1'], params['Nh2'], params['No'], params['activation'], params['p_dropout'], params['normalize'])
net.to(device)
if preTrained_dir is not None:
	print("Loading pre-trained model from " + preTrained_dir)
	net.load_state_dict(torch.load(preTrained_dir, map_location=device), strict=False)

# Define the loss function
loss_fn = params['loss']

# Define the optimizer
optimizer = params['optimizer'](net.parameters(), lr=params['learning_rate'])

# Split into training and testing datasets
train_size = int(train_percentage * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
"""if params['normalize']:
	dataset.normalize_data()  # Normalize the data
	input_scaler, label_scaler = dataset.get_scalers()

	# Save scalers
	torch.save(input_scaler, params_dir + 'input_scaler.torch')
	torch.save(label_scaler, params_dir + 'label_scaler.torch')"""

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
												num_workers=os.cpu_count(), prefetch_factor=4, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False,
												num_workers=0)

def finish_training(train_loss_log, test_loss_log, best_model_state, net, file_name, params_dir):
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
	if best_model_state is not None:
		torch.save(best_model_state, params_dir + 'net_parameters.torch')
	else:
		torch.save(net.state_dict(), params_dir + 'net_parameters.torch')
	print("Hyperparameters saved at " + file_name)


print("Starting learning...")
train_loss_log = []
test_loss_log = []
start_time = time()

patience = 70  # Number of epochs to wait for improvement
min_val_loss = 20  # Min loss after which early stopping occurs
best_val_loss = float('inf')
epochs_no_improve = 0
best_model_state = None
try:
	for epoch_num in range(num_epochs):

		### TRAIN
		train_loss = []
		net.train()  # Training mode (e.g. enable dropout, batchnorm updates,...)
		for sample_batched in train_dataloader:
			# Move data to device
			x_batch = sample_batched[0].to(device, non_blocking=True)
			label_batch = sample_batched[1].to(device, non_blocking=True)

			###########################################
			# Forward pass
			out = net(x_batch)

			# Compute loss
			loss = loss_fn(out, label_batch)

			# Backpropagation
			net.zero_grad()
			loss.backward()

			# Update the weights
			optimizer.step()
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
		net.eval()  # Evaluation mode (e.g. disable dropout, batchnorm,...)
		with torch.no_grad():  # Disable gradient tracking
			for sample_batched in test_dataloader:
				# Move data to device
				x_batch = sample_batched[0].to(device)
				label_batch = sample_batched[1].to(device)

				###########################################
				# Forward pass
				out = net(x_batch)

				# Compute loss on scaled data
				loss = loss_fn(out, label_batch)
				###########################################

				# Save val loss for this batch (scaled)
				loss_batch = loss.detach().cpu().numpy()
				test_loss.append(loss_batch)
				"""if params['normalize']:
					# If normalization is applied, compute unscaled loss
					# Note: Uncomment the following lines if you want to compute unscaled loss
					labels_unscaled = label_scaler.inverse_transform(label_batch.cpu().numpy())
					out_unscaled = label_scaler.inverse_transform(out.detach().cpu().numpy())

					# Compute loss on unscaled data
					loss_unscaled = loss_fn(torch.tensor(out_unscaled, device=device), torch.tensor(labels_unscaled, device=device))

					# Save val loss for this batch (unscaled)
					loss_batch_unscaled = loss_unscaled.detach().cpu().numpy()
					test_loss_unscaled.append(loss_batch_unscaled)"""

			# Save average validation loss (scaled)
			test_loss = np.mean(test_loss)
			test_loss_log.append(test_loss)

			if params['normalize']:
				# Save average validation loss (unscaled)
				test_loss_unscaled = np.mean(test_loss_unscaled)

			# Early stopping logic
			if test_loss < best_val_loss:
				best_val_loss = test_loss
				epochs_no_improve = 0
				best_model_state = net.state_dict()  # Save best model
			else:
				epochs_no_improve += 1
				if epochs_no_improve >= patience and test_loss < min_val_loss:
					print(f"Early stopping at epoch {epoch_num}. Best validation loss: {best_val_loss}")
					break

			if epoch_num % num_print == 0:
				print('#################')
				print(f'# EPOCH {epoch_num}')
				print('#################')
				print(f"AVERAGE TRAIN LOSS: {train_loss}")
				print(f"AVERAGE VAL LOSS: {test_loss}")
				print(f"EPOCHS WITHOUT IMPROVEMENT {epochs_no_improve}")
				print(f"ELAPSED TIME: {time() - start_time}")
except KeyboardInterrupt:
	print("\n⏸ Training interrupted by user. Saving current state…")
	finish_training(train_loss_log, test_loss_log, best_model_state, net, file_name, params_dir)

finish_training(train_loss_log, test_loss_log, best_model_state, net, file_name, params_dir)