import torch
import numpy as np
from cl_orion import Encoder, HeadClassification
from utils import VoxelDataset

csv_path  = "../ModelNet10/new_metadata_modelnet10.csv"  # or wherever you put it
data_root = "../ModelNet10"
encoder_path = "36_contrastive/encoder_parameters.torch"
head_path = "88_after_cl/net_parameters.torch"
train_ds = VoxelDataset(csv_path, data_root, split="train", transform=None)
test_ds = VoxelDataset(csv_path, data_root, split="train", transform=None)

device = torch.device("cuda")

encoder = Encoder()
encoder.to(device)
encoder.load_state_dict(torch.load(encoder_path))
encoder.eval()
transform_head = HeadClassification(input_dim=128)
transform_head.to(device)
transform_head.load_state_dict(torch.load(head_path))
transform_head.eval()

len_train = len(train_ds)
len_test = len(test_ds)

train_accuracy = []
for i in range(10):
	voxel = train_ds[i]['voxel'].unsqueeze(0).unsqueeze(0).to(device)
	label = torch.tensor(train_ds[i]['label']).to(device)
	pred = transform_head(encoder(voxel))
	# Create a Categorical distribution using log-probabilities
	dist = torch.distributions.Categorical(logits=pred)

	# Sample one value (an integer in [0, 9])
	sample = dist.sample()
	print(label)
	print(sample)