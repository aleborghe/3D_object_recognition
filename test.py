import torch
import numpy as np
import matplotlib.pyplot as plt
from cl_utils import Encoder, HeadClassification
from utils import ORIONNet
from utils import VoxelDataset
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay


csv_path  = "../ModelNet10/new_metadata_modelnet10.csv"  # or wherever you put it
data_root = "../ModelNet10"
encoder_path = "CL_RESULTS/encoder_parameters.torch"
head_path = "FINAL_CLASSIFICATION_HEAD/net_parameters.torch"
orion_path = "baseline_ORION/net_parameters.torch"
train_ds = VoxelDataset(csv_path, data_root, split="train", transform=None)
test_ds = VoxelDataset(csv_path, data_root, split="test", transform=None)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

for i in range(len_train):
    voxel = train_ds[i]['voxel'].unsqueeze(0).unsqueeze(0).to(device)
    label = torch.tensor(train_ds[i]['label']).to(device)
    pred = transform_head(encoder(voxel))
    # Create a Categorical distribution using log-probabilities
    dist = torch.distributions.Categorical(logits=pred)

    # Sample one value (an integer in [0, 9])
    sample = dist.sample()
    train_accuracy.append((label==pred.argmax()).item())
print(np.array(train_accuracy).mean())

test_accuracy = []
test_labels = []
test_pred = []
for i in range(len_test):
    voxel = test_ds[i]['voxel'].unsqueeze(0).unsqueeze(0).to(device)
    label = torch.tensor(test_ds[i]['label']).to(device)
    test_labels.append(label.item())
    pred = transform_head(encoder(voxel))
    
    # Create a Categorical distribution using log-probabilities
    dist = torch.distributions.Categorical(logits=pred)

    # Sample one value (an integer in [0, 9])
    sample = dist.sample()
    test_pred.append(pred.argmax().item())
    test_accuracy.append((label==pred.argmax()).item())
print(np.array(test_accuracy).mean())


labels=['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night', 'sofa', 'table', 'toilet']

# Compute scores (macro averages across classes)
f1 = f1_score(test_labels, test_pred, average=None)
recall = recall_score(test_labels, test_pred, average=None)
precision = precision_score(test_labels, test_pred, average=None)


for i in range(len(labels)):
    print("##########################################")
    print(f"F1 Score for class {labels[i]}: {f1[i]}")
    print(f"Recall for class {labels[i]}: {recall[i]}")
    print(f"Precision for class {labels[i]}: {precision[i]}")
print("##########################################")
print(f"Average F1: {f1.mean()}")
print(f"Average Recall: {recall.mean()}")
print(f"Average precision: {precision.mean()}")

cm = confusion_matrix(test_labels, test_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)  # optional colormap
plt.title("Confusion Matrix")
plt.show()
