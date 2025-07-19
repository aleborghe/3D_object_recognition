from dataset import ModelNetDataset
from torch.utils.data import DataLoader

csv_path  = "../ModelNet10/metadata_modelnet10.csv"  # or wherever you put it
data_root = "../ModelNet10"        # root folder for the folders in object_path

# Create train & test sets, sharing the same class→idx map:
train_ds = ModelNetDataset(csv_path, data_root, split="train")
test_ds  = ModelNetDataset(csv_path, data_root,
                           split="test",
                           class_to_idx=train_ds.class_to_idx)

# Quick DataLoader sanity check:
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
batch = next(iter(train_loader))
print(batch["vertices"], batch["faces"], batch["label"], batch["label_class"])