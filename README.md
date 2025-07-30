# 3D object recognition
***Software overview***

train_baseline.py is the script to train the ORION network

train_cl.py is the script to train both the encoder and the transform head using the NT-Xent loss

transfer_learning.py is the script to train the classification head using the learned encoder

In utils.py you can find all the transformations to augment the data, the functions to manage datasets and the ORION net architecture
In cl_utils.py there are the encoder and transform heads definitions and the NT-Xent loss definition.

save_voxel.py is the script that takes the ModelNet10 dataset with .off files and saves them as 28x28x28 pytorch tensors grids

test.py is the script where the models can be tested and obtain the confusion matrices and various metrics

visualize.py samples each iteration a voxel and visualize it, then applyes cropping and rotation and visualizes a positive pair

The ModelNet10 dataset can be downloaded at https://www.kaggle.com/datasets/balraj98/modelnet10-princeton-3d-object-dataset

***Models***

In the baseline_ORION folder the trained ORION model parameters are saved

In the CL_RESULTS folder the encoder and the contrastive head are saved

In the FINAL_CLASSIFICATION_HEAD the classification transform head is saved
