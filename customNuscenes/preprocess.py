from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix
import numpy as np
import pandas as pandas
from tqdm import tqdm, tqdm_notebook
import os

ARTIFACTS_FOLDER = './lyft_custom_artifacts'
DATA_PATH = '/kaggle/input/3d-object-detection-for-autonomous-vehicles'
level5data = LyftDataset(data_path=DATA_PATH, json_path= os.path.join(DATA_PATH, 'train_data'))
os.makedirs(ARTIFACTS_FOLDER, exist_ok = True)
classes = ['car', 'motorcycle', 'bus', 'bicycle', 'truck', 'pedestrian', 'other_vehicle', 'animal', 'emergency_vehicle']

print(level5data.sample)

def remove_unwanted_class(target_class):
    pass
