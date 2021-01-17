import sys
import os
from tqdm import tqdm, tqdm_notebook
import numpy as np
import pandas as pandas
sys.path.append('/kaggle/code/lyft-tho')
import fire
import pickle
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix, quaternion_yaw
from second.data.dataset import Dataset, register_dataset,get_dataset_class
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box, Quaternion
from nuscenes.utils.geometry_utils import *

VERSION = 'trainval'
NUSC_DATASET_ROOT = f'/media/starlet/LdTho/data/sets/nuscenes/v1.0-{VERSION}'
OBJECT_CLASSES = ['animal',
                  'human.pedestrian.adult',
                  'human.pedestrian.child',
                  'human.pedestrian.construction_worker',
                  'human.pedestrian.personal_mobility',
                  'human.pedestrian.police_officer',
                  'human.pedestrian.stroller',
                  'human.pedestrian.wheelchair',
                  'movable_object.barrier',
                  'movable_object.debris',
                  'movable_object.pushable_pullable',
                  'movable_object.trafficcone',
                  'static_object.bicycle_rack',
                  'vehicle.bicycle',
                  'vehicle.bus.bendy',
                  'vehicle.bus.rigid',
                  'vehicle.car',
                  'vehicle.construction',
                  'vehicle.emergency.ambulance',
                  'vehicle.emergency.police',
                  'vehicle.motorcycle',
                  'vehicle.trailer',
                  'vehicle.truck']

@register_dataset
class CustomNuscDataset(Dataset):
    NumPointFeatures = 5

    def __init__(self, root_path=NUSC_DATASET_ROOT, info_path=None,
                 class_names=["movable_object.trafficcone"], prep_func=None,
                 num_point_features=None):

        data_dir = root_path
        json_dir = os.path.join(root_path, f'v1.0-{VERSION}')
        print(json_dir)
        self.class_names = class_names
        self.nusc = NuScenes(dataroot=data_dir,version=f'v1.0-{VERSION}')
        self._prep_func = prep_func
        self.box_classes = set()
        self.filtered_sample_tokens = []
        for sample in self.nusc.sample:
            sample_token = sample['token']
            sample_lidar_token = sample['data']['LIDAR_TOP']
            boxes = self.nusc.get_boxes(sample_lidar_token)
            for box in boxes:
                self.box_classes.add(box.name)
                if box.name in self.class_names:
                    self.filtered_sample_tokens.append(sample_token)
                    break

        self.split = np.arange(len(self.filtered_sample_tokens))

    def __len__(self):
        return self.split.shape[0]

    def __getitem__(self, index):
        input_dict = self.get_sensor_data(index)
        try:
            example = self._prep_func(input_dict=input_dict)
            return example
        except:
            return input_dict
    def __name__(self):
        return "CustomNuscDataset"
    def get_sensor_data(self, query):
        res = {
            'lidar': {
                'type': 'lidar',
                'points': None,
            },
            'metadata': {
                'token': self.filtered_sample_tokens[query]
            }
        }
        points = self.getPoints(query)
        boxes_dict = self.getBoxes(query)

        res['lidar']['points'] = points

        gt_boxes = []
        gt_names = []

        for box in boxes_dict:
            xyz = box.center
            wlh = box.wlh
            theta = quaternion_yaw(box.orientation)
            gt_boxes.append([xyz[0], xyz[1], xyz[2], wlh[0], wlh[1], wlh[2], -theta - np.pi / 2])
            gt_names.append(box.name)
            print(box.name)

        res['lidar']['annotations'] = {
            'boxes': np.asarray(gt_boxes,dtype=np.float32),
            'names': gt_names,
        }
        return res

        ###

    def getPoints(self, index):
        sample = self.nusc.get('sample', self.filtered_sample_tokens[index])
        sample_lidar_token = sample['data']['LIDAR_TOP']

        lidar_data = self.nusc.get('sample_data', sample_lidar_token)
        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        calibrated_sensor = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

        global_from_car = transform_matrix(ego_pose['translation'],
                                           Quaternion(ego_pose['rotation']), inverse=False)
        car_from_sensor = transform_matrix(calibrated_sensor['translation'],
                                           Quaternion(calibrated_sensor['rotation']), inverse=False)
        try:
            lidar_pointcloud, times = LidarPointCloud.from_file_multisweep(self.nusc, sample, 'LIDAR_TOP',
                                                                           'LIDAR_TOP')
            lidar_pointcloud.transform(car_from_sensor)
        except Exception as e:
            print(f"Failed to load Lidar Pointcloud for {sample}:{e}")
        points = lidar_pointcloud.points
        points[3, :] /= 255
        points[3, :] -= 0.5

        points_cat = np.concatenate([points, times], axis=0).transpose()
        points_cat = points_cat[~np.isnan(points_cat).any(axis=1)]

        return points_cat

    def getBoxes(self, index):

        sample = self.nusc.get('sample', self.filtered_sample_tokens[index])
        sample_lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', sample_lidar_token)
        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])

        boxes_dict = self.nusc.get_boxes(sample_lidar_token)

        keep_box_idx = []
        for i, box in enumerate(boxes_dict):
            if box.name in self.class_names:
                keep_box_idx.append(i)

        boxes_dict = [box for i, box in enumerate(boxes_dict) if i in keep_box_idx]

        return boxes_dict

    def move_boxes_to_car_space(self,boxes, ego_pose):
        """
        Move boxes from world space to car space.
        Note: mutates input boxes.
        """
        translation = -np.array(ego_pose['translation'])
        rotation = Quaternion(ego_pose['rotation']).inverse

        for box in boxes:
            # Bring box to car space
            box.translate(translation)
            box.rotate(rotation)



if __name__ == '__main__':
    fire.Fire()
    # train_data = CustomNuscDataset()
#     print(train_data[1])
