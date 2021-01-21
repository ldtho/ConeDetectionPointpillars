import sys
import os
from tqdm import tqdm, tqdm_notebook
import numpy as np
import pandas as pandas

sys.path.append('/kaggle/code/ConeDetectionPointpillars')
import fire
import pickle
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix, quaternion_yaw
from second.data.dataset import Dataset, register_dataset, get_dataset_class
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box, Quaternion
from nuscenes.utils.geometry_utils import *
from second.core import box_np_ops

VERSION = 'trainval'
NUSC_DATASET_ROOT = f'/media/starlet/LdTho/data/sets/nuscenes/v1.0-{VERSION}'
NameMapping = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck',
    'movable_object.pushable_pullable': 'DontCare',
    'movable_object.debris': 'DontCare'
}
DefaultAttribute = {
    "car": "vehicle.parked",
    "pedestrian": "pedestrian.moving",
    "trailer": "vehicle.parked",
    "truck": "vehicle.parked",
    "bus": "vehicle.parked",
    "motorcycle": "cycle.without_rider",
    "construction_vehicle": "vehicle.parked",
    "bicycle": "cycle.without_rider",
    "barrier": "",
    "traffic_cone": "",
}

@register_dataset
class CustomNuscDataset(Dataset):
    NumPointFeatures = 5
    def __init__(self, root_path=NUSC_DATASET_ROOT, info_path=None,
                 class_names=["traffic_cone"], prep_func=None,
                 num_point_features=None):
        self.NumPointFeatures = 5
        self.class_names = class_names
        self.nusc = NuScenes(dataroot=root_path, version=f'v1.0-{VERSION}')
        self._prep_func = prep_func
        # self.box_classes = set()
        self.filtered_sample_tokens = []
        for sample in self.nusc.sample:
            sample_token = sample['token']
            sample_lidar_token = sample['data']['LIDAR_TOP']
            boxes = self.nusc.get_boxes(sample_lidar_token)
            for box in boxes:
                # self.box_classes.add(box.name
                if box.name not in NameMapping.keys():
                    continue
                if NameMapping[box.name] in self.class_names:
                    self.filtered_sample_tokens.append(sample_token)
                    break

        self.split = np.arange(len(self.filtered_sample_tokens))

    def __len__(self):
        return self.split.shape[0]

    def __getitem__(self, index):
        input_dict = self.get_sensor_data(index)
        example = self._prep_func(input_dict=input_dict)
        example["metadata"] = input_dict["metadata"]
        if "anchors_mask" in example:
            example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        return example

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
        gt_boxes = np.concatenate(gt_boxes).reshape(-1, 7)
        gt_names = np.array(gt_names)
        res['lidar']['annotations'] = {
            'boxes': gt_boxes,
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
            if box.name not in NameMapping.keys():
                continue
            if NameMapping[box.name] in self.class_names:
                box.name = NameMapping[box.name]
                keep_box_idx.append(i)

        boxes_dict = [box for i, box in enumerate(boxes_dict) if i in keep_box_idx]
        self.move_boxes_to_car_space(boxes_dict, ego_pose)
        # print(boxes_dict)
        return boxes_dict

    def move_boxes_to_car_space(self, boxes, ego_pose):
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


@register_dataset
class CustomNuscTestDataset(Dataset):
    NumPointFeatures = 5

    def __init__(self, root_path=NUSC_DATASET_ROOT,
                 info_path=None,
                 class_names=['traffic_cone'],
                 prep_func=None,
                 num_point_features=None,
                 multi_test=False):
        self.nusc = NuScenes(dataroot = root_path, version=f'v1.0-{VERSION}')
        self.class_names = class_names
        self.prep_func = prep_func
        self.filtered_sample_tokens = []
        self.multi_test = multi_test
        for sample in self.nusc.sample:
            sample_token = sample['token']
            sample_lidar_token = sample['data']['LIDAR_TOP']
            boxes = self.nusc.get_boxes(sample_lidar_token)
            for box in boxes:
                # self.box_classes.add(box.name
                if box.name not in NameMapping.keys():
                    continue
                if NameMapping[box.name] in self.class_names:
                    self.filtered_sample_tokens.append(sample_token)
                    break

        self.split = np.arange(len(self.filtered_sample_tokens))
        self.num_samples = len(self.filtered_sample_tokens)
        self.rot = 0.0
        self.scale = 1.0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        input_dict = self.get_sensor_data(index)
        input_dict['lidar']['points'][:, :3] = box_np_ops.rotation_points_single_angle(
            input_dict['lidar']['points'][:, :3],
            self.rot,
            axis=2)
        if not self.multi_test:
            example = self.prep_func(input_dict=input_dict)
        else:
            points = input_dict['lidar']['points']
            points_x = np.copy(points)
            points_y = np.copy(points)
            points_xy = np.copy(points)

            points_x[:, 0] *= -1
            points_y[:, 1] *= -1
            points_xy[:, 0:2] *= -1
            example = []

            input_dict['lidar']['points'] = points
            example.append(self.prep_func(input_dict))

            input_dict['lidar']['points'] = points_x
            example.append(self.prep_func(input_dict))

            input_dict['lidar']['points'] = points_y
            example.append(self.prep_func(input_dict))

            input_dict['lidar']['points'] = points_xy
            example.append(self.prep_func(input_dict))

        return example

    ### to be continue

    def get_sensor_data(self, query):
        res = {
            'lidar': {
                'type': 'lidar',
                'points': None
            },
            'metadata': {
                'token': self.filtered_sample_tokens[query],
            }
        }
        points = self.getPoints(query)
        res['lidar']['points'] = points
        return res

    def getPoints(self, query):
        sample = self.nusc.get('sample', self.filtered_sample_tokens[query])
        sample_lidar_token = sample['data']['LIDAR_TOP']

        lidar_data = self.nusc.get('sample_data', sample_lidar_token)
        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        calibrated_sensor = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])

        global_from_car = transform_matrix(ego_pose['translation'],
                                           Quaternion(ego_pose['rotation']), inverse=False)
        car_from_sensor = transform_matrix(calibrated_sensor['translation'],
                                           Quaternion(calibrated_sensor['rotation']), inverse=False)
        try:
            lidar_pointcloud, times = LidarPointCloud.from_file_multisweep(self.nusc, sample, 'LIDAR_TOP', 'LIDAR_TOP')
            lidar_pointcloud.transform(car_from_sensor)
        except Exception as e:
            print(f"failed to load pointcloud for {sample}: {e}")
        points = lidar_pointcloud.points
        points[3, :] /= 255
        points[3, :] -= 0.5
        points_cat = np.concatenate([points, times], axis=0).transpose()
        points_cat = points_cat[~np.isnan(points_cat).any(axis=1)]
        return points_cat


if __name__ == '__main__':
    fire.Fire()
    # train_data = CustomNuscDataset()
    # test_data = CustomNuscTestDataset(root_path='/media/starlet/LdTho/data/sets/nuscenes/v1.0-trainval',
    #                                   )
#     print(train_data[1])
