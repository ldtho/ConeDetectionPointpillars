import os
from tqdm import tqdm, tqdm_notebook
import numpy as np
import pandas as pandas
import sys

sys.path.append('/kaggle/code/ConeDetectionPointpillars')
import fire
import pickle
import json
from lyft_dataset_sdk.utils.geometry_utils import quaternion_yaw
from second.data.dataset import Dataset, register_dataset, get_dataset_class
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box, Quaternion
from nuscenes.utils.geometry_utils import transform_matrix, view_points
from second.core import box_np_ops
from pathlib import Path
import subprocess
from nuscenes.nuscenes import NuScenesExplorer
VERSION = 'trainval'
TRAINVAL_SPLIT_PERCENTAGE = 0.99 if VERSION == 'trainval' else 0.8
MIN_CONES_PER_SAMPLE = 8
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

    def __init__(self, root_path=f'/media/starlet/LdTho/data/sets/nuscenes/v1.0-{VERSION}', info_path=None,
                 class_names=["traffic_cone"], prep_func=None,
                 num_point_features=None):
        self.NumPointFeatures = 5
        self.class_names = class_names
        self.nusc = NuScenes(dataroot=root_path, version=f'v1.0-{VERSION}')
        self._prep_func = prep_func
        self.filtered_sample_tokens = []
        for sample in self.nusc.sample:
            sample_token = sample['token']
            sample_lidar_token = sample['data']['LIDAR_TOP']
            boxes = self.nusc.get_boxes(sample_lidar_token)
            box_names = [NameMapping[b.name] for b in boxes if b.name in NameMapping.keys()]
            for box in boxes:
                if box.name not in NameMapping.keys():
                    continue
                # if NameMapping[box.name] in self.class_names:
                if (NameMapping[box.name] in ["traffic_cone"]) & (
                        box_names.count('traffic_cone') > MIN_CONES_PER_SAMPLE):
                    self.filtered_sample_tokens.append(sample_token)
                    break
        self.filtered_sample_tokens = self.filtered_sample_tokens[
                                      :round(len(self.filtered_sample_tokens) * TRAINVAL_SPLIT_PERCENTAGE)]

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

    def get_sensor_data(self, query, token=None):
        res = {
            'lidar': {
                'type': 'lidar',
                'points': None,
            },
            'metadata': {
                'token': self.filtered_sample_tokens[query]
            }
        }
        if token:
            query = self.filtered_sample_tokens.index(token)
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

    def __init__(self, root_path=f'/media/starlet/LdTho/data/sets/nuscenes/v1.0-{VERSION}',
                 info_path=None,
                 class_names=['traffic_cone'],
                 prep_func=None,
                 num_point_features=None,
                 multi_test=False):
        print(root_path)
        self.nusc = NuScenes(dataroot=root_path, version=f'v1.0-{VERSION}')
        self.class_names = class_names
        self._prep_func = prep_func
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
        self.filtered_sample_tokens = self.filtered_sample_tokens[
                                      round(len(self.filtered_sample_tokens) * TRAINVAL_SPLIT_PERCENTAGE):]
        self.split = np.arange(len(self.filtered_sample_tokens))
        self.num_samples = len(self.filtered_sample_tokens)
        self.rot = 0.0
        self.scale = 1.0

    def __len__(self):
        return self.num_samples

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

    def evaluation(self, detections, output_dir):
        res_custom_nusc = self.evaluation_custom_nusc(detections, output_dir)
        res = {
            "results": {
                "nusc": res_custom_nusc["results"]["nusc"],
            },
            "details": {
                "eval.nusc": res_custom_nusc["detail"]["nusc"]
            }
        }
        return res

    def evaluation_custom_nusc(self, detections, output_dir):
        pass


# should shorten by inherit from parent class CustomNuscDataset
EVAL_VERSION = 'mini'


@register_dataset
class CustomNuscEvalDataset(Dataset):
    NumPointFeatures = 5

    def __init__(self, root_path=f'/media/starlet/LdTho/data/sets/nuscenes/v1.0-{EVAL_VERSION}', info_path=None,
                 class_names=["traffic_cone"], prep_func=None,
                 num_point_features=None):
        self.NumPointFeatures = 5
        self.class_names = class_names
        self.nusc = NuScenes(dataroot=root_path, version=f'v1.0-{EVAL_VERSION}')
        self._prep_func = prep_func
        self.root_path = root_path
        self.eval_version = "detection_cvpr_2019"
        # self.filtered_sample_tokens = [s['token'] for s in self.nusc.sample]

        self.filtered_sample_tokens = []
        for sample in self.nusc.sample:
            sample_token = sample['token']
            sample_lidar_token = sample['data']['LIDAR_TOP']
            boxes = self.nusc.get_boxes(sample_lidar_token)
            box_names = [NameMapping[b.name] for b in boxes if b.name in NameMapping.keys()]
            for box in boxes:
                if box.name not in NameMapping.keys():
                    continue
                # if NameMapping[box.name] in self.class_names:
                if (NameMapping[box.name] in ["traffic_cone"]) & (
                        box_names.count('traffic_cone') > MIN_CONES_PER_SAMPLE):
                    self.filtered_sample_tokens.append(sample_token)
                    break
        if EVAL_VERSION == "trainval":
            self.filtered_sample_tokens = self.filtered_sample_tokens[
                                          round(len(self.filtered_sample_tokens) * TRAINVAL_SPLIT_PERCENTAGE):]

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

    def get_sensor_data(self, query, token=None):
        res = {
            'lidar': {
                'type': 'lidar',
                'points': None,
            },
            'metadata': {
                'token': self.filtered_sample_tokens[query]
            }
        }
        if token:
            query = self.filtered_sample_tokens.index(token)
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

    def move_boxes_to_car_space(self, boxes, ego_pose, eval = False):
        """
        Move boxes from world space to car space.
        Note: mutates input boxes.
        """
        translation = -np.array(ego_pose['translation'])
        rotation = Quaternion(ego_pose['rotation']).inverse
        box_list = []
        for box in boxes:
            # Bring box to car space
            box.translate(translation)
            box.rotate(rotation)
            box_list.append(box)
        return box_list

    def evaluation(self, detections, output_dir):
        """kitti evaluation is very slow, remove it.
        """
        # res_kitti = self.evaluation_kitti(detections, output_dir)
        res_nusc = self.evaluation_nusc(detections, output_dir)
        res = {
            "results": {
                "nusc": res_nusc["results"]["nusc"],
                # "kitti.official": res_kitti["results"]["official"],
                # "kitti.coco": res_kitti["results"]["coco"],
            },
            "detail": {
                "eval.nusc": res_nusc["detail"]["nusc"],
                # "eval.kitti": {
                #     "official": res_kitti["detail"]["official"],
                #     "coco": res_kitti["detail"]["coco"],
                # },
            },
        }
        return res

    def evaluation_nusc(self, detections, output_dir):
        mapped_class_names = self.class_names
        nusc_annos = {}
        token2info = {}
        for det in detections:
            annos = []
            boxes = _second_det_to_nusc_box(det)
            boxes = self.transform_box_back_to_global(boxes, det["metadata"]["token"])

            # boxes = self.move_boxes_to_car_space(boxes, ego_pose)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                velocity = [np.nan, np.nan]
                nusc_anno = {
                    "sample_token": det["metadata"]["token"],
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": velocity,
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": DefaultAttribute[name]
                }
                annos.append(nusc_anno)
            nusc_annos[det["metadata"]["token"]] = annos
        nusc_submissions = {
            "meta": {
                "use_camera": False,
                "use_lidar": False,
                "use_radar": False,
                "use_map": False,
                "use_external": False
            },
            "results": nusc_annos
        }
        res_path = Path(output_dir) / "result_nusc.json"
        with open(res_path, "w") as f:
            json.dump(nusc_submissions, f)
        eval_main_file = Path(__file__).resolve().parent / "nusc_eval.py"
        cmd = f"python {str(eval_main_file)} --root_path=\"{self.root_path}\""
        cmd += f" --version=\"v1.0-mini\" --eval_version={self.eval_version}"
        cmd += f" --res_path=\"{str(res_path)}\" --eval_set=mini_train"
        cmd += f" --output_dir=\"{output_dir}\""
        print(cmd)
        subprocess.check_output(cmd, shell=True)
        with open(Path(output_dir) / "metrics_summary.json", "r") as f:
            metrics = json.load(f)
        detail = {}
        res_path.unlink()
        result = f"Nusc {VERSION} evaluation"
        for name in mapped_class_names:
            detail[name] = {}
            for k, v in metrics["label_aps"][name].items():
                detail[name][f"dist@{k}"] = v
            tp_errs = []
            tp_names = []
            for k, v in metrics["label_tp_errors"][name].items():
                detail[name][k] = v
                tp_errs.append(f"{v:.4f}")
                tp_names.append(k)
            threshs = ', '.join(list(metrics["label_aps"][name].keys()))
            scores = list(metrics["label_aps"][name].values())
            scores = ', '.join([f"{s * 100:.2f}" for s in scores])
            result += f"{name} Nusc dist AP@{threshs} and TP errors\n"
            result += scores
            result += ', '.join(tp_names) + ": " + ", ".join(tp_errs)
            result += "\n"
        return {
            "results": {
                "nusc": result,
            },
            "detail": {
                "nusc": detail
            }
        }

    def transform_box_back_to_global(self, boxes, token):
        sample = self.nusc.get('sample', token)
        sample_lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', sample_lidar_token)
        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])

        translation = np.array(ego_pose['translation'])
        rotation = Quaternion(ego_pose['rotation'])
        box_list = []
        for box in boxes:
            box.rotate(rotation)
            box.translate(translation)
            box_list.append(box)
        return box_list


def _second_det_to_nusc_box(detection):
    from nuscenes.utils.data_classes import Box
    import pyquaternion
    box3d = detection["box3d_lidar"].detach().cpu().numpy()
    scores = detection["scores"].detach().cpu().numpy()
    labels = detection["label_preds"].detach().cpu().numpy()
    box3d[:, 6] = -box3d[:, 6] - np.pi / 2
    box_list = []
    for i in range(box3d.shape[0]):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box3d[i, 6])
        velocity = (np.nan, np.nan, np.nan)
        if box3d.shape[1] == 9:
            velocity = (*box3d[i, 7:9], 0.0)
            # velo_val = np.linalg.norm(box3d[i, 7:9])
            # velo_ori = box3d[i, 6]
            # velocity = (velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = Box(
            box3d[i, :3],
            box3d[i, 3:6],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list

if __name__ == '__main__':
    fire.Fire()
    # train_data = CustomNuscDataset()
    # test_data = CustomNuscTestDataset(root_path='/media/starlet/LdTho/data/sets/nuscenes/v1.0-trainval',
    #                                   )
#     print(train_data[1])
