import time
from pathlib import Path
import os
#here and solution/second/data/nuscenes_dataset
from second.core import box_np_ops
from second.data.dataset import Dataset, register_dataset
from lyft_dataset_sdk.lyftdataset import LyftDatasetExplorer, LyftDataset,LidarPointCloud,Quaternion
from lyft_dataset_sdk.utils.geometry_utils import *


LYFT_DATASET_ROOT = "/media/wenjing/ssd/lyft/"


def move_boxes_to_car_space(boxes, ego_pose):
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
class AkLyftDataset(Dataset):
    NumPointFeatures = 5  # xyz, illu, timestamp.
    def __init__(self,
                 root_path = '',
                 info_path = None, #not used only there for compatibility
                 class_names=None,
                 prep_func=None,
                 num_point_features=None):
        self.NumPointFeatures = 5
        data_dir = os.path.join(LYFT_DATASET_ROOT,"train")
        json_dir = os.path.join(data_dir, "train" + "_data")
        self.lyft = LyftDataset(data_path=data_dir, json_path=json_dir)
        self._prep_func = prep_func
        self.split = np.arange(len(self.lyft.sample))

    def __len__(self):
        return self.split.shape[0]


    def __getitem__(self, idx):

        idx = self.split[idx]
        input_dict = self.get_sensor_data(idx)
        example = self._prep_func(input_dict=input_dict)
        #example["metadata"] = input_dict["metadata"]

        return example

    def get_sensor_data(self, query):

        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },

            "metadata":
                {
                    "token":self.lyft.sample[query]['token']
                }
        }

        points = self.getPoints(query)
        boxes_dict = self.getBoxes(query)

        res["lidar"]["points"] = points

        gt_boxes = []
        gt_names = []

        for box in boxes_dict:
            xyz = box.center
            wlh = box.wlh
            theta = quaternion_yaw(box.orientation)
            gt_boxes.append([xyz[0], xyz[1], xyz[2], wlh[0], wlh[1], wlh[2], -theta - np.pi/2])
            gt_names.append(box.name)

        gt_boxes = np.concatenate(gt_boxes).reshape(-1,7)
        gt_names = np.array(gt_names)

        res["lidar"]["annotations"] = {
            'boxes': gt_boxes,
            'names': gt_names,
        }

        return res

    def getPoints(self,index):
        #index = self.split[index]
        sample_rec = self.lyft.sample[index]
        #sample_rec = self.lyft.sample[index]

        level5data = self.lyft

        sample_lidar_token = sample_rec["data"]["LIDAR_TOP"]
        lidar_data = level5data.get("sample_data", sample_lidar_token)

        ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])
        calibrated_sensor = level5data.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

        global_from_car = transform_matrix(ego_pose['translation'],
                                           Quaternion(ego_pose['rotation']), inverse=False)

        car_from_sensor = transform_matrix(calibrated_sensor['translation'], Quaternion(calibrated_sensor['rotation']),
                                           inverse=False)

        try:
            lidar_pointcloud, times = LidarPointCloud.from_file_multisweep(level5data, sample_rec, "LIDAR_TOP",
                                                                           "LIDAR_TOP",
                                                                           num_sweeps=10)
            lidar_pointcloud.transform(car_from_sensor)
        except Exception as e:
            print("Failed to load Lidar Pointcloud for {}: {}:".format(sample_rec, e))



        points = lidar_pointcloud.points  # .transpose()  # .reshape(-1, 5)

        # points = points[:3, :]
        # points[:3, :] = points
        points[3, :] /= 255
        points[3, :] -= 0.5

        points_cat = np.concatenate([points, times], axis=0).transpose()

        points_cat = points_cat[~np.isnan(points_cat).any(axis=1)]

        return points_cat

    def getBoxes(self,index):
        #index = self.split[index]
        #sample_rec = self.lyft.sample[index]
        sample_rec = self.lyft.sample[index]

        level5data = self.lyft

        sample_lidar_token = sample_rec["data"]["LIDAR_TOP"]
        lidar_data = level5data.get("sample_data", sample_lidar_token)

        ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])

        boxes_dict = level5data.get_boxes(sample_lidar_token)
        move_boxes_to_car_space(boxes_dict, ego_pose)

        return boxes_dict


class LyftTestDataset(LyftDataset):
    """Database class for Lyft Dataset to help query and retrieve information from the database."""

    def __init__(self, data_path: str, json_path: str, verbose: bool = True, map_resolution: float = 0.1):
        """Loads database and creates reverse indexes and shortcuts.
        Args:
            data_path: Path to the tables and data.
            json_path: Path to the folder with json files
            verbose: Whether to print status messages during load.
            map_resolution: Resolution of maps (meters).
        """

        self.data_path = Path(data_path).expanduser().absolute()
        self.json_path = Path(json_path)

        self.table_names = [
            "category",
            "attribute",
            "sensor",
            "calibrated_sensor",
            "ego_pose",
            "log",
            "scene",
            "sample",
            "sample_data",
            "map",
        ]

        start_time = time.time()

        # Explicitly assign tables to help the IDE determine valid class members.
        self.category = self.__load_table__("category")
        self.attribute = self.__load_table__("attribute")

        self.sensor = self.__load_table__("sensor")
        self.calibrated_sensor = self.__load_table__("calibrated_sensor")
        self.ego_pose = self.__load_table__("ego_pose")
        self.log = self.__load_table__("log")
        self.scene = self.__load_table__("scene")
        self.sample = self.__load_table__("sample")
        self.sample_data = self.__load_table__("sample_data")

        self.map = self.__load_table__("map")

        if verbose:
            for table in self.table_names:
                print("{} {},".format(len(getattr(self, table)), table))
            print("Done loading in {:.1f} seconds.\n======".format(time.time() - start_time))

        # Initialize LyftDatasetExplorer class
        self.explorer = LyftDatasetExplorer(self)
        # Make reverse indexes for common lookups.
        self.__make_reverse_index__(verbose)

    def __make_reverse_index__(self, verbose: bool) -> None:
        """De-normalizes database to create reverse indices for common cases.
        Args:
            verbose: Whether to print outputs.
        """

        start_time = time.time()
        if verbose:
            print("Reverse indexing ...")

        # Store the mapping from token to table index for each table.
        self._token2ind = dict()
        for table in self.table_names:
            self._token2ind[table] = dict()

            for ind, member in enumerate(getattr(self, table)):
                self._token2ind[table][member["token"]] = ind

        # Decorate (adds short-cut) sample_data with sensor information.
        for record in self.sample_data:
            cs_record = self.get("calibrated_sensor", record["calibrated_sensor_token"])
            sensor_record = self.get("sensor", cs_record["sensor_token"])
            record["sensor_modality"] = sensor_record["modality"]
            record["channel"] = sensor_record["channel"]

        # Reverse-index samples with sample_data and annotations.
        for record in self.sample:
            record["data"] = {}
            record["anns"] = []

        for record in self.sample_data:
            if record["is_key_frame"]:
                sample_record = self.get("sample", record["sample_token"])
                sample_record["data"][record["channel"]] = record["token"]

        if verbose:
            print("Done reverse indexing in {:.1f} seconds.\n======".format(time.time() - start_time))



@register_dataset
class AkLyftTestDataset(Dataset):
    NumPointFeatures = 5  # xyz, illu timestamp

    def __init__(self,
                 root_path = "",
                 info_path = None,
                 class_names=None,
                 prep_func=None,
                 num_point_features=None,
                 multi_test = False):

        data_dir = os.path.join(LYFT_DATASET_ROOT,"test")
        json_dir = os.path.join(data_dir, "test" + "_data")
        self.lyft = LyftTestDataset(data_path=data_dir, json_path=json_dir)
        self._prep_func = prep_func
        self.multi_test = multi_test
        self.num_samples = len(self.lyft.sample)
        self.rot = 0.0
        self.scale = 1.0


    def __len__(self):
        return self.num_samples


    def getPoints(self, index):
        sample_rec = self.lyft.sample[index]

        level5data = self.lyft

        sample_lidar_token = sample_rec["data"]["LIDAR_TOP"]
        lidar_data = level5data.get("sample_data", sample_lidar_token)

        ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])
        calibrated_sensor = level5data.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

        global_from_car = transform_matrix(ego_pose['translation'],
                                           Quaternion(ego_pose['rotation']), inverse=False)

        car_from_sensor = transform_matrix(calibrated_sensor['translation'], Quaternion(calibrated_sensor['rotation']),
                                           inverse=False)

        try:
            lidar_pointcloud, times = LidarPointCloud.from_file_multisweep(level5data, sample_rec, "LIDAR_TOP",
                                                                           "LIDAR_TOP",
                                                                           num_sweeps=10)
            lidar_pointcloud.transform(car_from_sensor)
        except Exception as e:
            print("Failed to load Lidar Pointcloud for {}: {}:".format(sample_rec, e))

        points = lidar_pointcloud.points  # .transpose()  # .reshape(-1, 5)

        # points = points[:3, :]
        # points[:3, :] = points
        points[3, :] /= 255
        points[3, :] -= 0.5

        points_cat = np.concatenate([points, times], axis=0).transpose()

        points_cat = points_cat[~np.isnan(points_cat).any(axis=1)]

        return points_cat



    def __getitem__(self, idx):
        input_dict = self.get_sensor_data(idx)



        input_dict['lidar']['points'][:,:3] = box_np_ops.rotation_points_single_angle(input_dict['lidar']['points'][:,:3],self.rot,axis=2)
        input_dict['lidar']['points'] *= self.scale

        if not self.multi_test:
            example = self._prep_func(input_dict=input_dict)
        else:
            points = input_dict["lidar"]["points"]
            points_x = np.copy(points)
            points_y = np.copy(points)
            points_xy = np.copy(points)

            points_x[:,0] *= -1
            points_y[:,1] *= -1
            points_xy[:,0:2] *= -1

            example = []
            input_dict['lidar']['points'] = points
            example.append(self._prep_func(input_dict=input_dict))

            input_dict['lidar']['points'] = points_x
            example.append(self._prep_func(input_dict=input_dict))

            input_dict['lidar']['points'] = points_y
            example.append(self._prep_func(input_dict=input_dict))

            input_dict['lidar']['points'] = points_xy
            example.append(self._prep_func(input_dict=input_dict))
        # example["metadata"] = input_dict["metadata"]
        return example

    def get_sensor_data(self, query):

        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },
        }

        points = self.getPoints(query)

        res["lidar"]["points"] = points

        return res


if __name__ == "__main__":
    train_data = AkLyftDataset()
    test_data = AkLyftTestDataset()


