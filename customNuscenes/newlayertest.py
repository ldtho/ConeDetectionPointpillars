import open3d as o3d
from numba.core.errors import NumbaDeprecationWarning,NumbaPendingDeprecationWarning, NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)
import sys
sys.path.append('/kaggle/code/ConeDetectionPointpillars')

from second.data.CustomNuscDataset import * #to register dataset
from models import * #to register model
import torch
from second.utils.log_tool import SimpleModelLog
from second.builder import target_assigner_builder, voxel_builder
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.pytorch.core import box_torch_ops
import torchplus
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from google.protobuf import text_format
# from lyft_dataset_sdk.utils.geometry_utils import *
from lyft_dataset_sdk.lyftdataset import Quaternion
from apex import amp
# from lyft_dataset_sdk.utils.data_classes import Box
from nuscenes.utils.geometry_utils import *
from nuscenes.utils.data_classes import Box
from second.utils.progress_bar import ProgressBar
from tqdm import tqdm, tqdm_notebook

#commented second.core.non_max_suppression, nms_cpu, __init__.py,
# pytorch.core.box_torch_ops line 524
def build_network(model_cfg, measure_time=False):
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    box_coder.custom_ndim = target_assigner._anchor_generators[0].custom_ndim
    net = second_builder.build(
        model_cfg, voxel_generator, target_assigner, measure_time=measure_time)
    return net


def merge_second_batch_multigpu(batch_list):
    if isinstance(batch_list[0], list):
        batch_list_c = []
        for example in batch_list:
            batch_list_c += example
        batch_list = batch_list_c

    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    for key, elems in example_merged.items():
        if key == 'metadata':
            ret[key] = elems
        elif key == "calib":
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = np.stack(v1, axis=0)
        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.stack(coors, axis=0)
        elif key in ['gt_names', 'gt_classes', 'gt_boxes']:
            continue
        else:
            ret[key] = np.stack(elems, axis=0)

    return ret

def buildBBox(points,color = [1,0,0]):
    #print("Let's draw a cubic using o3d.geometry.LineSet")
    # points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1],
    #           [0, 1, 1], [1, 1, 1]] x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1

    points = points[[0,4,3,7,1,5,2,6],:]
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)


    return  line_set
def merge_second_batch(batch_list):
    if isinstance(batch_list[0], list):
        batch_list_c = []
        for example in batch_list:
            batch_list_c += example
        batch_list = batch_list_c

    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    for key, elems in example_merged.items():
        if key in [
            'voxels', 'num_points', 'num_gt', 'voxel_labels', 'gt_names', 'gt_classes', 'gt_boxes'
        ]:
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'metadata':
            ret[key] = elems
        elif key == "calib":
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = np.stack(v1, axis=0)
        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        elif key == 'metrics':
            ret[key] = elems
        else:
            ret[key] = np.stack(elems, axis=0)
    return ret


def _worker_init_fn(worker_id):
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)


def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "importance"
    ]
    for k, v in example.items():
        if k in ['gt_names', 'gt_classes', 'gt_boxes', 'points']:
            example_torch[k] = example[k]
            continue

        if k in float_names:
            # slow when directly provide fp32 data with dtype=torch.half
            example_torch[k] = torch.tensor(
                v, dtype=torch.float32, device=device).to(dtype)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.uint8, device=device)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = torch.tensor(
                    v1, dtype=dtype, device=device).to(dtype)
            example_torch[k] = calib
        elif k == "num_voxels":
            example_torch[k] = torch.tensor(v)
        else:
            example_torch[k] = v
    return example_torch


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d m' % (hr, min)

    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)


    else:
        raise NotImplementedError
def get_paddings_indicator(actual_num, max_num, axis=0):
    """
    Create boolean mask by actually number of a padded tensor.
    :param actual_num:
    :param max_num:
    :param axis:
    :return: [type]: [description]
    """
    actual_num = torch.unsqueeze(actual_num, axis+1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis+1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num : [N, M, 1]
    # tiled_actual_num : [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # title_max_num : [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape : [batch_size, max_num]
    return paddings_indicator
config_path = '/kaggle/code/ConeDetectionPointpillars/customNuscenes/configs/cones_pp_new_layers.config'
# config_path = '/kaggle/code/ConeDetectionPointpillars/customNuscenes/configs/cones_pp_initial_v2.config'
# config_path = '/kaggle/code/ConeDetectionPointpillars/customNuscenes/configs/cones_pp_initialak.config'
model_dir = f'/kaggle/code/ConeDetectionPointpillars/customNuscenes/outputs/cones_pp_new_layers.{time.time()}'
result_path = None
create_folder = False
display_step = 50
# pretrained_path="/kaggle/code/ConeDetectionPointpillars/customNuscenes/outputs/1611755918.3152874/29-1-2021_9:53/voxelnet-5850.tckpt"
pretrained_path = None
multi_gpu = False
measure_time = False
resume = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir = str(Path(model_dir).resolve())
cur_time = time.localtime(time.time())
cur_time = f'{cur_time.tm_year}-{cur_time.tm_mon}-{cur_time.tm_mday}_{cur_time.tm_hour}:{cur_time.tm_min}'
if create_folder:
    if Path(model_dir).exists():
        model_dir = torchplus.train.create_folder(model_dir)
model_dir = Path(model_dir)

if not resume and model_dir.exists():
    raise ValueError("model dir exists and you don't specify resume")

model_dir.mkdir(parents=True, exist_ok=True)
if result_path is None:
    result_path = model_dir / 'results'/ cur_time
config_file_bkp = "pipeline.config"

if isinstance(config_path, str):
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
else:
    config = config_path
    proto_str = text_format.MessageToString(config, indent=2)

with (model_dir / config_file_bkp).open('w') as f:
    f.write(proto_str)
# Read config file
input_cfg = config.train_input_reader
eval_input_cfg = config.eval_input_reader
model_cfg = config.model.second  # model's config
train_cfg = config.train_config  # training config
# Build neural network
net = build_network(model_cfg, measure_time).to(device)

target_assigner = net.target_assigner
voxel_generator = net.voxel_generator
print("num parameter: ", len(list(net.parameters())))
torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
if pretrained_path is not None:
    print('warning pretrain is loaded after restore, careful with resume')
    model_dict = net.state_dict()
    pretrained_dict = torch.load(pretrained_path)

    new_pretrained_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            new_pretrained_dict[k] = v
    print("Load pretrained parameters: ")
    for k, v in new_pretrained_dict.items():
        print(k, v.shape)
    model_dict.update(new_pretrained_dict)
    net.load_state_dict(model_dict)
    net.clear_global_step()
    net.clear_metrics()
if multi_gpu:
    net_parallel = torch.nn.DataParallel(net)
else:
    net_parallel = net
optimizer_cfg = train_cfg.optimizer
loss_scale = train_cfg.loss_scale_factor
fastai_optimizer = optimizer_builder.build(
    optimizer_cfg,
    net,
    mixed=False,
    loss_scale=loss_scale)
if loss_scale < 0:
    loss_scale = "dynamic"
if train_cfg.enable_mixed_precision:
    max_num_voxels = input_cfg.preprocess.max_number_of_voxels * input_cfg.batch_size
    print("max_num_voxels: %d" % (max_num_voxels))

    net, amp_optimizer = amp.initialize(net, fastai_optimizer,
                                        opt_level="O1",
                                        keep_batchnorm_fp32=None,
                                        loss_scale=loss_scale)
    net.metrics_to_float()
else:
    amp_optimizer = fastai_optimizer
torchplus.train.try_restore_latest_checkpoints(model_dir, [fastai_optimizer])
lr_scheduler = lr_scheduler_builder.build(optimizer_cfg, amp_optimizer, train_cfg.steps)

if train_cfg.enable_mixed_precision:
    float_dtype = torch.float16
else:
    float_dtype = torch.float32

if multi_gpu:
    num_gpu = torch.cuda.device_count()
    print(f"MULTI_GPU: use {num_gpu} gpus")
    collate_fn = merge_second_batch_multigpu
else:
    collate_fn = merge_second_batch
    num_gpu = 1
dataset = input_reader_builder.build(
    input_cfg,
    model_cfg,
    training=True,
    voxel_generator=voxel_generator,
    target_assigner=target_assigner,
    multi_gpu=multi_gpu
)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=input_cfg.batch_size * num_gpu,
    shuffle=True,
    num_workers=input_cfg.preprocess.num_workers * num_gpu,
    pin_memory=False,
    collate_fn=collate_fn,
    worker_init_fn=_worker_init_fn,
    drop_last=not multi_gpu
)
model_logging = SimpleModelLog(model_dir)
model_logging.open()
model_logging.log_text(proto_str + "\n", 0, tag="config")

start_step = net.get_global_step()
total_step = train_cfg.steps
t = time.time()
steps_per_eval = train_cfg.steps_per_eval
clear_metrics_every_epoch = train_cfg.clear_metrics_every_epoch

amp_optimizer.zero_grad()
step_times = []
step = start_step
run = True
ave_valid_loss = 0.0
example = next(iter(dataloader))
lr_scheduler.step(net.get_global_step())
example.pop("metrics")
example_torch = example_convert_to_torch(example, float_dtype)
batch_size = example_torch['anchors'].shape[0]
example_tuple = list(example_torch.values())

# traning input from example
pillar_x = example_tuple[0][:,:,0].unsqueeze(0).unsqueeze(0)
pillar_y = example_tuple[0][:,:,1].unsqueeze(0).unsqueeze(0)
pillar_z = example_tuple[0][:,:,2].unsqueeze(0).unsqueeze(0)
pillar_i = example_tuple[0][:,:,3].unsqueeze(0).unsqueeze(0)
num_points_per_pillar = example_tuple[1].float().unsqueeze(0)
coors_x = example_tuple[2][:, 3].float()
coors_y = example_tuple[2][:, 2].float()
vx, vy = voxel_generator.voxel_size[0], voxel_generator.voxel_size[1]
x_offset = vx/2 + voxel_generator.point_cloud_range[0]
y_offset = vy/2 + voxel_generator.point_cloud_range[1]
x_sub = coors_x.unsqueeze(1)*vx + x_offset
y_sub = coors_y.unsqueeze(1)*vy + y_offset
ones = torch.ones([1,voxel_generator._max_num_points], dtype = torch.float32, device = pillar_x.device)
x_sub_shaped = torch.mm(x_sub, ones)
y_sub_shaped = torch.mm(y_sub, ones)
num_points_for_a_pillar = pillar_x.size()[3]
mask = get_paddings_indicator(num_points_per_pillar, num_points_for_a_pillar, axis=0)
mask = mask.permute(0,2,1)
mask = mask.unsqueeze(1)
mask = mask.type_as(pillar_x)
coors = example_tuple[2]
anchors = example_tuple[4]
labels = example_tuple[6]
reg_targets = example_tuple[7]
input_example = [pillar_x, pillar_y, pillar_z, pillar_i, num_points_per_pillar,
                 x_sub_shaped, y_sub_shaped, mask, coors, anchors, labels, reg_targets]
ret_dict = net(input_example)
print(ret_dict)
