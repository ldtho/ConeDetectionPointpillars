import open3d as o3d
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)
import sys
from os import path

sys.path.append('/kaggle/code/ConeDetectionPointpillars')

from second.data.CustomNuscDataset import *  # to register dataset
from models import *  # to register model
import torch
from second.utils.log_tool import SimpleModelLog
from second.builder import target_assigner_builder, voxel_builder
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.utils.progress_bar import ProgressBar
from apex import amp
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
# from lyft_dataset_sdk.utils.data_classes import Box
from nuscenes.utils.geometry_utils import *
from nuscenes.utils.data_classes import Box
from tqdm import tqdm, tqdm_notebook

NameMappingInverse = {
    "barrier": "movable_object.barrier",
    "bicycle": "vehicle.bicycle",
    "bus": "vehicle.bus.rigid",
    "car": "vehicle.car",
    "construction_vehicle": "vehicle.construction",
    "motorcycle": "vehicle.motorcycle",
    "pedestrian": "human.pedestrian.adult",
    "traffic_cone": "movable_object.trafficcone",
    "trailer": "vehicle.trailer",
    "truck": "vehicle.truck",
}


def run_train(config_path,
              model_dir,
              result_path=None,
              create_folder=False,
              display_step=50,
              pretrained_path=None,
              multi_gpu=False,
              measure_time=False,
              resume=False,
              debug=False
              ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = str(Path(model_dir).resolve())
    if create_folder:
        if Path(model_dir).exists():
            model_dir = torchplus.train.create_folder(model_dir)
    model_dir = Path(model_dir)

    if not resume and model_dir.exists():
        raise ValueError("model dir exists and you don't specify resume")
    cur_time = time.localtime(time.time())
    cur_time = f'{cur_time.tm_mday}-{cur_time.tm_mon}-{cur_time.tm_year}_{cur_time.tm_hour}:{cur_time.tm_min}'
    model_dir.mkdir(parents=True, exist_ok=True)
    if result_path is None:
        result_path = model_dir / 'results' / cur_time
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
        from apex import amp
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
    ################
    # PREPARE INPUT
    ################
    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        multi_gpu=multi_gpu
    )
    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
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
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=input_cfg.batch_size,
        shuffle=False,
        num_workers=eval_input_cfg.preprocess.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    ######################
    # TRAINING
    ######################
    model_logging = SimpleModelLog(model_dir)
    model_logging.open()
    model_logging.log_text(proto_str + "\n", 0, tag='config')
    start_step = net.get_global_step()
    total_step = train_cfg.steps
    t = time.time()
    steps_per_eval = train_cfg.steps_per_eval
    clear_metrics_every_epoch = train_cfg.clear_metrics_every_epoch

    amp_optimizer.zero_grad()
    step_times = []
    step = start_step

    ave_valid_loss = 0.0

    from tqdm import tqdm, tqdm_notebook
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
    model_dir = model_dir / cur_time
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        start_tic = time.time()
        print("num samples: %d" % (len(dataset)))
        while run == True:
            if clear_metrics_every_epoch:
                net.clear_metrics()
            for example in tqdm_notebook(dataloader):
                lr_scheduler.step(net.get_global_step())
                example.pop("metrics")
                example_torch = example_convert_to_torch(example, float_dtype)

                ret_dict = net_parallel(example_torch)
                loss = ret_dict["loss"].mean()
                cls_loss_reduced = ret_dict["cls_loss_reduced"].mean()
                loc_loss_reduced = ret_dict["loc_loss_reduced"].mean()

                if train_cfg.enable_mixed_precision:
                    if net.get_global_step() < 100:
                        loss *= 1e-3
                    with amp.scale_loss(loss, amp_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                amp_optimizer.step()
                amp_optimizer.zero_grad()
                net.update_global_step()

                cls_preds = ret_dict["cls_preds"]
                labels = example_torch["labels"]
                cared = ret_dict["cared"]

                net_metrics = net.update_metrics(cls_loss_reduced,
                                                 loc_loss_reduced, cls_preds,
                                                 labels, cared)
                step_time = (time.time() - t)
                step_times.append(step_time)
                t = time.time()
                metrics = {}
                global_step = net.get_global_step()

                if global_step % display_step == 0:
                    net.eval()
                    det = net(example_torch)
                    print(det[0]['label_preds'])
                    print(det[0]['scores'])
                    net.train()
                    eta = time.time() - start_tic
                    if measure_time:
                        for name, val in net.get_avg_time_dict().items():
                            print(f"avg {name} time = {val * 1000:.3f} ms")

                    metrics["step"] = global_step
                    metrics['epoch'] = global_step / len(dataloader)
                    metrics['steptime'] = np.mean(step_times)
                    metrics['valid'] = ave_valid_loss
                    step_times = []

                    metrics["loss"] = net_metrics['loss']['cls_loss'] + net_metrics['loss']['loc_loss']
                    metrics["cls_loss"] = net_metrics['loss']['cls_loss']
                    metrics["loc_loss"] = net_metrics['loss']['loc_loss']

                    if model_cfg.use_direction_classifier:
                        dir_loss_reduced = ret_dict["dir_loss_reduced"].mean()
                        metrics["dir_rt"] = float(
                            dir_loss_reduced.detach().cpu().numpy())

                    metrics['lr'] = float(amp_optimizer.lr)
                    metrics['eta'] = time_to_str(eta)
                    model_logging.log_metrics(metrics, global_step)

                    net.clear_metrics()
                if global_step % steps_per_eval == 0:
                    torchplus.train.save_models(model_dir, [net, amp_optimizer],
                                                net.get_global_step())
                    model_logging.log_text(f"Model saved: {model_dir}, {net.get_global_step()}", global_step)
                    net.eval()
                    result_path_step = result_path / f"step_{net.get_global_step()}"
                    result_path_step.mkdir(parents=True, exist_ok=True)
                    model_logging.log_text("########################", global_step)
                    model_logging.log_text(" EVALUATE", global_step)
                    model_logging.log_text("########################", global_step)
                    model_logging.log_text("Generating eval predictions...", global_step)
                    t = time.time()
                    detections = []
                    prog_bar = ProgressBar()
                    net.clear_timer()
                    cnt = 0
                    for example in tqdm_notebook(iter(eval_dataloader)):
                        example = example_convert_to_torch(example, float_dtype)
                        detections += net(example)
                    sec_per_ex = len(eval_dataset) / (time.time() - t)
                    model_logging.log_text(
                        f'generate eval predictions finished({sec_per_ex:.2f}/s). Start eval:',
                        global_step)
                    result_dict = eval_dataset.dataset.evaluation(
                        detections, str(result_path_step)
                    )
                    for k, v in result_dict['results'].items():
                        model_logging.log_text(f"Evaluation {k}", global_step)
                        model_logging.log_text(v, global_step)
                    model_logging.log_metrics(result_dict["detail"], global_step)
                    with open(result_path_step / "result.pkl", "wb") as f:
                        pickle.dump(detections, f)
                    net.train()

                step += 1
                if step >= total_step:
                    break
            if step >= total_step:
                break
    except Exception as e:
        model_logging.log_text(str(e), step)
        model_logging.log_text(json.dumps(example['metadata'], indent=2), step)
        torchplus.train.save_models(model_dir, [net, amp_optimizer], step)
        raise e
    finally:
        model_logging.close()
    torchplus.train.save_models(model_dir, [net, amp_optimizer], step)

    ########### to be continued


def buildBBox(points, color=[1, 0, 0]):
    # print("Let's draw a cubic using o3d.geometry.LineSet")
    # points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1],
    #           [0, 1, 1], [1, 1, 1]] x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1

    points = points[[0, 4, 3, 7, 1, 5, 2, 6], :]
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def evaluate(config_path,
             model_dir=None,
             result_path=None,
             ckpt_path=None,
             measure_time=False,
             batch_size=None,
             **kwargs):
    pass


def build_network(model_cfg, measure_time=False):
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner = model_cfg.target_assigner
    box_coder.custom_ndim = target_assigner._anchor_generators[0].custom_ndim
    net = second_builder.build(model_cfg, voxel_generator, target_assigner,
                               measure_time=measure_time)
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


def visualize_evaluation(config_path, model_dir, pretrained_path, multi_gpu=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(config_path, str):
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path
        proto_str = text_format.MessageToString(config, indent=2)


    # Read config file
    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second  # model's config
    train_cfg = config.train_config

    # Build neural network
    net = build_network(model_cfg).to(device)

    # Build Model
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

    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=input_cfg.batch_size,
        shuffle=True,
        num_workers=eval_input_cfg.preprocess.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    # Start visualizing
    net.eval()
    # example = next(iter(dataloader))
    for example in iter(eval_dataloader):
        detection = example_convert_to_torch(example, float_dtype)
        detection = net(detection)
        print(detection)
        filtered_sample_tokens = eval_dataset.dataset.filtered_sample_tokens
        # filtered_sample_tokens = dataset.dataset.filtered_sample_tokens
        index = filtered_sample_tokens.index(detection[0]['metadata']['token'])

        gt_example = eval_dataset.dataset.get_sensor_data(index)
        # gt_example = dataset.dataset.get_sensor_data(index)
        points = gt_example['lidar']['points']
        pc_range = model_cfg.voxel_generator.point_cloud_range
        points = np.array(
            [p for p in points if (pc_range[0] < p[0] < pc_range[3]) & (pc_range[1] < p[1] < pc_range[4]) & (
                    pc_range[2] < p[2] < pc_range[5])])

        gt_boxes = gt_example['lidar']['annotations']['boxes']
        gt_labels = gt_example['lidar']['annotations']['names']
        c = points[:, 3].reshape(-1, 1)
        c = np.concatenate([c, c, c], axis=1)
        points = points[:, 0:3]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(c)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0,
                                                                       origin=[-0, -0, -0])
        geo = [pc, mesh_frame]
        geo = add_prediction_per_class(eval_dataset.dataset.nusc,
                                       detection, gt_boxes, gt_labels,
                                       target_assigner.classes, geo)
        o3d.visualization.draw_geometries(geo)

    net.train()



def visualize_evaluation(config_path, model_dir, pretrained_path, multi_gpu=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(config_path, str):
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path
        proto_str = text_format.MessageToString(config, indent=2)


    # Read config file
    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second  # model's config
    train_cfg = config.train_config

    # Build neural network
    net = build_network(model_cfg).to(device)

    # Build Model
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

    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=input_cfg.batch_size,
        shuffle=True,
        num_workers=eval_input_cfg.preprocess.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    # Start visualizing
    net.eval()
    # example = next(iter(dataloader))
    for example in iter(eval_dataloader):
        detection = example_convert_to_torch(example, float_dtype)
        detection = net(detection)
        print(detection)
        filtered_sample_tokens = eval_dataset.dataset.filtered_sample_tokens
        # filtered_sample_tokens = dataset.dataset.filtered_sample_tokens
        index = filtered_sample_tokens.index(detection[0]['metadata']['token'])

        gt_example = eval_dataset.dataset.get_sensor_data(index)
        # gt_example = dataset.dataset.get_sensor_data(index)
        points = gt_example['lidar']['points']
        pc_range = model_cfg.voxel_generator.point_cloud_range
        points = np.array(
            [p for p in points if (pc_range[0] < p[0] < pc_range[3]) & (pc_range[1] < p[1] < pc_range[4]) & (
                    pc_range[2] < p[2] < pc_range[5])])

        gt_boxes = gt_example['lidar']['annotations']['boxes']
        gt_labels = gt_example['lidar']['annotations']['names']
        c = points[:, 3].reshape(-1, 1)
        c = np.concatenate([c, c, c], axis=1)
        points = points[:, 0:3]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points)
        pc.colors = o3d.utility.Vector3dVector(c)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0,
                                                                       origin=[-0, -0, -0])
        geo = [pc, mesh_frame]
        geo = add_prediction_per_class(eval_dataset.dataset.nusc,
                                       detection, gt_boxes, gt_labels,
                                       target_assigner.classes, geo)
        o3d.visualization.draw_geometries(geo)

    net.train()




def add_prediction_per_class(nusc, detection, gt_boxes, gt_labels, class_names, geometries):
    color = {
    "traffic_cone": (1,0,0),
    "gt_traffic_cone": (0,1,0),
    "pedestrian": (1,1,0),
    "gt_pedestrian": (0,0,1)
    }
    det_boxes = detection[0]['box3d_lidar'].cpu().detach().numpy()
    det_labels = detection[0]['label_preds'].cpu().detach().numpy()
    det_scores = detection[0]['scores'].cpu().detach().numpy()
    for i, class_name in enumerate(class_names):
        mask = np.logical_and(det_labels == i, det_scores > 0.5)
        class_det_boxes = det_boxes[mask]
        class_det_scores = det_scores[mask]
        class_det_labels = det_labels[mask]
        print(len(class_det_boxes),len(class_det_scores),len(class_det_labels))
        print(class_det_scores)
        class_gt_boxes = gt_boxes[gt_labels == class_name]
        class_gt_labels = gt_labels[gt_labels == class_name]

        rbbox_corners = box_np_ops.center_to_corner_box3d(class_det_boxes[:, :3],
                                                          class_det_boxes[:, 3:6],
                                                          class_det_boxes[:, 6],
                                                          origin=(0.5, 0.5, 0.5), axis=2)
        gt_rbbox_corners = box_np_ops.center_to_corner_box3d(class_gt_boxes[:, :3],
                                                             class_gt_boxes[:, 3:6],
                                                             class_gt_boxes[:, 6],
                                                             origin=(0.5, 0.5, 0.5), axis=2)
        for j in range(len(rbbox_corners)):
            geometries.append(buildBBox(rbbox_corners[j],
                                        color=color[class_name]))
        for j in range(len(gt_rbbox_corners)):
            geometries.append(buildBBox(gt_rbbox_corners[j],
                                        color=color[f'gt_{class_name}']))
    return geometries

def plot_mAP():
    pass
