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
import tqdm

def run_train(config_path,
              model_dir,
              create_folder=False,
              display_step=50,
              pretrained_path=None,
              multi_gpu=False,
              measure_time=False,
              resume=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dir = str(Path(model_dir).resolve())
    if create_folder:
        if Path(model_dir).exists():
            model_dir = torchplus.train.create_folder(model_dir)
    model_dir = Path(model_dir)

    if not resume and model_dir.exists():
        raise ValueError("model dir exists and you don't specify resume")

    model_dir.mkdir(parents=True, exist_ok=True)
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
                                            opt_level="01",
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

    try:
        start_tic = time.time()
        print("num samples: %d" % (len(dataset)))
        while True:
            if clear_metrics_every_epoch:
                net.clear_metrics()
            for example in tqdm_notebook(dataloader):
                #             print(example)
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
                    eta = time.time() - start_tic
                    if measure_time:
                        for name, val in net.get_avg_time_dict().items():
                            print(f"avg {name} time = {val * 1000:.3f} ms")

                    metrics["step"] = global_step
                    metrics['epoch'] = global_step / len(dataloader)
                    metrics['steptime'] = np.mean(step_times)
                    # metrics["runtime"].update(time_metrics[0])
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
    torchplus.train.save_models(model_dir, [net, amp_optimizer], net.get_global_step())

    ########### to be continued


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
