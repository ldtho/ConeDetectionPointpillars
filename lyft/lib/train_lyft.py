import open3d as o3d

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)
from aklyftdataset import *  #to register dataset
from models import *   #to register model


import torch
from second.utils.log_tool import SimpleModelLog
from second.builder import target_assigner_builder, voxel_builder
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.pytorch.core import box_torch_ops
import torchplus

import time
import re
from pathlib import Path
import numpy as np
import pandas as pd
import json
from collections import defaultdict

from google.protobuf import text_format



from lyft_dataset_sdk.utils.geometry_utils import *
from lyft_dataset_sdk.lyftdataset import Quaternion
from lyft_dataset_sdk.utils.data_classes import Box as LyftBox


def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d m'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)


    else:
        raise NotImplementedError

def buildBBox(points,color = [1,0,0]):
    #print("Let\'s draw a cubic using o3d.geometry.LineSet")
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

def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "importance"
    ]
    for k, v in example.items():
        if k in ['gt_names', 'gt_classes', 'gt_boxes','points']:
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


def merge_second_batch(batch_list):

    if isinstance(batch_list[0],list):
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


def merge_second_batch_multigpu(batch_list):

    if isinstance(batch_list[0],list):
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


def _worker_init_fn(worker_id):
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)
    #print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])

def run_train(config_path,
          model_dir,
          create_folder=False,
          display_step=50,
          pretrained_path=None,
          multi_gpu=False,
          measure_time=False,
          resume=False):
    """train a VoxelNet model specified by a config file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = str(Path(model_dir).resolve())
    if create_folder:
        if Path(model_dir).exists():
            model_dir = torchplus.train.create_folder(model_dir)
    model_dir = Path(model_dir)

    if not resume and model_dir.exists():
        raise ValueError("model dir exists and you don't specify resume.")

    model_dir.mkdir(parents=True, exist_ok=True)

    config_file_bkp = "pipeline.config"

    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to train with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path
        proto_str = text_format.MessageToString(config, indent=2)

    with (model_dir / config_file_bkp).open("w") as f:
        f.write(proto_str)

    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    net = build_network(model_cfg, measure_time).to(device)

    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator
    print("num parameters:", len(list(net.parameters())))
    torchplus.train.try_restore_latest_checkpoints(model_dir, [net])

    if pretrained_path is not None:
        print("warning pretrain is loaded after restore, careful with resume")
        model_dict = net.state_dict()
        pretrained_dict = torch.load(pretrained_path)

        new_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                new_pretrained_dict[k] = v
        print("Load pretrained parameters:")
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
        #assert max_num_voxels < 65535, "spconv fp16 training only support this"
        from apex import amp
        net, amp_optimizer = amp.initialize(net, fastai_optimizer,
                                            opt_level="O1",
                                            keep_batchnorm_fp32=None,
                                            loss_scale=loss_scale
                                            )
        net.metrics_to_float()
    else:
        amp_optimizer = fastai_optimizer

    torchplus.train.try_restore_latest_checkpoints(model_dir,
                                                   [fastai_optimizer])
    lr_scheduler = lr_scheduler_builder.build(optimizer_cfg, amp_optimizer,
                                              train_cfg.steps)

    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    if multi_gpu:
        num_gpu = torch.cuda.device_count()
        print(f"MULTI-GPU: use {num_gpu} gpu")
        collate_fn = merge_second_batch_multigpu
    else:
        collate_fn = merge_second_batch
        num_gpu = 1

    ######################
    # PREPARE INPUT
    ######################
    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        multi_gpu=multi_gpu)


    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=input_cfg.batch_size * num_gpu,
        shuffle=True,
        num_workers=input_cfg.preprocess.num_workers * num_gpu,
        pin_memory=False,
        collate_fn=collate_fn,
        worker_init_fn=_worker_init_fn,
        drop_last=not multi_gpu)


    ######################
    # TRAINING
    ######################
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

    ave_valid_loss = 0.0



    if False:
        for index in range(len(dataloader)):
            example = dataset.__getitem__(index)

            voxels = example['voxels']
            gt_boxes = example['gt_boxes']
            p = voxels.reshape(-1, 5)
            c = p[:, 3:4]
            c = np.concatenate([c, c, c], axis=1)

            p = p[:, 0:3]

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(p)
            pc.colors = o3d.utility.Vector3dVector(c)
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[-0, -0, -0])
            geo = [pc, mesh_frame]

            rbbox_corners = box_np_ops.center_to_corner_box3d(
                gt_boxes[:, :3], gt_boxes[:, 3:6], gt_boxes[:, 6], origin=(0.5, 0.5, 0.5), axis=2)

            for i in range(gt_boxes.shape[0]):
                geo.append(buildBBox(rbbox_corners[i], color=[1, 0, 0]))

            o3d.visualization.draw_geometries(geo)
    try:
        start_tic = time.time()
        print("num samples: %d"%(len(dataset)))
        while True:
            if clear_metrics_every_epoch:
                net.clear_metrics()
            for example in dataloader:
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
                                                 loc_loss_reduced,  cls_preds,
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
                    #metrics["runtime"].update(time_metrics[0])
                    metrics['valid'] = ave_valid_loss
                    step_times = []

                    metrics["loss"] =  net_metrics['loss']['cls_loss'] + net_metrics['loss']['loc_loss']
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
        example["metadata"] = "None"
        #print(json.dumps(example["metadata"], indent=2))
        model_logging.log_text(str(e), step)
        model_logging.log_text(json.dumps(example["metadata"], indent=2), step)
        torchplus.train.save_models(model_dir, [net, amp_optimizer],step)
        raise e
    finally:
        model_logging.close()
    torchplus.train.save_models(model_dir, [net, amp_optimizer],net.get_global_step())

def transform_box_t(dataset, boxes, index):
    sample_rec = dataset.lyft.sample[index]
    sample_data_token = sample_rec['data']['LIDAR_TOP']

    sd_record = dataset.lyft.get("sample_data", sample_data_token)
    ego_pose = dataset.lyft.get("ego_pose", sd_record["ego_pose_token"])

    translation = np.array(ego_pose['translation'])
    rotation = Quaternion(ego_pose['rotation'])

    for i in range(boxes.shape[0]):
        center = boxes[i][0:3]
        size = boxes[i][3:6]
        theta = boxes[i][-1]
        rot = Quaternion(scalar=np.cos(theta / 2), vector=[0, 0, np.sin(theta / 2)])
        lBox = LyftBox(center=center, size=size, orientation=rot)
        lBox.rotate(rotation)
        lBox.translate(translation)
        #
        boxes[i][0:3] = lBox.center
        boxes[i][3:6] = lBox.wlh
        boxes[i][-1] = quaternion_yaw(lBox.orientation)

    return boxes


#evaluate with test time augmentation
def run_evaluate_TTA(
             config_path,  #the config file
             model_dir=None, #the trained model directory
             result_name=None, #the output csv
             ckpt_path=None,   #do not use model directory but provide the trained model checkpoint directly
             batch_size=1,
             debug = False,    #use open3d to view the result for debug
             split = False,    #for each frame, we eval 4 copies, if out of memory, set this is to true then it will process 4 copies one by one
             use_train = False, #sometimes we found eval with net.train() result in better result
             with_mix_precision = False  #eval with mix precision to reduce memory
):

    if model_dir is not None:
        model_dir = str(Path(model_dir).resolve())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if result_name is None:
        assert model_dir is not None
        model_dir = Path(model_dir)
        result_name = model_dir / 'lyft3d_pred_tta.csv'
    else:
        result_name = Path(result_name)

    if isinstance(config_path, str):
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second


    net = build_network(model_cfg, measure_time=True).to(device)
    float_type = torch.float32
    if with_mix_precision:
        from apex import amp
        net.half()
        net.metrics_to_float()
        float_type = torch.float16


    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator

    if ckpt_path is None:
        assert model_dir is not None
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)

    eval_dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        multi_gpu = True #must be true if you want to split for less memory, so let set it to be true
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,# * torch.cuda.device_count(),
        shuffle=False,
        num_workers=4 * torch.cuda.device_count(),
        pin_memory=False,
        collate_fn= merge_second_batch_multigpu)


    if use_train:
        net.train()
    else:
        net.eval()

    eval_dataset.dataset.multi_test = True

    t = time.time()

    print("Generate output labels...")


    num_processed = 0

    #net_parallel = torch.nn.DataParallel(net)

    tic = time.time()

    detections = []

    for example in eval_dataloader:

        num_processed += 1
        eta = time.time() - tic
        eta = len(eval_dataloader) / num_processed * eta - eta
        print("\r processed: %d out of %d eta: %s" % (num_processed, len(eval_dataloader), time_to_str(eta, 'sec')), end='', flush=True)


        example_torch = example_convert_to_torch(example,float_type)
        with torch.no_grad():
            if not split:
                preds_dict = net(example_torch, True)
            else:
                example_torch_1 = {}
                example_torch_2 = {}
                example_torch_3 = {}
                example_torch_4 = {}
                for key in example_torch.keys():
                    example_torch_1[key] = example_torch[key][:1]
                    example_torch_2[key] = example_torch[key][1:2]
                    example_torch_3[key] = example_torch[key][2:3]
                    example_torch_4[key] = example_torch[key][3:]

                preds_dict_1 = net(example_torch_1, True)
                preds_dict_2 = net(example_torch_2, True)
                preds_dict_3 = net(example_torch_3, True)
                preds_dict_4 = net(example_torch_4, True)

                preds_dict = {}
                for key in preds_dict_1.keys():
                    preds_dict[key] = torch.cat([preds_dict_1[key],preds_dict_2[key],preds_dict_3[key],preds_dict_4[key]],dim=0)

        if with_mix_precision:
            for key in example_torch.keys():
                if isinstance(example_torch[key],torch.Tensor):
                    if example_torch[key].dtype == torch.float16:
                        example_torch[key] = example_torch[key].float()
            for key in preds_dict.keys():
                if isinstance(preds_dict[key], torch.Tensor):
                    if preds_dict[key].dtype == torch.float16:
                        preds_dict[key] = preds_dict[key].float()

        num_class_with_bg = net._num_class
        if not net._encode_background_as_zeros:
            num_class_with_bg = net._num_class + 1

        batch_size_dev = example_torch['anchors'].shape[0]

        batch_anchors = example_torch["anchors"].view(batch_size_dev, -1, example["anchors"].shape[-1])

        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]

        box_shape = batch_box_preds.shape

        batch_box_preds = batch_box_preds.view(batch_size_dev, -1, net._box_coder.code_size)
        batch_box_preds = net._box_coder.decode_torch(batch_box_preds, batch_anchors).view(box_shape)

        batch_size_dev //= 4
        batch_box_preds_all = batch_box_preds.view(batch_size_dev,4,box_shape[1],box_shape[2],box_shape[3],box_shape[4])
        cls_shape = batch_cls_preds.shape
        batch_cls_preds_all = batch_cls_preds.view(batch_size_dev,4,cls_shape[1],cls_shape[2],cls_shape[3],cls_shape[4])

        predictions_dicts = []

        for batch_idx  in range(batch_size_dev):
            batch_box_preds = batch_box_preds_all[batch_idx]
            batch_cls_preds = batch_cls_preds_all[batch_idx]


            box_preds_o = batch_box_preds[0].view(-1, 7)
            box_preds_flipX = torch.flip(batch_box_preds[1], dims=(-2,)).view(-1, 7)
            box_preds_flipY = torch.flip(batch_box_preds[2], dims=(-3,)).view(-1, 7)
            box_preds_flipXY = torch.flip(batch_box_preds[3], dims=(-2, -3)).view(-1, 7)

            box_preds_flipX[:, 0] *= -1
            box_preds_flipY[:, 1] *= -1

            box_preds_flipXY[:, 0] *= -1
            box_preds_flipXY[:, 1] *= -1

            box_preds_flipX[:, -1] = -box_preds_flipX[:, -1] + np.pi
            box_preds_flipY[:, -1] = -box_preds_flipY[:, -1]
            box_preds_flipXY[:, -1] += np.pi

            cls_preds_o = batch_cls_preds[0].view(-1, num_class_with_bg)
            cls_preds_flipX = torch.flip(batch_cls_preds[1], dims=(-2,)).view(-1, num_class_with_bg)
            cls_preds_flipY = torch.flip(batch_cls_preds[2], dims=(-3,)).view(-1, num_class_with_bg)
            cls_preds_flipXY = torch.flip(batch_cls_preds[3], dims=(-2, -3)).view(-1, num_class_with_bg)

            # batch_cls_preds = batch_cls_preds.view(batch_size, -1,num_class_with_bg)
            cls_preds = (cls_preds_o + cls_preds_flipX + cls_preds_flipY + cls_preds_flipXY) / 4



            # box_preds = box_preds_o
            # cls_preds = batch_cls_preds[0]

            box_preds = (box_preds_o + box_preds_flipX + box_preds_flipY + box_preds_flipXY) / 4
            box_preds[:, -1] = box_preds_o[:, -1]
            cos_t = torch.cos(box_preds_o[:, -1] * 2) + \
                    torch.cos(box_preds_flipX[:, -1] * 2) + \
                    torch.cos(box_preds_flipY[:, -1] * 2) + \
                    torch.cos(box_preds_flipXY[:, -1] * 2)

            sin_t = torch.sin(box_preds_o[:, -1] * 2) + \
                    torch.sin(box_preds_flipX[:, -1] * 2) + \
                    torch.sin(box_preds_flipY[:, -1] * 2) + \
                    torch.sin(box_preds_flipXY[:, -1] * 2)

            cos_t /= 4
            sin_t /= 4

            theta = torch.atan2(sin_t, cos_t) * 0.5
            box_preds[:, -1] = theta

            if True:

                #box_preds = box_preds.float()
                #cls_preds = cls_preds.float()
                total_scores = torch.sigmoid(cls_preds)
                nms_func = box_torch_ops.nms

                if True:
                    top_scores, top_labels = torch.max(
                        total_scores, dim=-1)
                    top_scores_keep = top_scores >= 0.1
                    top_scores = top_scores.masked_select(top_scores_keep)

                    if top_scores.shape[0] != 0:
                        if net._nms_score_thresholds[0] > 0.0:
                            box_preds = box_preds[top_scores_keep]
                            if net._use_direction_classifier:
                                dir_labels = dir_labels[top_scores_keep]
                            top_labels = top_labels[top_scores_keep]
                        boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                        if not net._use_rotate_nms:
                            box_preds_corners = box_torch_ops.center_to_corner_box2d(
                                boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                                boxes_for_nms[:, 4])
                            boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                                box_preds_corners)
                        # the nms in 3d detection just remove overlap boxes.
                        selected = nms_func(
                            boxes_for_nms,
                            top_scores,
                            pre_max_size=net._nms_pre_max_sizes[0],
                            post_max_size=net._nms_post_max_sizes[0],
                            iou_threshold=net._nms_iou_thresholds[0],
                        )
                    else:
                        selected = []
                    # if selected is not None:
                    selected_boxes = box_preds[selected]
                    selected_labels = top_labels[selected]
                    selected_scores = top_scores[selected]
                # finally generate predictions.
                if selected_boxes.shape[0] != 0:
                    box_preds = selected_boxes
                    scores = selected_scores
                    label_preds = selected_labels

                    final_box_preds = box_preds
                    final_scores = scores
                    final_labels = label_preds
                    predictions_dict = {
                        "box3d_lidar": final_box_preds,
                        "scores": final_scores,
                        "label_preds": label_preds,
                    }

                else:
                    dtype = batch_box_preds.dtype
                    device = batch_box_preds.device
                    predictions_dict = {
                        "box3d_lidar":
                            torch.zeros([0, box_preds.shape[-1]],
                                        dtype=dtype,
                                        device=device),
                        "scores":
                            torch.zeros([0], dtype=dtype, device=device),
                        "label_preds":
                            torch.zeros([0], dtype=top_labels.dtype, device=device),
                    }
                predictions_dicts.append(predictions_dict)
        detections += predictions_dicts

        if debug:
            index = batch_size * num_processed - 1
            det_boxes = detections[-1]['box3d_lidar'].cpu().detach().numpy()
            det_label = detections[-1]['label_preds'].cpu().detach().numpy()
            det_scores = detections[-1]['scores'].cpu().detach().numpy()
            example = eval_dataset.dataset.get_sensor_data(index)

            points = example['lidar']['points']

            #example = eval_dataset.dataset._prep_func(input_dict=example)
            #points = example['voxels'].reshape(-1, 5)  # .cpu().detach().numpy()
            p = points[points[:,-1] <0.01]
            c = p[:, 3].reshape(-1, 1)
            c = np.concatenate([c, c, c], axis=1)

            p = p[:, 0:3]

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(p)
            pc.colors = o3d.utility.Vector3dVector(c)
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[-0, -0, -0])
            geo = [pc, mesh_frame]

            rbbox_corners = box_np_ops.center_to_corner_box3d(
                det_boxes[:, :3], det_boxes[:, 3:6], det_boxes[:, 6], origin=(0.5, 0.5, 0.5), axis=2)

            for i in range(det_boxes.shape[0]):
                geo.append(buildBBox(rbbox_corners[i], color=[1, 0, 0]))

            o3d.visualization.draw_geometries(geo)



    print("\n generating csv")
    sub = {}
    for index in range(len(eval_dataset)):
        print("\r %d out of %d"%(index + 1,len(eval_dataset)), end='', flush=True)
        token = eval_dataset.dataset.lyft.sample[index]['token']
        sub_string = " "
        box3d_lidar = detections[index]['box3d_lidar'].cpu().detach().numpy()
        box3d_scores = detections[index]['scores'].cpu().detach().numpy()
        label_preds = detections[index]['label_preds'].cpu().detach().numpy()

        box3d_lidar[:,-1] = -box3d_lidar[:,-1] - np.pi/2

        class_name = target_assigner.classes

        box3d_lidar = transform_box_t(eval_dataset.dataset,box3d_lidar, index)

        for i in range(label_preds.shape[0]):
            box = box3d_lidar[i]

            pred = "%f %f %f %f %f %f %f %f %s "%(
                box3d_scores[i],box[0],box[1],box[2],box[3],box[4],box[5],box[6],class_name[label_preds[i]]
            )
            sub_string += pred
        sub[token] = sub_string


    sub = pd.DataFrame(list(sub.items()))
    sub.columns = pd.Index(['Id', 'PredictionString'])
    sub.head()
    sub.tail()
    sub.to_csv(result_name, index=False)
    print("\n Done")
    print("csv saved to %s"%(result_name))


import math
def convertStringToBox(box_string):
    box_string = box_string.split()
    nbox = len(box_string) // 9
    box = np.zeros((nbox,7))
    box_names = []
    box_scores = np.zeros(nbox)
    for i in range(nbox):
        scores = box_string[9 * i + 0]
        box_val = [a for a in box_string[9 * i + 1:9*i+8]]
        name = box_string[9 * i + 8]
        box[i] = box_val
        box_scores[i] = scores
        box_names.append(name)
    return {
        'boxes':box,
        'names':box_names,
        'scores':box_scores
    }

from second.utils.eval import box3d_overlap_kernel
import second.core.box_np_ops as box_np_ops
def box3d_overlap(boxes, qboxes, criterion=-1, z_axis=1, z_center=1.0):
    """kitti camera format z_axis=1.
    """
    bev_axes = list(range(7))
    bev_axes.pop(z_axis + 3)
    bev_axes.pop(z_axis)

    rinc = box_np_ops.rinter_cc(boxes[:, bev_axes], qboxes[:, bev_axes])
    box3d_overlap_kernel(boxes, qboxes, rinc, criterion, z_axis, z_center)

    return rinc

def do_fuse_boxes(boxes_list,score_thresh = -1.0):


    boxes_list = [b for b in boxes_list if b['boxes'].shape[0] > 0]
    if len(boxes_list) == 0:
        return " "

    det_boxes = []
    det_names = []
    det_scores = []
    if len(boxes_list) == 1:
        box_a = boxes_list[0]
        boxes_list = []
        for i in range(box_a['boxes'].shape[0]):
            det_boxes.append(box_a['boxes'][i])
            det_names.append(box_a['names'][i])
            det_scores.append(box_a['scores'][i])

    else:
        box_a = boxes_list[0]
        boxes_list = boxes_list[1:]
        rinc_list = []
        b_assigned_list = []
        for i in range(len(boxes_list)):
            rinc = box3d_overlap(box_a['boxes'], boxes_list[i]['boxes'], z_axis=2, z_center=0.5)
            rinc_list.append(rinc)
            b_assigned = [False] * boxes_list[i]['boxes'].shape[0]
            b_assigned_list.append(b_assigned)

        for i in range(box_a['boxes'].shape[0]):
            b_idxes = []
            scores = []
            fuse_boxes = []

            for box_b,b_assigned,rinc in zip(boxes_list,b_assigned_list,rinc_list):
                b_idx = -1
                score = 0.0
                for j in range(box_b['boxes'].shape[0]):
                    if b_assigned[j]:
                        continue
                    if rinc[i, j] >= 0.6 and rinc[i, j] > score and box_a['names'][i] == box_b['names'][j]:
                        score = rinc[i, j]
                        b_idx = j

                if b_idx != -1:
                    b_idxes.append(b_idx)
                    scores.append(score)
                    fuse_boxes.append(box_b['boxes'][b_idx])
                    b_assigned[b_idx] = True

            fuse_boxes.append(box_a['boxes'][i])
            scores.append(box_a['scores'][i])

            box = np.zeros(8)
            weight = 0.0
            for bb,score in zip(fuse_boxes,scores):
                box[:6] += score * bb[:6]
                box[-2] += score * math.cos(2 * bb[6])
                box[-1] += score * math.sin(2 * bb[6])
                weight += score
            box /= weight
            angle = math.atan2(box[-1],box[-2])
            box[-2] = angle * 0.5



            det_boxes.append(box[:7])
            det_names.append(box_a['names'][i])
            det_scores.append(weight/len(scores))


    sub_string = " "

    for i in range(len(det_boxes)):
        box = det_boxes[i]
        if det_scores[i] < score_thresh:
            continue
        pred = "%f %f %f %f %f %f %f %f %s " % (
            det_scores[i], box[0], box[1], box[2], box[3], box[4], box[5], box[6], det_names[i]
        )
        sub_string += pred

    if len(boxes_list) > 0:
        for box_b,b_assigned in zip(boxes_list,b_assigned_list):
            b_available = np.array(b_assigned) == False
            box_b['boxes'] = box_b['boxes'][b_available]
            box_b['names'] = np.array(box_b['names'])[b_available]
            box_b['scores'] = np.array(box_b['scores'])[b_available]

        sub_string += " " + do_fuse_boxes(boxes_list[1:])

    return sub_string


def ensemble_csv_multi(csv_names,score_thresh = -1.0):
    import pandas as pd
    pd_list = []
    for name in csv_names:
        pd_list.append(pd.read_csv(name))

    sub = {}
    for id in range(len(pd_list[0].Id)):
        print("\r%d out of %d" % (id, len(pd_list[0].Id)), end='', flush=True)
        token = pd_list[0].Id[id]

        boxes_list = []
        for pd_a in pd_list:
            box_string = pd_a.PredictionString[id]
            boxes = convertStringToBox(box_string)
            boxes_list.append(boxes)
        sub_string = do_fuse_boxes(boxes_list,score_thresh=score_thresh)
        sub[token] = sub_string

    sub = pd.DataFrame(list(sub.items()))
    sub.columns = pd_list[0].columns
    sub.head()
    sub.tail()
    sub.to_csv('../outputs/lyft3d_pred_merge_mutli.csv', index=False)
    print("\n Done")
    print("file saved to %s"%('../outputs/lyft3d_pred_merge_mutli.csv'))


import numpy as np

from lyft_dataset_sdk.utils.geometry_utils import *
from lyft_dataset_sdk.lyftdataset import Quaternion
from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D
import copy
import os
import pandas as pd
import math
import time
from multiprocessing import Process, Queue,Pool




def bb_intersection_over_union(A, B):

    #A = A.astype(np.float32).reshape(1,-1)
    #B = B.astype(np.float32).reshape(1,-1)

    #rinc = box3d_overlap(A, B, z_axis=2, z_center=0.5)

    return A.get_iou(B)


def prefilter_boxes(boxes, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()
    for t in range(len(boxes)):
        for j in range(len(boxes[t])):
            label = labels[t][j]
            score = scores[t][j]
            if score < thr:
                break
            box_part = boxes[t][j]
            # b = [label, float(score) * weights[t], float(box_part[0]), float(box_part[1]), float(box_part[2]),
            #      float(box_part[3]),float(box_part[4]),float(box_part[5]),float(box_part[6])]

            center = box_part[0:3]
            size = box_part[3:6]
            theta = box_part[-1]
            rot = Quaternion(scalar=np.cos(theta / 2), vector=[0, 0, np.sin(theta / 2)])

            b_dict = {
                'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
                'translation': center,
                'size': size,
                'rotation': list(rot),
                'name': label,
                'score': score * weights[t]
            }

            b = Box3D(**b_dict)
            #b.name = label
            #b.score = score * weights[t]


            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict and transform it to numpy array
    for k in new_boxes:
        current_boxes = new_boxes[k]
        scores = []
        for b in  current_boxes:
            scores.append(b.score)
        scores = np.array(scores)
        idx = scores.argsort()[::-1]

        new_boxes[k] = [current_boxes[id] for id in list(idx)]#current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes



def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box
    """
    name = boxes[0].name

    conf = 0
    conf_list = []

    cen = np.zeros(3)
    size = np.zeros(3)
    sin_t = 0
    cos_t = 0

    weight = 0.0
    for bb in boxes:
        score = bb.score
        cen += score * bb.translation
        size += score * bb.size
        theta = quaternion_yaw(bb.quaternion)

        sin_t = score * math.sin(2 * theta)
        cos_t = score * math.cos(2 * theta)

        weight += score

        conf_list.append(score)
        conf += score

    cen /= weight
    size /= weight

    sin_t /= weight
    cos_t /= weight
    theta = math.atan2(sin_t,cos_t) * 0.5


    if conf_type == 'avg':
        score = conf / len(boxes)
    elif conf_type == 'max':
        score = np.array(conf_list).max()


    rot = Quaternion(scalar=np.cos(theta / 2), vector=[0, 0, np.sin(theta / 2)])

    b_dict = {
        'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207fbb039a550991a5149214f98cec136ac',
        'translation': cen,
        'size': size,
        'rotation': list(rot),
        'name': name,
        'score': score
    }


    return Box3D(**b_dict)


def find_matching_box(boxes_list, new_box, match_iou):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box.name != new_box.name:
            continue
        iou = bb_intersection_over_union(box, new_box)
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.6, skip_box_thr=0.0,
                          conf_type='avg', allows_overflow=False):
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 4 numbers.
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param intersection_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param conf_type: how to calculate confidence in weighted boxes. 'avg': average value, 'max': maximum value
    :param allows_overflow: false if we want confidence score not exceed 1.0

    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2).
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights),
                                                                                                     len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max']:
        print('Unknown conf_type: {}. Must be "avg" or "max"'.format(conf_type))
        exit()

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return None

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = []

        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box(weighted_boxes, boxes[j], iou_thr)
            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([copy.deepcopy(boxes[j])])
                weighted_boxes.append(copy.deepcopy(boxes[j]))

        # Rescale confidence based on number of models and boxes
        for i in range(len(new_boxes)):
            if not allows_overflow:
                #score = float(weighted_boxes[i][1])
                weighted_boxes[i].score = weighted_boxes[i].score * min(weights.sum(), len(new_boxes[i])) / weights.sum()
                #weighted_boxes[i][1] = "%f"%(score)
                #weighted_boxes[i][1] * min(weights.sum(), len(new_boxes[i])) / weights.sum()
            else:
                weighted_boxes[i][1] = weighted_boxes[i][1] * len(new_boxes[i]) / weights.sum()
        overall_boxes.append(weighted_boxes)

    overall_boxes = np.concatenate(overall_boxes, axis=0)
    # overall_boxes = overall_boxes[overall_boxes[:, 1].astype(np.float32).argsort()[::-1]]
    # boxes = overall_boxes[:, 2:].astype(np.float32)
    # scores = overall_boxes[:, 1].astype(np.float32)
    # labels = overall_boxes[:, 0]
    return overall_boxes



def fuse_single_process(args):
    start_i, end_i, csv_names=args
    pd_list = []
    for name in csv_names:
        pd_list.append(pd.read_csv(name))

    sub = {}
    tic = time.time()
    num_p = 0
    total_num = end_i - start_i
    for id in range(start_i,end_i):
        num_p += 1
        # if num_p == 10:
        #     break
        t1 = time.time()
        rates = (t1 - tic) / (num_p)
        eta = (total_num - num_p) * rates
        if start_i == 0:
            print("\r%d out of %d eta: %f min" % (num_p, end_i - start_i, eta / 60), end='', flush=True)

        if id >= len(pd_list[0].Id):
            break
        token = pd_list[0].Id[id]

        boxes_dict_list = []
        for pd_a in pd_list:
            box_string = pd_a.PredictionString[id]
            boxes = convertStringToBox(box_string)
            boxes_dict_list.append(boxes)

        boxes_list = []
        scores_list = []
        labels_list = []

        for boxes_dict in boxes_dict_list:
            boxes_list_sub = []
            scores_list_sub = []
            labels_list_sub = []

            for i in range(boxes_dict['boxes'].shape[0]):
                boxes_list_sub.append(boxes_dict['boxes'][i])
                scores_list_sub.append(boxes_dict['scores'][i])
                labels_list_sub.append(boxes_dict['names'][i])

            boxes_list.append(boxes_list_sub)
            scores_list.append(scores_list_sub)
            labels_list.append(labels_list_sub)

        overall_boxes = weighted_boxes_fusion(boxes_list=boxes_list, scores_list=scores_list, labels_list=labels_list)

        sub_string = " "

        for bb in overall_boxes:
            score = bb.score
            cen = bb.translation
            size = bb.size
            theta = quaternion_yaw(bb.quaternion)
            pred = "%f %f %f %f %f %f %f %f %s " % (
                score, cen[0], cen[1], cen[2], size[0], size[1], size[2], theta, bb.name
            )
            sub_string += pred
        # sub_string = do_fuse_boxes(boxes_list, score_thresh=score_thresh)
        sub[token] = sub_string

    sub = pd.DataFrame(list(sub.items()))
    sub.columns = pd_list[0].columns
    return sub
    #Q.put(sub,False)
    # sub.head()
    # sub.tail()
    # sub.to_csv('../outputs/lyft3d_pred_merge_mutli.csv', index=False)
    # print("\n Done")
    # print("file saved to %s" % ('../outputs/lyft3d_pred_merge_mutli.csv'))



def ensemble_wbf(csv_names):

    total_num = 27468
    num_process = os.cpu_count()
    b_size = total_num // num_process
    start_i = 0
    end_i = start_i + b_size
    args = []
    while start_i < total_num:
        print(start_i," ",end_i)
        args.append((start_i,end_i,csv_names))
        #p_list.append(Process(target=fuse_single_process,args=(start_i,end_i,csv_names,Q)))
        start_i = end_i
        end_i += b_size
    p = Pool(num_process)
    result =  p.map(fuse_single_process,args)
    p.close()
    p.join()

    sub = pd.concat(result, ignore_index=True)

    sub.to_csv('../outputs/lyft3d_pred_merge_mutli.csv', index=False)
    print("\n Done")
    print("file saved to %s" % ('../outputs/lyft3d_pred_merge_mutli.csv'))




