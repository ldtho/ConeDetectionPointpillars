import os
import pathlib
import pickle
import shutil
import time
from functools import partial
import sys
sys.path.append('/kaggle/code/ConeDetectionPointpillarsV2')
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

import fire
import numpy as np
import torch
from google.protobuf import text_format
from tensorboardX import SummaryWriter

import torchplus
import second.data.kitti_common as kitti
from second.builder import target_assigner_builder, voxel_builder
from second.data.preprocess import merge_second_batch
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                      lr_scheduler_builder, optimizer_builder,
                                      second_builder)
from second.utils.progress_bar import ProgressBar

from second.pytorch.utils import get_paddings_indicator
from tqdm import tqdm, tqdm_notebook
from second.core import box_np_ops
import open3d as o3d

def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


def _flat_nested_json_dict(json_dict, flatted, sep=".", start=""):
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, start + sep + k)
        else:
            flatted[start + sep + k] = v


def flat_nested_json_dict(json_dict, sep=".") -> dict:
    """flat a nested json-like dict. this function make shadow copy.
    """
    flatted = {}
    for k, v in json_dict.items():
        if isinstance(v, dict):
            _flat_nested_json_dict(v, flatted, sep, k)
        else:
            flatted[k] = v
    return flatted


def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels", "anchors", "reg_targets", "reg_weights", "bev_map", "rect",
        "Trv2c", "P2"
    ]

    for k, v in example.items():
        if k in float_names:
            example_torch[k] = torch.as_tensor(v, dtype=dtype, device=device)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.as_tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.as_tensor(
                v, dtype=torch.uint8, device=device)
        else:
            example_torch[k] = v
    return example_torch
def _worker_init_fn(worker_id):
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)

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
color = {
    "traffic_cone": (1,0,0),
    "gt_traffic_cone": (0,1,0),
    "pedestrian": (1,1,0),
    "gt_pedestrian": (0,0,1)
}

def train(config_path = "/kaggle/code/ConeDetectionPointpillarsV2/second/configs/pointpillars/cone/xyres_10.proto",
          model_dir = "/kaggle/code/ConeDetectionPointpillarsV2/second/pytorch/outputs/xyres_10",
          ckpt_path = None,
          optim_dir = "/kaggle/code/ConeDetectionPointpillarsV2/second/pytorch/outputs/xyres_10",
          result_path=None,
          create_folder=True,
          display_step=50):
    if create_folder:
        if pathlib.Path(model_dir).exists():
            model_dir = torchplus.train.create_folder(model_dir)

    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_checkpoint_dir = model_dir / 'eval_checkpoints'
    eval_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if result_path is None:
        result_path = model_dir / 'results'
    config_file_bkp = "pipeline.config"
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, config)
    shutil.copyfile(config_path, str(model_dir / config_file_bkp))
    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    class_names = list(input_cfg.class_names)
    ######################
    # BUILD VOXEL GENERATOR
    ######################
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    ######################
    # BUILD TARGET ASSIGNER
    ######################
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    ######################
    # BUILD NET
    ######################
    center_limit_range = model_cfg.post_center_limit_range
    # net = second_builder.build(model_cfg, voxel_generator, target_assigner)
    net = second_builder.build(model_cfg, voxel_generator, target_assigner, input_cfg.batch_size)
    net.cuda()
    # net_train = torch.nn.DataParallel(net).cuda()
    print("num_trainable parameters:", len(list(net.parameters())))
    if ckpt_path is None:
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)

    ######################
    # BUILD OPTIMIZER
    ######################
    # we need global_step to create lr_scheduler, so restore net first.
    gstep = net.get_global_step() - 1
    optimizer_cfg = train_cfg.optimizer
    if train_cfg.enable_mixed_precision:
        net.half()
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    optimizer = optimizer_builder.build(optimizer_cfg, net.parameters())
    if train_cfg.enable_mixed_precision:
        loss_scale = train_cfg.loss_scale_factor
        mixed_optimizer = torchplus.train.MixedPrecisionWrapper(
            optimizer, loss_scale)
    else:
        mixed_optimizer = optimizer
    # must restore optimizer AFTER using MixedPrecisionWrapper
    torchplus.train.try_restore_latest_checkpoints(optim_dir,
                                                   [mixed_optimizer])
    # lr_scheduler = lr_scheduler_builder.build(optimizer_cfg, optimizer, gstep)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.001, steps_per_epoch=3125, epochs=20,
                                                       pct_start=0.4, base_momentum=0.85, max_momentum=0.95,
                                                       div_factor=10.0)
    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=input_cfg.batch_size,
        shuffle=True,
        #     num_workers=input_cfg.num_workers,
        num_workers=16,
        pin_memory=False,
        collate_fn=merge_second_batch,
        worker_init_fn=_worker_init_fn)
    net.train()
    ######################
    # TRAINING
    ######################
    log_path = model_dir / 'log.txt'
    logf = open(log_path, 'a')
    logf.write(proto_str)
    logf.write("\n")
    summary_dir = model_dir / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(summary_dir))

    total_step_elapsed = 0
    remain_steps = train_cfg.steps - net.get_global_step()
    t = time.time()
    ckpt_start_time = t

    total_loop = train_cfg.steps // train_cfg.steps_per_eval + 1
    # total_loop = remain_steps // train_cfg.steps_per_eval + 1
    clear_metrics_every_epoch = train_cfg.clear_metrics_every_epoch

    if train_cfg.steps % train_cfg.steps_per_eval == 0:
        total_loop -= 1
    mixed_optimizer.zero_grad()

    run = True
    debug = False
    display_step = 20
    total_epoch = 20
    epoch = 0
    gstep_to_plot = (30000, 40000, 50000, 60000)
    try:
        while run == True:
            print("num samples: %d" % (len(dataset)))
            epoch += 1
            print("epoch", epoch)
            if epoch > total_epoch:
                break
            if clear_metrics_every_epoch:
                net.clear_metrics()
            for example in tqdm(dataloader):
                lr_scheduler.step()
                example_torch = example_convert_to_torch(example, float_dtype)
                batch_size = example["anchors"].shape[0]
                example_tuple = list(example_torch.values())
                # 0 voxels
                # 1 num_points
                # 2 coordinates
                # 3 anchors
                # 4 anchors_mask
                # 5 labels
                # 6 reg_targets
                # 7 reg_weights
                # 8 metadata
                pillar_x = example_tuple[0][:, :, 0].unsqueeze(0).unsqueeze(0)
                pillar_y = example_tuple[0][:, :, 1].unsqueeze(0).unsqueeze(0)
                pillar_z = example_tuple[0][:, :, 2].unsqueeze(0).unsqueeze(0)
                pillar_i = example_tuple[0][:, :, 3].unsqueeze(0).unsqueeze(0)
                #             pillar_i = torch.ones(pillar_x.shape,dtype=torch.float32, device=pillar_x.device )

                num_points_per_pillar = example_tuple[1].float().unsqueeze(0)

                # Find distance of x, y, and z from pillar center
                # assuming xyres_16.proto
                coors_x = example_tuple[2][:, 3].float()
                coors_y = example_tuple[2][:, 2].float()
                vx, vy = voxel_generator.voxel_size[0], voxel_generator.voxel_size[1]
                x_offset = vx / 2 + voxel_generator.point_cloud_range[0]
                y_offset = vy / 2 + voxel_generator.point_cloud_range[1]
                # self.x_offset = self.vx / 2 + pc_range[0]
                # self.y_offset = self.vy / 2 + pc_range[1]
                # this assumes xyres 20
                # x_sub = coors_x.unsqueeze(1) * 0.16 + 0.1
                # y_sub = coors_y.unsqueeze(1) * 0.16 + -39.9
                # here assumes xyres 16
                x_sub = coors_x.unsqueeze(1) * vx + x_offset
                y_sub = coors_y.unsqueeze(1) * vy + y_offset
                # x_sub = coors_x.unsqueeze(1)*0.28 + 0.14
                # y_sub = coors_y.unsqueeze(1)*0.28 - 20.0
                ones = torch.ones([1, voxel_generator._max_num_points], dtype=torch.float32, device=pillar_x.device)
                x_sub_shaped = torch.mm(x_sub, ones)
                y_sub_shaped = torch.mm(y_sub, ones)

                num_points_for_a_pillar = pillar_x.size()[3]
                mask = get_paddings_indicator(num_points_per_pillar, num_points_for_a_pillar, axis=0)
                mask = mask.permute(0, 2, 1)
                mask = mask.unsqueeze(1)
                mask = mask.type_as(pillar_x)

                coors = example_tuple[2]
                anchors = example_tuple[3]
                labels = example_tuple[4]
                reg_targets = example_tuple[5]

                input = [pillar_x, pillar_y, pillar_z, pillar_i,
                         num_points_per_pillar, x_sub_shaped, y_sub_shaped, mask, coors,
                         anchors, labels, reg_targets]

                ret_dict = net(input)
                # return 0
                # ret_dict {
                #     0:"loss": loss,
                #     1:"cls_loss": cls_loss,
                #     2:"loc_loss": loc_loss,
                #     3:"cls_pos_loss": cls_pos_loss,
                #     4:"cls_neg_loss": cls_neg_loss,
                #     5:"cls_preds": cls_preds,
                #     6:"dir_loss_reduced": dir_loss_reduced,
                #     7:"cls_loss_reduced": cls_loss_reduced,
                #     8:"loc_loss_reduced": loc_loss_reduced,
                #     9:"cared": cared,
                # }
                # cls_preds = ret_dict["cls_preds"]
                cls_preds = ret_dict[5]
                # loss = ret_dict["loss"].mean()
                loss = ret_dict[0].mean()
                # cls_loss_reduced = ret_dict["cls_loss_reduced"].mean()
                cls_loss_reduced = ret_dict[7].mean()
                # loc_loss_reduced = ret_dict["loc_loss_reduced"].mean()
                loc_loss_reduced = ret_dict[8].mean()
                # cls_pos_loss = ret_dict["cls_pos_loss"]
                cls_pos_loss = ret_dict[3]
                # cls_neg_loss = ret_dict["cls_neg_loss"]
                cls_neg_loss = ret_dict[4]
                # loc_loss = ret_dict["loc_loss"]
                loc_loss = ret_dict[2]
                # cls_loss = ret_dict["cls_loss"]
                cls_loss = ret_dict[1]
                # dir_loss_reduced = ret_dict["dir_loss_reduced"]
                dir_loss_reduced = ret_dict[6]
                # cared = ret_dict["cared"]
                cared = ret_dict[9]
                # labels = example_torch["labels"]
                labels = example_tuple[5]
                if train_cfg.enable_mixed_precision:
                    loss *= loss_scale
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                mixed_optimizer.step()
                mixed_optimizer.zero_grad()
                net.update_global_step()
                #             net_metrics = net.update_metrics(cls_loss_reduced,
                #                                              loc_loss_reduced, cls_preds,
                #                                              labels, cared)

                step_time = (time.time() - t)
                t = time.time()
                metrics = {}
                num_pos = int((labels > 0)[0].float().sum().cpu().numpy())
                num_neg = int((labels == 0)[0].float().sum().cpu().numpy())
                # if 'anchors_mask' not in example_torch:
                #     num_anchors = example_torch['anchors'].shape[1]
                # else:
                #     num_anchors = int(example_torch['anchors_mask'][0].sum())
                #             num_anchors = int(example_tuple[7][0].sum())
                global_step = net.get_global_step()
                if global_step % display_step == 0:
                    print("global_step", global_step, "loss", loss.detach().cpu(), "loc_loss",
                          loc_loss.detach().cpu().sum() * 2, "cls_loss", cls_loss.detach().cpu().sum())
                #                 loc_loss_elem = [
                #                     float(loc_loss[:, :, i].sum().detach().cpu().numpy() /
                #                           batch_size) for i in range(loc_loss.shape[-1])
                #                 ]
                #                 metrics["step"] = global_step
                #                 metrics["steptime"] = step_time
                #                 metrics.update(net_metrics)
                #                 metrics["loss"] = {}
                #                 metrics["loss"]["loc_elem"] = loc_loss_elem
                #                 metrics["loss"]["cls_pos_rt"] = float(
                #                     cls_pos_loss.detach().cpu().numpy())
                #                 metrics["loss"]["cls_neg_rt"] = float(
                #                     cls_neg_loss.detach().cpu().numpy())
                #                 # if unlabeled_training:
                #                 #     metrics["loss"]["diff_rt"] = float(
                #                 #         diff_loc_loss_reduced.detach().cpu().numpy())
                #                 if model_cfg.use_direction_classifier:
                #                     metrics["loss"]["dir_rt"] = float(
                #                         dir_loss_reduced.detach().cpu().numpy())
                #                 # metrics["num_vox"] = int(example_torch["voxels"].shape[0])
                #                 metrics["num_vox"] = int(example_tuple[0].shape[0])
                #                 metrics["num_pos"] = int(num_pos)
                #                 metrics["num_neg"] = int(num_neg)
                #                 metrics["num_anchors"] = int(num_anchors)
                #                 metrics["lr"] = float(
                #                     mixed_optimizer.param_groups[0]['lr'])
                #                 # metrics["image_idx"] = example['image_idx'][0]
                #                 flatted_metrics = flat_nested_json_dict(metrics)
                #                 flatted_summarys = flat_nested_json_dict(metrics, "/")
                #                 for k, v in flatted_summarys.items():
                #                     if isinstance(v, (list, tuple)):
                #                         v = {str(i): e for i, e in enumerate(v)}
                #                         writer.add_scalars(k, v, global_step)
                #                     else:
                #                         writer.add_scalar(k, v, global_step)
                #                 metrics_str_list = []
                #                 for k, v in flatted_metrics.items():
                #                     if isinstance(v, float):
                #                         metrics_str_list.append(f"{k}={v:.3}")
                #                     elif isinstance(v, (list, tuple)):
                #                         if v and isinstance(v[0], float):
                #                             v_str = ', '.join([f"{e:.3}" for e in v])
                #                             metrics_str_list.append(f"{k}=[{v_str}]")
                #                         else:
                #                             metrics_str_list.append(f"{k}={v}")
                #                     else:
                #                         metrics_str_list.append(f"{k}={v}")
                #                 log_str = ', '.join(metrics_str_list)
                #                 print(log_str, file=logf)
                #                 print(log_str)
                ckpt_elasped_time = time.time() - ckpt_start_time
                if ckpt_elasped_time > train_cfg.save_checkpoints_secs:
                    torchplus.train.save_models(model_dir, [net, optimizer],
                                                net.get_global_step())
                    ckpt_start_time = time.time()
            #         total_step_elapsed += steps
            torchplus.train.save_models(model_dir, [net, optimizer],
                                        net.get_global_step())


            # Ensure that all evaluation points are saved forever
            torchplus.train.save_models(eval_checkpoint_dir, [net, optimizer], net.get_global_step(), max_to_keep=100)

    except Exception as e:
        torchplus.train.save_models(model_dir, [net, optimizer],
                                    net.get_global_step())
        logf.close()
        raise e
    finally:
        torchplus.train.save_models(model_dir, [net, optimizer], net.get_global_step())
        logf.close()

if __name__ == '__main__':
    train()