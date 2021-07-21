"""
PointPillars fork from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""

import torch
from torch import nn
from torch.nn import functional as F

from second.pytorch.utils import get_paddings_indicator
from torchplus.nn import Empty
from torchplus.tools import change_default_args


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels = 9,
                 out_channels = (64,),
                 use_norm=True,
                 last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels
        self.in_channels = in_channels

        self.linear= nn.Linear(self.in_channels, self.units, bias = False)
        self.norm = nn.BatchNorm2d(self.units, eps=1e-3, momentum=0.01)

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.units, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=100, out_channels=1, kernel_size=1, stride=1)

        self.t_conv = nn.ConvTranspose2d(100, 1, (1,8), stride = (1, 7))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1,34), stride = (1, 1), dilation = (1, 3))

    def forward(self, input):
        print('x.shape1',input.shape)
        # x = self.t_conv(input)
        x = self.conv1(input)
        print('x.shape2',x.shape)
        # return x
        x = self.norm(x)
        x = F.relu(x)
        x =self.conv3(x)
        print('x.shape3',x.shape)
        return x
        # x = inputs.permute(0, 3, 2, 1).contiguous()
        # x = self.conv1(input)
        # x = self.norm(x)
        # x = F.relu(x)
        # x = x.permute(0, 3, 2, 1)
        # x = self.conv2(x)
        # x = x.squeeze()
        # return x

class PillarFeatureNet(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64,),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'PillarFeatureNet'
        assert len(num_filters) > 0
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        num_filters = [num_input_features] + list(num_filters)
        # self.pfn_layer = PFNLayer(9, 64, use_norm, last_layer=True)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(PFNLayer(in_filters, out_filters, use_norm, last_layer=last_layer))
        # print("num pfn", len(pfn_layers))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, pillar_x, pillar_y, pillar_z, pillar_i, num_points_per_pillar, x_sub_shaped, y_sub_shaped, mask):
    # def forward(self, pillar_x, pillar_y, pillar_z):
        # Find distance of x, y, and z from cluster center
        # pillar_xyz =  torch.cat((pillar_x, pillar_y, pillar_z), 3)
        pillar_xyz =  torch.cat((pillar_x, pillar_y, pillar_z), 1)

        # points_mean = pillar_xyz.sum(dim=2, keepdim=True) / num_points_per_pillar.view(1,-1, 1, 1)
        points_mean = pillar_xyz.sum(dim=3, keepdim=True) / num_points_per_pillar.view(1, 1, -1, 1)
        f_cluster = pillar_xyz - points_mean

        f_center_offset_0 = pillar_x - x_sub_shaped
        f_center_offset_1 = pillar_y - y_sub_shaped
        # f_center_concat = torch.cat((f_center_offset_0, f_center_offset_1), 3)
        f_center_concat = torch.cat((f_center_offset_0, f_center_offset_1), 1)

        # TODO bug!!! in learning pipeline below implementation is not correct
        # f_center_concat = torch.stack((pillar_x, pillar_y), 3)

        # Combine together feature decorations
        # pillar_xyzi =  torch.cat((pillar_x, pillar_y, pillar_z, pillar_i), 3)
        pillar_xyzi =  torch.cat((pillar_x, pillar_y, pillar_z, pillar_i), 1)
        features_list = [pillar_xyzi, f_cluster, f_center_concat]

        # features = torch.cat(features_list, dim=3)
        features = torch.cat(features_list, dim=1)
        masked_features = features * mask
        # return masked_features
        pillar_feature = self.pfn_layers[0](masked_features)
        # print("pilalr_feature size", pillar_feature.size())
        return pillar_feature

class PointPillarsScatter(nn.Module):
    def __init__(self,
                 output_shape,
                 num_input_features=64,
                 batch_size=1):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape
        self.ny = output_shape[2]
        self.nx = output_shape[3]
        self.nchannels = num_input_features
        self.batch_size=batch_size

    # def forward(self, voxel_features, coords, batch_size):
    def forward(self, voxel_features, coords):

        # batch_canvas will be the final output.
        batch_canvas = []
        if self.batch_size == 1:
            canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype,
                                 device=voxel_features.device)
            indices = coords[:, 2] * self.nx + coords[:, 3]
            indices = indices.type(torch.float64)
            transposed_voxel_features = voxel_features.t()

            # Now scatter the blob back to the canvas.
            indices_2d = indices.view(1, -1)
            ones = torch.ones([self.nchannels, 1],dtype=torch.float64, device=voxel_features.device )
            indices_num_channel = torch.mm(ones, indices_2d)
            indices_num_channel = indices_num_channel.type(torch.int64)
            scattered_canvas = canvas.scatter_(1, indices_num_channel, transposed_voxel_features)

            # Append to a list for later stacking.
            batch_canvas.append(scattered_canvas)

            # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
            batch_canvas = torch.stack(batch_canvas, 0)

            # Undo the column stacking to final 4-dim tensor
            batch_canvas = batch_canvas.view(1, self.nchannels, self.ny, self.nx)
            return batch_canvas
        elif self.batch_size == 2:
            first_canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype,
                                  device=voxel_features.device)
            # Only include non-empty pillars
            first_batch_mask = coords[:, 0] == 0
            first_this_coords = coords[first_batch_mask, :]
            first_indices = first_this_coords[:, 2] * self.nx + first_this_coords[:, 3]
            first_indices = first_indices.type(torch.long)
            first_voxels = voxel_features[first_batch_mask, :]
            first_voxels = first_voxels.t()

            # Now scatter the blob back to the canvas.
            first_canvas[:, first_indices] = first_voxels

            # Append to a list for later stacking.
            batch_canvas.append(first_canvas)

            # Create the canvas for this sample
            second_canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype,
                                 device=voxel_features.device)

            second_batch_mask = coords[:, 0] == 1
            second_this_coords = coords[second_batch_mask, :]
            second_indices = second_this_coords[:, 2] * self.nx + second_this_coords[:, 3]
            second_indices = second_indices.type(torch.long)
            second_voxels = voxel_features[second_batch_mask, :]
            second_voxels = second_voxels.t()

            # Now scatter the blob back to the canvas.
            second_canvas[:, second_indices] = second_voxels

            # Append to a list for later stacking.
            batch_canvas.append(second_canvas)

            # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
            batch_canvas = torch.stack(batch_canvas, 0)

            # Undo the column stacking to final 4-dim tensor
            batch_canvas = batch_canvas.view(2, self.nchannels, self.ny, self.nx)
            return batch_canvas
        else:
            print("Expecting batch size less than 2")
            return 0


        # for batch_itt in range(batch_size):
        #     # Create the canvas for this sample
        #     canvas = torch.zeros(self.nchannels, self.nx * self.ny, dtype=voxel_features.dtype,
        #                          device=voxel_features.device)
        #
        #     # Only include non-empty pillars
        #     batch_mask = coords[:, 0] == batch_itt
        #     this_coords = coords[batch_mask, :]
        #     indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
        #     indices = indices.type(torch.long)
        #     voxels = voxel_features[batch_mask, :]
        #     voxels = voxels.t()
        #
        #     # Now scatter the blob back to the canvas.
        #     canvas[:, indices] = voxels
        #
        #     # Append to a list for later stacking.
        #     batch_canvas.append(canvas)
        #
        # # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        # batch_canvas = torch.stack(batch_canvas, 0)
        #
        # # Undo the column stacking to final 4-dim tensor
        # batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny, self.nx)
        #
        # return batch_canvas


model_name = 'BotNet110_S1_Lite_320_pretrained_relu_hybridfreeze_tta_test'

running_loss_history = []
running_corrects_history = []
val_running_loss_history = []
val_running_correct_history = []
val_kappa = []
kappa_epoch = []
best_kappa, best_tta = 0.0, 0.0
for e in range(epochs):

    running_loss = 0.0
    running_correct = 0.0
    val_running_loss = 0.0
    val_running_correct = 0.0
    val_running_kappa = 0.0
    tta_val_running_correct = 0.0
    running_val_preds, running_val_labels, running_tta_val_preds = [], [], []
    #     cur_lr = lr_scheduler.get_last_lr()
    for inputs, labels, metadata in tqdm_notebook(training_loader):
        optimizer.zero_grad()

        inputs = inputs.to(device)
        labels = labels.to(device)
        with autocast():
            outputs = model(inputs)
            #         loss = criterion(outputs, labels)
            loss = criterion(outputs[:, 0], labels.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer=optimizer)
        scaler.update()

        #         loss.backward()
        #         optimizer.step()

        #         lr_scheduler.step()
        #         lr_scheduler.step(e+1)

        preds = torch.round(torch.sigmoid(outputs[:, 0]))
        running_loss += loss.item()
        running_correct += torch.sum(preds == labels.data)
    cur_lr = lr_scheduler.get_last_lr()

    print(f'Evaluating epoch {e +1}')
    with torch.no_grad():
        for val_inputs, val_labels, val_metadata in tqdm_notebook(validation_loader):
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs[:, 0], val_labels.float())

            val_outputs_tta = tta.d4_image2label(model, val_inputs)  # model(val_inputs)
            val_preds_tta = torch.round(torch.sigmoid(val_outputs_tta[:, 0]))
            running_tta_val_preds += list(val_preds_tta.detach().cpu().numpy())
            tta_val_running_correct += torch.sum(val_preds_tta == val_labels.data)

            val_preds = torch.round(torch.sigmoid(val_outputs[:, 0]))
            val_running_loss += val_loss.item()
            val_running_correct += torch.sum(val_preds == val_labels.data)

            running_val_preds += list(val_preds.detach().cpu().numpy())
            running_val_labels += list(val_labels.detach().cpu().numpy())

    # if lr_scheduler.num_bad_epochs >20:
    #     break
    # lr_scheduler.step(val_loss.item())
    epoch_loss = running_loss/len(training_dataset)
    epoch_acc = running_correct.float()/len(training_dataset)
    running_loss_history.append(epoch_loss)
    running_corrects_history.append(epoch_acc)

    valid_kappa = cohen_kappa_score(running_val_labels, running_val_preds)
    tta_valid_kappa = cohen_kappa_score(running_val_labels, running_tta_val_preds)
    tta_val_epoch_acc = tta_val_running_correct.float() / len(validation_dataset)

    if valid_kappa > best_kappa:
        best_kappa = valid_kappa
        torch.save(model.state_dict(), f'{OUTPUT_DIR}/{model_name}_MURA_bestkappa.pth')
        print(f'Epoch {e + 1} best model saved with kappa: {best_kappa:2.2%}')
    if tta_valid_kappa > best_tta:
        best_tta = tta_valid_kappa
        torch.save(model.state_dict(), f'{OUTPUT_DIR}/{model_name}_MURA_bestTTA.pth')
        print(f'Epoch {e + 1} best model saved with tta_kappa: {best_tta:2.2%}')

    kappa_epoch.append(valid_kappa)

    val_epoch_loss = val_running_loss / len(validation_dataset)
    val_epoch_acc = val_running_correct.float() / len(validation_dataset)
    val_running_loss_history.append(val_epoch_loss)
    val_running_correct_history.append(val_epoch_acc)

    print('epoch :', (e+1),'lr:', cur_lr)
    # print('epoch :', (e+1))
    print('training loss: {:.4f}, acc {:.4f} '.format(epoch_loss, epoch_acc.item()))
    print('validation loss: {:.4f}, validation acc {:.4f},tta acc {:.4f} '.format(val_epoch_loss,
                                                                                  val_epoch_acc.item(),
                                                                                  tta_val_epoch_acc.item()))
    print('validation kappa {:.4f}, tta kappa {:.4f} '.format(valid_kappa, tta_valid_kappa))