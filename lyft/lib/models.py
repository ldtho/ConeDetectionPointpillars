from second.pytorch.models.rpn import *
from second.pytorch.models.middle import *
from second.pytorch.models.voxel_encoder import *
import sparseconvnet as scn
from sync_batchnorm import SynchronizedBatchNorm2d

class Conv2d_WS(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        #std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1) + 1e-5

        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class ConvTranspose2d_WS(nn.ConvTranspose2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                 padding, output_padding, groups, bias, dilation)

    def forward(self, input,output_size=None):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        #std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)

        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)
        return F.conv_transpose2d(
            input, weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)


class SEBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1,ws=True):
        super().__init__()
        if ws:
            Conv2d = change_default_args(bias = False)(Conv2d_WS)
            BatchNorm2d = change_default_args(
                eps=1e-3, num_groups = 32)(GroupNorm)
        else:
            Conv2d = change_default_args(bias = False)(nn.Conv2d)
            BatchNorm2d = change_default_args(
                eps=1e-3, momentum=0.01)(SynchronizedBatchNorm2d)

        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                BatchNorm2d(planes)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes//16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(planes//16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        out += self.shortcut(x)
        out = F.relu(out)
        return out

@register_rpn
class ResNetRPNV2(RPNBase):
    #weight average based convoluation and SE net
    def __init__(self, *args, **kw):
        self.inplanes = -1
        super(ResNetRPNV2,self).__init__(*args, **kw)

        for m in self.modules():
            if isinstance(m, Conv2d_WS):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # if zero_init_residual:
        for m in self.modules():
            if isinstance(m, resnet.Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, SEBasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        if self.inplanes == -1:
            self.inplanes = self._num_input_features
        block = SEBasicBlock


        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * 1#block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers), self.inplanes


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlockV3(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockV3, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = SynchronizedBatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = SynchronizedBatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

@register_rpn
class ResNetRPNV3(RPNBase):
    #Just the same to second ResNetRPN but with syn batch norm
    def __init__(self, *args, **kw):
        self.inplanes = -1
        super(ResNetRPNV3, self).__init__(*args, **kw)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # if zero_init_residual:
        for m in self.modules():
            if isinstance(m, resnet.Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlockV3):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        if self.inplanes == -1:
            self.inplanes = self._num_input_features
        block = BasicBlockV3
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                SynchronizedBatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers), self.inplanes


def SparseResNet(dimension, nInputPlanes, layers):
    """
    pre-activated ResNet
    e.g. layers = {{'basic',16,2,1},{'basic',32,2}}
    """
    nPlanes = nInputPlanes
    m = scn.Sequential()

    def residual(nIn, nOut, stride):
        if isinstance(stride,list):
            return scn.Convolution(dimension, nIn, nOut, 3, stride, False)
        if stride > 1:
            return scn.Convolution(dimension, nIn, nOut, 3, stride, False)
        elif nIn != nOut:
            return scn.NetworkInNetwork(nIn, nOut, False)
        else:
            return scn.Identity()
    for blockType, n, reps, stride in layers:
        for rep in range(reps):
            if blockType[0] == 'b':  # basic block
                if rep == 0:
                    m.add(scn.BatchNormReLU(nPlanes))
                    m.add(
                        scn.ConcatTable().add(
                            scn.Sequential().add(
                                scn.SubmanifoldConvolution(
                                    dimension,
                                    nPlanes,
                                    n,
                                    3,
                                    False) if stride == 1 else scn.Convolution(
                                    dimension,
                                    nPlanes,
                                    n,
                                    3,
                                    stride,
                                    False)) .add(
                                scn.BatchNormReLU(n)) .add(
                                scn.SubmanifoldConvolution(
                                    dimension,
                                    n,
                                    n,
                                    3,
                                    False))) .add(
                            residual(
                                nPlanes,
                                n,
                                stride)))
                else:
                    m.add(
                        scn.ConcatTable().add(
                            scn.Sequential().add(
                                scn.BatchNormReLU(nPlanes)) .add(
                                scn.SubmanifoldConvolution(
                                    dimension,
                                    nPlanes,
                                    n,
                                    3,
                                    False)) .add(
                                scn.BatchNormReLU(n)) .add(
                                scn.SubmanifoldConvolution(
                                    dimension,
                                    n,
                                    n,
                                    3,
                                    False))) .add(
                            scn.Identity()))
            nPlanes = n
            m.add(scn.AddTable())
    m.add(scn.BatchNormReLU(nPlanes))
    return m

@register_middle
class SpMiddleFacebook(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddleFHD'):
        super(SpMiddleFacebook, self).__init__()
        layers = [
            ['basic', 16, 2, 2],
            ['basic', 32, 2, 2],
            ['basic', 64, 2, 2],
            ['basic', 128, 2, [2, 1, 1]]
        ]
        self.middle_feature_extractor = scn.Sequential(
            SparseResNet(dimension=3, nInputPlanes=num_input_features, layers=layers),
            scn.SparseToDense(3, 128)
        )

        self.output_shape = output_shape
        # self.grid = torch.full([self.max_batch_size, *sparse_shape], -1, dtype=torch.int32).cuda()

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        inputSpatialSize = self.middle_feature_extractor.input_spatial_size(
            torch.LongTensor([self.output_shape[1] // 16, self.output_shape[2] // 8, self.output_shape[3] // 8]))
        input_layer = scn.InputLayer(3, inputSpatialSize)

        locations = coors[:, [1, 2, 3, 0]]

        # for padding
        for i in range(3):
            locations[:,i] += (inputSpatialSize[i] - self.output_shape[i+1])//2


        features = voxel_features
        input = input_layer([locations, features])
        spatial_features = self.middle_feature_extractor(input)
        spatial_features = spatial_features.view(spatial_features.shape[0], -1, spatial_features.shape[-2],
                                                 spatial_features.shape[-1])

        return spatial_features
