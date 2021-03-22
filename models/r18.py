import torch
import torch.nn as nn
import pdb


__all__2d = ['ResNet', 'resnet18', 'resnet50']
__all__3d = ['r3d_18', 'mc3_18', 'r2plus1d_18']


model_urls2d = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


model_urls = {
    'r3d_18': 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth',
    'mc3_18': 'https://download.pytorch.org/models/mc3_18-a90a0ba3.pth',
    'r2plus1d_18': 'https://download.pytorch.org/models/r2plus1d_18-91a641e6.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

'''
class Conv3DSimple(nn.Conv3d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)
'''

class Conv3DSimple(nn.Module):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):
        super(Conv3DSimple, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.padding = padding
        
    def forward(self, x):
        return nn.Conv3d(
        in_channels=self.in_planes,
        out_channels=self.out_planes,
        kernel_size=(3, 3, 3),
        stride=self.stride,
        padding=self.padding,
        bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)
#'''

class BasicBlock2d(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock2d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
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


class BasicBlock3d(nn.Module):

    __constants__ = ['downsample']
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock3d, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=False)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck2d(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck2d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck3d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):

        super(Bottleneck3d, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=False)
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=False)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicStem2d(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self):
        super(BasicStem2d, self).__init__(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False))


class BasicStem3d(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self):
        super(BasicStem3d, self).__init__(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False))


class ResNet(nn.Module):

    def __init__(self, block2d, block3d, layers, mix_loc=0, feature_res=False, to2_loc=0, num_classes=400, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        print('Mixed location is ', mix_loc)
        self.feature_res = feature_res
        self.dilation = 1
        self.inplanes = 64
        self.groups = 1
        self.base_width = width_per_group
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layers2d = [BasicStem2d, self._make_layer2d, self._make_layer2d, 
                                 self._make_layer2d, self._make_layer2d]
        layers3d = [BasicStem3d, self._make_layer3d, self._make_layer3d, 
                                 self._make_layer3d, self._make_layer3d]
        # define mix layers
        assert mix_loc >-1
        mix_layers = []
        mix_blocks = []
        for i in range(5):
          if i < mix_loc:
            mix_layers.append(layers2d[i])
            mix_blocks.append(block2d)
          else:
            mix_layers.append(layers3d[i])
            mix_blocks.append(block3d)
        # construct layers
        self.stem = mix_layers[0]()
        self.layer1 = mix_layers[1](mix_blocks[1], 64, layers[0])
        self.layer2 = mix_layers[2](mix_blocks[2], 128, layers[1], stride=2)
        self.layer3 = mix_layers[3](mix_blocks[3], 256, layers[2], stride=2)
        self.layer4 = mix_layers[4](mix_blocks[4], 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        #self.fc = nn.Linear(512 * mix_blocks[3].expansion, num_classes)

        # init weights
        self._initialize_weights() 

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
       # '''
        x = self.avgpool(x)
        # Flatten the layer to fc
        feat = x.flatten(1)
        return feat
        #x = self.fc(feat)

        #return x, feat

    def framediff(self, x):
        shift_x = torch.roll(x, 1, 2)
        return shift_x - x

    def reshape3to2(self, x):
        n, c, t, h, w = x.size()
        x = x.transpose(1,2) # n, t, c, h, w
        x = torch.reshape(x, (n*t, c, h, w))
        return x


    # need to know channel number? 
    # Temporal channel = 16, new_c = c // tc
    def reshape2to3(self, x, frames_per_clip=16):
        t = frames_per_clip
        nt, c, h, w = x.size()
        n = nt // t
        x = torch.reshape(x, (n, t, c, h, w))
        x = x.transpose(1,2) # n, c, t, h, w
        if self.feature_res:
            shift_x = torch.roll(x, 1, 2)
            return shift_x - x
        else:
            return x

    def _make_layer2d(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = nn.BatchNorm2d
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def _make_layer3d(self, block, planes, blocks, stride=1):
        downsample = None
        conv_builder = Conv3DSimple

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def resnet18(pretrained=False, mix_loc=0, **kwargs):
    model = ResNet(BasicBlock2d, BasicBlock3d, [2, 2, 2, 2], mix_loc=mix_loc, **kwargs)
    if pretrained:
        print('Mixed pretrained model weights from [resnet18-2d, resnet18-3d]')
        model = apply_pretrain(model, mix_loc=mix_loc)
    else:
        print('Do not use pretrained models')
    return model


def apply_pretrain(net, mix_loc=1):
  import torchvision.models as models
  net2d = models.resnet18(pretrained=True)
  net3d = models.video.r3d_18(pretrained=True)
  if mix_loc > 0:# default setting, the stem part uses 2d parameters
    net.stem[0] = net2d.conv1#.cuda()
    net.stem[1] = net2d.bn1#.cuda()
  else:
    net.stem[0] = net3d.stem[0]
    net.stem[1] = net3d.stem[1]
  
  # Error code
  # net.layer1 = net2d.layer1 if mix_loc > 1 else net3d.layer1
  
  net.layer1.load_state_dict((net2d.layer1 if mix_loc > 1 else net3d.layer1).state_dict())
  net.layer2.load_state_dict((net2d.layer2 if mix_loc > 2 else net3d.layer2).state_dict())
  net.layer3.load_state_dict((net2d.layer3 if mix_loc > 3 else net3d.layer3).state_dict())
  net.layer4.load_state_dict((net2d.layer4 if mix_loc > 4 else net3d.layer4).state_dict())
 
  del net2d
  del net3d
  return net


if __name__ == '__main__':
  mix_loc = 0
  model = ResNet(BasicBlock2d, BasicBlock3d, [2, 2, 2, 2], mix_loc=mix_loc, num_classes=101)
  from ptflops import get_model_complexity_info
  macs, params = get_model_complexity_info(model, (3, 16, 112, 112))
  print(macs)
  import pdb
  pdb.set_trace()
  model = ResNet(BasicBlock2d, BasicBlock3d, [2, 2, 2, 2], mix_loc=mix_loc, num_classes=101)
  #print(model)
  apply_pretrain(model, mix_loc)
  '''
  tmp = torch.ones([16, 3, 16, 112, 112], dtype=torch.float32)
  x = model(tmp)
  print(x.shape)
  '''
  save_file_path = './tmp.pth'
  states = {'state_dict': model.state_dict()}
  torch.save(states, save_file_path)
