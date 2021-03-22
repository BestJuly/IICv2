import torch
import torch.nn as nn
import torchvision
from models.r18 import resnet18
import pdb




class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def _init_weights(module, init_linear='normal'):
    assert init_linear in ['normal', 'kaiming'], \
        "Undefined init_linear: {}".format(init_linear)
    for m in module.modules():
        if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class MoCoHead(nn.Module):
    '''The non-linear neck in MoCO v2: fc-relu-fc
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 hid_channels=None,
                 norm=True):
        super(MoCoHead, self).__init__()
        if not hid_channels:
            hid_channels = in_channels
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))
        self.norm = norm
        if not norm:
            print('[Warning] Do not normalize features after projection.')
        self.l2norm = Normalize(2)

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        x = self.mlp(x)
        if self.norm:
            x = self.l2norm(x)
        return x


class IICnet(nn.Module):
    def __init__(self, base_network, feature_size=512, head=True, proj_dim=128):
        """
        Args:
            feature_size (int): 512
        """
        super(IICnet, self).__init__()
        self.base_network = base_network
        self.feature_size = feature_size

        self.head = head
        if self.head:
            self.projector = MoCoHead(feature_size, proj_dim)

    def forward(self, x):
        f = self.base_network(x)
        # using projection head, contrastive learning
        if self.head:
            f = self.projector(f)
        return f


class R18(nn.Module):
    def __init__(self, with_classifier=False, num_classes=101):
        super(R18, self).__init__()
        '''
        model = resnet18(
                pretrained = config.pretrain,
                mix_loc = 0, # 0 for all 3D conv
                feature_res = config.feature_res,
                num_classes = config.classnum)
        '''
        model = torchvision.models.video.r3d_18(pretrained=False, progress=True)
        self.base_network = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.with_classifier = with_classifier
        if with_classifier:
            self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base_network(x)
        x = x.view(-1, 512)
        if self.with_classifier:
            x = self.linear(x)
        return x