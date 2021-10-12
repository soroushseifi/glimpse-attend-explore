import torch as t
import torch.nn as nn
from settings import *
import torch.nn.functional as F
class ResidualAdd(t.nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn=fn
    def forward(self,x):
        res=x
        x=self.fn(x)
        x=x+res
        return x
class Contrastive_Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.contrastive_loss=0
        self.deconv_3=nn.ConvTranspose2d(16,128,3,stride=2,padding=1,output_padding=[1,1],bias=False)
        self.bn_deconv_3=nn.BatchNorm2d(128,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.deconv_1=t.nn.Sequential(
            ResidualAdd(t.nn.Sequential(
            nn.Conv2d(64,64,3,padding=1,bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(64,64,3,padding=1,bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )),
            nn.Conv2d(64,64,3,padding=1,bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            ResidualAdd(t.nn.Sequential(
            nn.Conv2d(64,64,3,padding=1,bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU())),
            nn.ConvTranspose2d(64,64,3,stride=2,padding=1,output_padding=[1,1],bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU() #extra
        )
        self.merge_1=nn.Conv2d(128,64,3,padding=1,bias=False)
        self.bn_merge_1=nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pred=t.nn.Sequential(
            nn.ConvTranspose2d(64,64,3,stride=2,padding=1,output_padding=[1,1],bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(64,channels,3,padding=1,bias=False),
            nn.ReLU()
        )
        self.bottleneck_reduce1=nn.Conv2d(256,16,1,bias=False)
        self.bottleneck_reduce2=nn.Conv2d(16,8,1,bias=False)
        self.bn_bottleneck_reduce=nn.BatchNorm2d(16,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.i3_reduce=nn.Conv2d(128,4,1,bias=False)
        self.i2_reduce=nn.Conv2d(64,1,1,bias=False)
        self.i1_reduce=nn.Conv2d(64,1,1,bias=False)
        self.merge_3=nn.Conv2d(256,128,3,padding=1,bias=False)
        self.bn_merge_3=nn.BatchNorm2d(128,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.deconv_2=nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=[1,1],bias=False)
        self.bn_deconv2=nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.merge_2=nn.Conv2d(128,64,3,padding=1,bias=False)
        self.bn_merge_2=nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    def get_contrast_vectors(self,bottleneck,intermediate_3,intermediate_2,intermediate_1):
        bottleneck_reduced=F.relu(self.bn_bottleneck_reduce(self.bottleneck_reduce1(bottleneck)))
        bottleneck_reduced2=F.relu(self.bottleneck_reduce2(bottleneck_reduced))
        bottleneck_reduced2=bottleneck_reduced2.reshape([batch_size,-1])
        bottlneck_reduced2_normalized=F.normalize(bottleneck_reduced2,dim=1)
        intermediate3_reduced=F.relu(self.i3_reduce(intermediate_3)).reshape([batch_size,-1])
        intermediate3_reduced_normalized=F.normalize(intermediate3_reduced,dim=1)
        intermediate2_reduced=F.relu(self.i2_reduce(intermediate_2)).reshape([batch_size,-1])
        intermediate2_reduced_normalized=F.normalize(intermediate2_reduced,dim=1)
        intermediate1_reduced=F.relu(self.i1_reduce(intermediate_1)).reshape([batch_size,-1])
        intermediate1_reduced_normalized=F.normalize(intermediate1_reduced,dim=1)
        return bottlneck_reduced2_normalized,intermediate3_reduced_normalized,intermediate2_reduced_normalized,intermediate1_reduced_normalized
    def forward(self,inpainted_bottleneck,mem_3,mem_2,mem_1):
        deconv_3=F.relu(self.bn_deconv_3(self.deconv_3(inpainted_bottleneck)))
        deconv3_skip=t.cat([deconv_3,mem_3],1)
        level_3=self.bn_merge_3(self.merge_3(deconv3_skip))
        deconv_2=F.relu(self.bn_deconv2(self.deconv_2(level_3)))
        deconv2_skip=t.cat([deconv_2,mem_2],1)
        level_2=self.bn_merge_2(self.merge_2(deconv2_skip))
        deconv_1=self.deconv_1(level_2)
        deconv_1_skip=t.cat([deconv_1,mem_1],1)
        level_1=self.merge_1(deconv_1_skip)
        pred=self.pred(level_1)
        return pred
#
