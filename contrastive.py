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
        self.conv_level_2=t.nn.Sequential(
            ResidualAdd(t.nn.Sequential(
            nn.Conv2d(64,64,3,padding=1,bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Conv2d(64,64,3,padding=1,bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
        )),
            nn.Conv2d(64,64,3,padding=1,bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.level_2=t.nn.Sequential(
            nn.Conv2d(64,64,3,padding=1,bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.deconv_1=t.nn.Sequential(
            nn.ConvTranspose2d(64,64,3,stride=2,padding=1,output_padding=[1,1],bias=False),
            nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
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
        self.i3_reduce=nn.Conv2d(128,4,1,bias=False)
        self.i2_reduce=nn.Conv2d(64,1,1,bias=False)
        self.i1_reduce=nn.Conv2d(64,1,1,bias=False)
        self.merge_3=nn.Conv2d(256,128,3,padding=1,bias=False)
        self.bn_merge_3=nn.BatchNorm2d(128,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.deconv_2=nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=[1,1],bias=False)
        self.bn_deconv2=nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.merge_2=nn.Conv2d(128,64,3,padding=1,bias=False)
        self.bn_merge_2=nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.project_bn=t.nn.Sequential(nn.Linear(16*8*8,16*8*8),nn.ReLU(),nn.Linear(16*8*8,16*8*8),nn.ReLU())
        self.project_i3=t.nn.Sequential(nn.Linear(32*16*4,32*16*4),nn.ReLU(),nn.Linear(32*16*4,32*16*4),nn.ReLU())
        self.project_i2=t.nn.Sequential(nn.Linear(64*32,64*32),nn.ReLU(),nn.Linear(64*32,64*32),nn.ReLU())
        self.project_i1=t.nn.Sequential(nn.Linear(128*64,128*64),nn.ReLU(),nn.Linear(128*64,128*64),nn.ReLU())
    def get_contrast_vectors_full(self,bottleneck_reduced2,intermediate_3,intermediate_2,intermediate_1):
        bottleneck_reduced2=bottleneck_reduced2.reshape([batch_size,-1])
        bottlneck_reduced2_normalized=F.normalize(bottleneck_reduced2,dim=1)
        intermediate3_reduced=F.relu(self.i3_reduce(intermediate_3)).reshape([batch_size,-1])
        intermediate3_reduced_normalized=F.normalize(intermediate3_reduced,dim=1)
        intermediate2_reduced=F.relu(self.i2_reduce(intermediate_2)).reshape([batch_size,-1])
        intermediate2_reduced_normalized=F.normalize(intermediate2_reduced,dim=1)
        intermediate1_reduced=F.relu(self.i1_reduce(intermediate_1)).reshape([batch_size,-1])
        intermediate1_reduced_normalized=F.normalize(intermediate1_reduced,dim=1)
        return bottlneck_reduced2_normalized,intermediate3_reduced_normalized,intermediate2_reduced_normalized,intermediate1_reduced_normalized
    def get_contrast_vectors(self,bottleneck_reduced2,intermediate_3,intermediate_2,intermediate_1):
        bottleneck_projected=self.project_bn(bottleneck_reduced2.reshape([batch_size,-1]))
        bottlneck_normalized=F.normalize(bottleneck_projected,dim=1)
        intermediate3_reduced=F.relu(self.i3_reduce(intermediate_3)).reshape([batch_size,-1])
        intermediate3_projected=self.project_i3(intermediate3_reduced)
        intermediate3_normalized=F.normalize(intermediate3_projected,dim=1)
        intermediate2_reduced=F.relu(self.i2_reduce(intermediate_2)).reshape([batch_size,-1])
        intermediate2_projected=self.project_i2(intermediate2_reduced)
        intermediate2_normalized=F.normalize(intermediate2_projected,dim=1)
        intermediate1_reduced=F.relu(self.i1_reduce(intermediate_1)).reshape([batch_size,-1])
        intermediate1_projected=self.project_i1(intermediate1_reduced)
        intermediate1_normalized=F.normalize(intermediate1_projected,dim=1)
        return bottlneck_normalized,intermediate3_normalized,intermediate2_normalized,intermediate1_normalized
    def forward(self,inpainted_bottleneck,mem_3,mem_2,mem_1):
        deconv_3=F.relu(self.bn_deconv_3(self.deconv_3(inpainted_bottleneck)))
        deconv3_skip=t.cat([deconv_3,mem_3],1)        
        level_3=self.bn_merge_3(self.merge_3(deconv3_skip))        
        deconv_2=F.relu(self.bn_deconv2(self.deconv_2(level_3)))
        deconv2_skip=t.cat([deconv_2,mem_2],1)  
        deconv_2_merge=self.bn_merge_2(self.merge_2(deconv2_skip))
        conv_level_2=self.conv_level_2(deconv_2_merge)
        level_2=self.level_2(conv_level_2)
        deconv_1=self.deconv_1(conv_level_2+level_2)
        deconv_1_skip=t.cat([deconv_1,mem_1],1)
        level_1=self.bn_merge_1(self.merge_1(deconv_1_skip))
        pred=self.pred(level_1)
        return pred,level_3,level_2,level_1
