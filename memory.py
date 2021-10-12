import torch as t
import torch.nn as nn
from settings import *
import torch.nn.functional as F
class Memory_Module(nn.Module):
    #initialize memory matrices
    def __init__(self,reduc_size: int = 16,bottleneck_height: int= img_size_height//16 ,bottleneck_width: int= img_size_width//16):
        super().__init__()
        self.b_height,self.b_width=bottleneck_height,bottleneck_width
        self.bottleneck=t.zeros([batch_size,256,self.b_height,self.b_width])
        self.intermediate_3=t.zeros([batch_size,128,2*self.b_height,2*self.b_width])
        self.intermediate_2=t.zeros([batch_size,64,4*self.b_height,4*self.b_width])
        self.intermediate_1=t.zeros([batch_size,64,8*self.b_height,8*self.b_width])
    #Reinitialize memory matrices for a new batch
    def reinitialize(self):
        self.bottleneck=t.zeros([batch_size,256,self.b_height,self.b_width])
        self.intermediate_3=t.zeros([batch_size,128,2*self.b_height,2*self.b_width])
        self.intermediate_2=t.zeros([batch_size,64,4*self.b_height,4*self.b_width])
        self.intermediate_1=t.zeros([batch_size,64,8*self.b_height,8*self.b_width])
    #Put glimpse features in corresponding spatial location of the memory matrices
    def forward(self,Loc,bottleneck,intermediate_3,intermediate_2,intermediate_1):
        feature_masks=[]
        padded_features=[]
        padded_masks=[]
        feat_height=bottleneck[0].size()[1]
        feat_width=bottleneck[0].size()[2]
        feat_depth=bottleneck[0].size()[0]
        for k in range(batch_size):
            mask=t.ones([feat_depth,feat_height,feat_width])
            height_offset=(Loc[k][1].int()//16)
            width_offset=(Loc[k][0].int()//16)
            pad_size=(width_offset,self.b_width-width_offset-feat_width,height_offset,
                      self.b_height-height_offset-feat_height)
    #pad around the glimpse features according to its spatial location + make the binary mask corresponding to glimpse location 
            padded_feat=F.pad(bottleneck[k], pad_size, "constant", 0)
            padded_mask=F.pad(mask, pad_size, "constant", 0)
            padded_masks.append(t.squeeze(padded_mask))
            padded_features.append(t.squeeze(padded_feat))
    #stack the padded features and masks for the whole batch
        padded_features=t.stack(padded_features)
        padded_masks=t.stack(padded_masks)
    #update the memory matrice with the new glimpse's features
        self.bottleneck=(1-padded_masks)*self.bottleneck+padded_masks*padded_features      
    #Repeat for intermediate_3 memory matrice  
        feature_masks=[]
        padded_features=[]
        padded_masks=[]
        feat_height=intermediate_3[0].size()[1]
        feat_width=intermediate_3[0].size()[2]
        feat_depth=intermediate_3[0].size()[0]
        mem_width=self.b_width*2
        mem_height=self.b_height*2
        for k in range(batch_size):
            mask=t.ones([feat_depth,feat_height,feat_width])
            height_offset=(Loc[k][1].int()//8)
            width_offset=(Loc[k][0].int()//8)
            pad_size=(width_offset,mem_width-width_offset-feat_width,height_offset,
                      mem_height-height_offset-feat_height)
            padded_feat=F.pad(intermediate_3[k], pad_size, "constant", 0)
            padded_mask=F.pad(mask, pad_size, "constant", 0)
            padded_masks.append(t.squeeze(padded_mask))
            padded_features.append(t.squeeze(padded_feat))
        padded_features=t.stack(padded_features)
        padded_masks=t.stack(padded_masks)
        self.intermediate_3=(1-padded_masks)*self.intermediate_3+padded_masks*padded_features
        #Repeat for intermediate_2 memory matrice  
        feature_masks=[]
        padded_features=[]
        padded_masks=[]
        feat_height=intermediate_2[0].size()[1]
        feat_width=intermediate_2[0].size()[2]
        feat_depth=intermediate_2[0].size()[0]
        mem_width=self.b_width*4
        mem_height=self.b_height*4
        for k in range(batch_size):
            mask=t.ones([feat_depth,feat_height,feat_width])
            height_offset=(Loc[k][1].int()//4)
            width_offset=(Loc[k][0].int()//4)
            pad_size=(width_offset,mem_width-width_offset-feat_width,height_offset,
                      mem_height-height_offset-feat_height)
            padded_feat=F.pad(intermediate_2[k], pad_size, "constant", 0)
            padded_mask=F.pad(mask, pad_size, "constant", 0)
            padded_masks.append(t.squeeze(padded_mask))
            padded_features.append(t.squeeze(padded_feat))
        padded_features=t.stack(padded_features)
        padded_masks=t.stack(padded_masks)
        self.intermediate_2=(1-padded_masks)*self.intermediate_2+padded_masks*padded_features
        #Repeat for intermediate_1 memory matrice  
        feature_masks=[]
        padded_features=[]
        padded_masks=[]
        feat_height=intermediate_1[0].size()[1]
        feat_width=intermediate_1[0].size()[2]
        feat_depth=intermediate_1[0].size()[0]
        mem_width=self.b_width*8
        mem_height=self.b_height*8
        for k in range(batch_size):
            mask=t.ones([feat_depth,feat_height,feat_width])
            height_offset=(Loc[k][1].int()//2)
            width_offset=(Loc[k][0].int()//2)
            pad_size=(width_offset,mem_width-width_offset-feat_width,height_offset,
                      mem_height-height_offset-feat_height)
            padded_feat=F.pad(intermediate_1[k], pad_size, "constant", 0)
            padded_mask=F.pad(mask, pad_size, "constant", 0)
            padded_masks.append(t.squeeze(padded_mask))
            padded_features.append(t.squeeze(padded_feat))
        padded_features=t.stack(padded_features)
        padded_masks=t.stack(padded_masks)
        self.intermediate_1=(1-padded_masks)*self.intermediate_1+padded_masks*padded_features
        return self.bottleneck,self.intermediate_3,self.intermediate_2,self.intermediate_1
