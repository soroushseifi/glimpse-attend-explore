from settings import *
from utils import *
from contrastive import Contrastive_Module
from attention import Attention_Module
from memory import Memory_Module
from encoder import Encoder
import torch.nn as nn
import copy
class GAE(nn.Module):
    def __init__(self):
        super().__init__()
        #bottleneck features size
        b_height=img_size_height//16
        b_width=img_size_width//16
        self.memory=Memory_Module()
        self.attention_stream=Attention_Module()
        self.contrastive_stream=Contrastive_Module()
        self.encoder=Encoder()
        self.bottleneck_reduce1=nn.Conv2d(256,16,1,bias=False)
        self.bottleneck_reduce2=nn.Conv2d(16,8,1,bias=False)
        self.bn_bottleneck_reduce1=nn.BatchNorm2d(16,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.inpaint=nn.Linear(16*(b_height)*(b_width),16*(b_height)*(b_width))
        self.final_pred=t.zeros([batch_size,channels,img_size_height,img_size_width])
        self.att_pred=t.zeros([batch_size,channels,img_size_height,img_size_width])
        self.cont_pred=t.zeros([batch_size,channels,img_size_height,img_size_width])
        self.masks=t.zeros([batch_size,1,nb_height_patches,nb_width_patches])
        self.attention=t.zeros([batch_size,1,nb_height_patches,nb_width_patches])
        self.errors=[]
        self.best_results=999999.0
        self.best_epoch=-1
        self.contrastive_loss=0
        self.step_loss=0
        self.all_steps_loss=0
        self.all_att_loss=0
        self.all_cont_loss=0
        self.conv_pred1=nn.Conv2d(3*channels,channels,3,padding=1,bias=False)
        self.conv_pred2=nn.Conv2d(channels,channels,3,padding=1,bias=False)
        self.conv_pred3=nn.Conv2d(channels,channels,3,padding=1,bias=False)
        self.conv_pred4=nn.Conv2d(channels,channels,3,padding=1,bias=False)
        self.conv_pred5=nn.Conv2d(channels,channels,3,padding=1,bias=False)
        self.optimizer = t.optim.Adam(self.parameters(), lr = 0.0001)
    def reinitialize(self):
        self.step_loss=0
        self.all_steps_loss=0
        self.all_att_loss=0
        self.all_cont_loss=0
        self.contrastive_loss=0
        self.memory.reinitialize()
        self.final_pred=t.zeros([batch_size,channels,img_size_height,img_size_width])
        self.att_pred=t.zeros([batch_size,channels,img_size_height,img_size_width])
        self.cont_pred=t.zeros([batch_size,channels,img_size_height,img_size_width])
        self.masks=t.zeros([batch_size,1,nb_height_patches,nb_width_patches])
        rand_locs=t.empty(batch_size).uniform_(0,int((nb_height_patches)*(nb_width_patches))).int()
        row_temp=(rand_locs//nb_width_patches).int()
        col_temp=(rand_locs%nb_width_patches).int()
        col_indices=col_temp*(glimpse_size//3)
        row_indices=row_temp*(glimpse_size//3)
        self.Loc=t.cat([col_indices.view([batch_size,-1]),row_indices.view([batch_size,-1])],1)
    def set_best(self,best,best_epoch):
        self.best_results=copy.deepcopy(best)
        self.best_epoch=copy.deepcopy(best_epoch)
    def update_masks(self):
        #update masks for selecting the next attention
        masks_t=[]
        #attention is genertaed at the bottleneck layer, so the mask for glimpse has a size of glimpse_size/3=16
        mask=t.ones([1,3,3])
        for k in range(batch_size):
            height_offset=self.Loc[k][1].int()//16
            width_offset=self.Loc[k][0].int()//16
            pad_size=(width_offset,nb_width_patches-width_offset-3,height_offset,nb_height_patches-height_offset-3)
            padded_mask=F.pad(mask, pad_size, "constant", 0)
            masks_t.append(padded_mask)
        masks_t=t.stack(masks_t)
        self.masks=t.clamp(self.masks+masks_t,0,1)
        return
    def reduce_bottleneck_full(self,bottleneck):
        bottleneck_reduced1=F.relu(self.bottleneck_reduce1(bottleneck))
        bottlneck_reduced2=F.relu(self.bottleneck_reduce2(bottleneck_reduced1))
        return bottlneck_reduced2
    def inpaint_bottleneck(self,bottleneck):
        #reduce the number of channels in the bottleneck memory
        bottleneck_reduced=F.relu(self.bn_bottleneck_reduce1(self.bottleneck_reduce1(bottleneck))).reshape([batch_size,-1])
        #inpaint the features in the bottleneck memory
        #conv6_pred
        bottleneck_inpainted=F.relu(self.inpaint(bottleneck_reduced)).view([batch_size,16,nb_height_patches,nb_width_patches])
        bottleneck_inpainted_reduced=F.relu(self.bottleneck_reduce2(bottleneck_inpainted)).reshape([batch_size,-1])
        return bottleneck_reduced,bottleneck_inpainted,bottleneck_inpainted_reduced
    def final_prediction(self):
        concat=t.cat([self.final_pred,self.att_pred,self.cont_pred],1)
        pred1=F.relu(self.conv_pred1(concat))
        pred2=F.relu(self.conv_pred2(pred1))
        pred3=F.relu(self.conv_pred3(pred2))
        pred4=F.relu(self.conv_pred4(pred3))
        return F.relu(self.conv_pred5(pred4))
    def calculate_contrastive_loss(self,pred_feat,full_feat):
        pos_examples=t.sum(full_feat.detach().view([batch_size,-1])*pred_feat.view([batch_size,-1]),1)
        cont_loss=-(pos_examples.mean())
        self.contrastive_loss=self.contrastive_loss+cont_loss
        return
    def calculate_downstream_loss(self,inputs_placeholder):
        self.error=t.mean(t.sqrt(t.sum((self.final_pred-inputs_placeholder)**2,1)),[1,2])
        self.step_loss=t.sqrt(t.sum((self.final_pred-inputs_placeholder)**2,1)).mean()
        self.all_steps_loss=self.all_steps_loss+self.step_loss
        att_step_loss=t.sqrt(t.sum((self.att_pred-inputs_placeholder)**2,1)).mean()
        self.all_att_loss=self.all_att_loss+att_step_loss
        cont_step_loss=t.sqrt(t.sum((self.cont_pred-inputs_placeholder)**2,1)).mean()
        self.all_cont_loss=self.all_cont_loss+cont_step_loss
        return
    def find_next_location(self):
        _,indices=t.max(((1-self.masks)*self.attention).view([batch_size,-1]),-1)
        col_indices_temp=(indices % nb_width_patches).int()
        row_indices_temp=(indices // nb_width_patches).int()
        col_indices=col_indices_temp*(glimpse_size//3)
        row_indices=row_indices_temp*(glimpse_size//3)
        Loc = t.cat([col_indices.view([batch_size,-1]), row_indices.view([batch_size,-1])],1).int()
        return Loc
    def optimize(self,last_batch):
        cost = self.all_steps_loss+self.all_att_loss+self.all_cont_loss+self.contrastive_loss
        cost.backward()
        self.optimizer.step()
        return
    def scale_to_255(self,img):
        img=img.view([-1]).cpu().data.numpy()
        minimum, maximum = min(img), max(img)
        scale = 255/ (maximum - minimum)
        img=(img-minimum)*scale
        return img
    def save_batch_statistics(self,write_file,epoch,is_training=True,last_batch=False):
        self.errors.append(self.error)
        if last_batch==True:
            self.errors=t.stack(self.errors).mean()
            if self.errors<self.best_results and is_training==False:
                self.best_results=self.errors
                self.best_epoch=epoch
                t.save({'model_state_dict':self.state_dict(),'optimizer_state_dict':self.optimizer.state_dict(),},'checktrain/checkpoint')
            write_file.write(('\nStep %d:\nMean Acuuracy=%5f\nbest@%5f\nbestepoch#%5d'% (epoch,self.errors,self.best_results,self.best_epoch)))
            write_file.flush()
            self.errors=[]
        return
    def save(self,epoch,step,b_number,inputs,is_training):
        if is_training==True:
            directory='reconstructedimages/'
        else:
            directory='reconstructedimages-test/'
        for b in range(batch_size):
            img_number=b_number*batch_size+b
            if step==0:
                img=inputs[b].permute([1,2,0]).cpu().data
                cv2.imwrite(directory+'E'+str(epoch)+'I'+str(img_number)+'-input'+'.png',np.array(np.clip(img,0,255),dtype='uint8'))
            img=self.final_pred[b].permute([1,2,0]).cpu().data
            cv2.imwrite(directory+'E'+str(epoch)+'I'+str(img_number)+'S'+str(step)+'-pred.png',np.array(np.clip(img,0,255),dtype='uint8'))
            img=self.att_pred[b].permute([1,2,0]).cpu().data
            cv2.imwrite(directory+'E'+str(epoch)+'I'+str(img_number)+'S'+str(step)+'-attpred.png',np.array(np.clip(img,0,255),dtype='uint8'))
            img=self.cont_pred[b].permute([1,2,0]).cpu().data
            cv2.imwrite(directory+'E'+str(epoch)+'I'+str(img_number)+'S'+str(step)+'-contpred.png',np.array(np.clip(img,0,255),dtype='uint8'))
            img=cv2.resize(self.scale_to_255(self.masks[b]).reshape([nb_height_patches,nb_width_patches,1]),(img_size_width,img_size_height))
            cv2.imwrite(directory+'E'+str(epoch)+'I'+str(img_number)+'S'+str(step)+'-masks.png',np.array(np.clip(img,0,255),dtype='uint8'))
            img=self.scale_to_255(self.attention[b])
            np.save(directory+'E'+str(epoch)+'I'+str(img_number)+'S'+str(step)+'-heatmap',np.array(np.clip(img,0,255),dtype='uint8'))
        return
    def save_sample(self,epoch,step,b_number,inputs,is_training):
        if is_training==True:
            directory='reconstructedimages/'
        else:
            directory='reconstructedimages-test/'
        img_number=b_number*batch_size
        if step==0:
            img=inputs[0].permute([1,2,0]).cpu().data
            cv2.imwrite(directory+'E'+str(epoch)+'-I'+str(img_number)+'-input'+'.png',np.array(np.clip(img,0,255),dtype='uint8'))
        img=self.final_pred[0].permute([1,2,0]).cpu().data
        cv2.imwrite(directory+'E'+str(epoch)+'-I'+str(img_number)+'S'+str(step)+'-pred.png',np.array(np.clip(img,0,255),dtype='uint8'))
        img=self.att_pred[0].permute([1,2,0]).cpu().data
        cv2.imwrite(directory+'E'+str(epoch)+'-I'+str(img_number)+'-S'+str(step)+'-attpred.png',np.array(np.clip(img,0,255),dtype='uint8'))
        img=self.cont_pred[0].permute([1,2,0]).cpu().data
        cv2.imwrite(directory+'E'+str(epoch)+'-I'+str(img_number)+'-S'+str(step)+'-contpred.png',np.array(np.clip(img,0,255),dtype='uint8'))
        img=cv2.resize(self.scale_to_255(self.masks[0]).reshape([nb_height_patches,nb_width_patches,1]),(img_size_width,img_size_height))
        cv2.imwrite(directory+'E'+str(epoch)+'-I'+str(img_number)+'-S'+str(step)+'-masks.png',np.array(np.clip(img,0,255),dtype='uint8'))
        img=self.scale_to_255(self.attention[0])
        np.save(directory+'E'+str(epoch)+'-I'+str(img_number)+'-S'+str(step)+'-heatmap',np.array(np.clip(img,0,255),dtype='uint8'))  
    def forward(self,inputs,epoch,b_number,write_file,last_batch,is_training,save_outputs=True,save_all=False):
        self.reinitialize()
        for step in range(nglimpses):
            #extract glimpses according to the location - adapt the location so that glimpses fall inside image borders
            glimpses,self.Loc=extract_glimpse(self.Loc,inputs)
            #update attention masks according to the visited location
            self.update_masks()
            #extract the features for glimpses
            bn,i3,i2,i1=self.encoder(glimpses)
            #store features in the memory module
            bn_memory,i3_memory,i2_memory,i1_memory=self.memory(self.Loc,bn,i3,i2,i1)
            #reduce number of bottleneck channels and inpaint the missing features
            reduced_bottleneck,inpainted_bottleneck,inpainted_bottleneck_reduced=self.inpaint_bottleneck(bn_memory)
            #get attention stream's prediction + attention map for exploration
            self.att_pred,self.attention=self.attention_stream(reduced_bottleneck,inpainted_bottleneck,i3_memory,i2_memory,i1_memory)
            #get contrastive stream's prediction
            self.cont_pred,c3,c2,c1=self.contrastive_stream(inpainted_bottleneck,i3_memory,i2_memory,i1_memory)
            #combine attention and contrastive module's preditions with last step's final prediction to get current step's output
            self.final_pred=self.final_prediction()
            #choose the next location to attend in the scene
            self.Loc=self.find_next_location()
            if is_training==True:
                #calculate the downstream and contrastive loss
                self.calculate_downstream_loss(inputs)
                #get the groundtruth features for the full image to be used for the contrastive loss
                bn_gt,i3_gt,i2_gt,i1_gt=self.encoder(inputs)
                #reduce groundtruth bottleneck features
                bn_gt_reduced=self.reduce_bottleneck_full(bn_gt)
                #reduce and normalize groundtruth featurs
                bn_gt_cont,i3_gt_cont,i2_gt_cont,i1_gt_cont=self.contrastive_stream.get_contrast_vectors_full(bn_gt_reduced,i3_gt,i2_gt,i1_gt)
 #reduce project and normalize the predcited features               
                bn_pred_cont,i3_pred_cont,i2_pred_cont,i1_pred_cont=self.contrastive_stream.get_contrast_vectors(inpainted_bottleneck_reduced,c3,c2,c1)
                self.calculate_contrastive_loss(bn_pred_cont,bn_gt_cont)
                self.calculate_contrastive_loss(i3_pred_cont,i3_gt_cont)
                self.calculate_contrastive_loss(i2_pred_cont,i2_gt_cont)
                self.calculate_contrastive_loss(i1_pred_cont,i1_gt_cont)
            if save_outputs==True:
                if save_all==True:
                    self.save(epoch,step,b_number,inputs,is_training)
                elif last_batch==True:
                    self.save_sample(epoch,step,b_number,inputs,is_training)
        self.save_batch_statistics(write_file,epoch,is_training,last_batch)
        if is_training==True:
            self.optimize(last_batch)
        return
    
    
