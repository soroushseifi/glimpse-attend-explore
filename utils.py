from settings import *
import numpy as np
import cv2
import torch.nn.functional as F
def extract_glimpse(Loc,inputs_placeholder):
    #Loc stores the coordinates of the left corner of the glimpse
    Loc=Loc.int()
    glimpses=[]
    retina_size=glimpse_size//3
    #mask for scale 1
    mask_s1=t.ones([retina_size,retina_size])
    #scale 1 to scale 2 and scale 2 to scale 3 padding
    pad_size=(retina_size//2,retina_size//2,retina_size//2,retina_size//2)
    #mask for scale 2
    mask_s2=t.ones([2*retina_size,2*retina_size])
    #the offset of the corners of each scale compared to the left corner of the glimpse
    offsets_right=[2*glimpse_size//3,(5/6)*glimpse_size,glimpse_size]
    offsets_left=[glimpse_size//3,glimpse_size//6,0]
    for k in range(Loc.shape[0]):
        #if Left corner's location + glimpse height/width goes out of the borders of the image, change the location until it fits 
        while Loc[k][0]>img_size_width-glimpse_size:
            Loc[k][0]=Loc[k][0]-1
        while Loc[k][1]>img_size_height-glimpse_size:
            Loc[k][1]=Loc[k][1]-1
        #scale1 - full resolution center of the glimpse
        borders=[Loc[k][1]+offsets_left[0],Loc[k][1]+offsets_right[0],Loc[k][0]+offsets_left[0],Loc[k][0]+offsets_right[0]]
        scale1=inputs_placeholder[k,:,borders[0]:borders[1],borders[2]:borders[3]].float()
        #scale 2 - x2 downscaled mid-part of the glimpse
        borders=[Loc[k][1]+offsets_left[1],Loc[k][1]+offsets_right[1],Loc[k][0]+offsets_left[1],Loc[k][0]+offsets_right[1]]
        scale2_t=inputs_placeholder[k,:,int(borders[0]):int(borders[1]),int(borders[2]):int(borders[3])].float()
        #x2 downscale
        scale2_ds=F.interpolate(t.unsqueeze(scale2_t,0),size=[retina_size,retina_size],mode='bilinear',align_corners=False)
        scale2=t.squeeze(F.interpolate(scale2_ds,size=[2*retina_size,2*retina_size],mode='bilinear',align_corners=False))
        #scale 3 - x3 downscaled outer part of the glimpse
        borders=[Loc[k][1]+offsets_left[2],Loc[k][1]+offsets_right[2],Loc[k][0]+offsets_left[2],Loc[k][0]+offsets_right[2]]
        scale3_t=inputs_placeholder[k,:,borders[0]:borders[1],borders[2]:borders[3]].float()
        # x3 downscale
        scale3_ds=F.interpolate(t.unsqueeze(scale3_t,0),size=[retina_size,retina_size],mode='bilinear',align_corners=False)
        scale3=t.squeeze(F.interpolate(scale3_ds,size=[glimpse_size,glimpse_size],mode='bilinear',align_corners=False))
        #pad scale1 with zeros so that it has the same size as scale2
        scale1_padded=F.pad(scale1, pad_size, "constant", 0)
        mask_s12=F.pad(mask_s1, pad_size, "constant", 0)
        #fill the surrondings of scale 1 with scale2 (scale1+scale2)
        scale12=(1-mask_s12)*scale2+mask_s12*scale1_padded
        #pad (scale1+scale2) to have the same size as scale 3
        scale12_padded=F.pad(scale12, pad_size, "constant", 0)
        mask_s123=F.pad(mask_s12, pad_size, "constant", 0)
        #fill the surronding of (scale1+scale2) with scale 3 (scale2+scale1+scale3=glimpse)
        scale123=(1-mask_s123)*scale3+mask_s123*scale12_padded
        glimpse = scale123.view(channels,glimpse_size, glimpse_size)
        glimpses.append(glimpse)
    glimpses=t.stack(glimpses)
    return glimpses,Loc
import os
def load_model(model):
    best_epoch=0
    epoch_found=False
    best=9999999.0
    curr_epoch=0
    if os.path.exists('checktrain/checkpoint'):
        checkpoint = t.load('checktrain/checkpoint')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lines2 = list(reversed((open('results-test.txt').readlines())))
        for i, line in enumerate(lines2[:]):
            try:
                first,second=line.split('#')
                best_epoch=int(second)+1
                epoch_found=True
            except:
                try:
                    first,second=line.split('@')
                    best=float(second)
                    if epoch_found==True:
                        break
                except:
                    pass
        lines = list((open('results-train.txt').readlines()))
        lines2 = list((open('results-test.txt').readlines()))
        deleted=0
        for i, line in enumerate(lines[:]):
            try:
                first,second=line.split(':')
                _,number=first.split(' ')
                curr_epoch=int(number)
                if curr_epoch<best_epoch:
                    pass
                else:
                    del lines[i-deleted]
                    del lines2[i-deleted]
                    deleted=deleted+1
            except:
                if curr_epoch<best_epoch:
                    pass
                else:
                    del lines[i-deleted]
                    del lines2[i-deleted]
                    deleted=deleted+1
        open('results-train.txt','w').writelines(list(lines))
        open('results-test.txt','w').writelines(list(lines2))
        results_train=open('results-train.txt','a')
        results_test=open('results-test.txt','a')
    else:
        gae=model
        results_train=open('results-train.txt','w')
        results_test=open('results-test.txt','w')
    model.set_best(best,best_epoch)
    return model,best_epoch,results_train,results_test
def set_directories():
    if not os.path.exists('reconstructedimages/'):
        os.makedirs('reconstructedimages/')
    if not os.path.exists('reconstructedimages-test/'):
        os.makedirs('reconstructedimages-test/')
    if not os.path.exists('checktrain/'):
        os.makedirs('checktrain/')
    return
def get_data():
    images = []
    images_test = []
    directory='/sun360/'
    dataset='images.txt'
    counter=0
    with open(directory+dataset) as f:
        for line in f:
            counter=counter+1
    data_counter=0
    with open(directory+dataset) as f:
        for line in f:
            data_counter=data_counter+1
            if data_counter<counter//10:
                images_test.append(directory+line.rstrip('\n'))
            else:
                images.append(directory+line.rstrip('\n'))
    return images,images_test
def resize(image):
    X = cv2.imread(image)
    X=cv2.resize(X,(img_size_width,img_size_height))
    return X
def generate_data(images):
    while True:
        indices=np.arange(len(images))
        np.random.shuffle(indices)
        for i in indices:
            image=resize(images[i])
            yield image
def generate_batch(generator): 
    image_batch=[]
    for i in range(batch_size):
        img=next(generator)
        image_batch.append(img)
    return np.array(image_batch)
