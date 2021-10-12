#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch as t
from utils import *
from gae import GAE
#Set folders for storing results
set_directories()
#get a list of train and test images 
images,test_images=get_data()
#find number of training and test batches
nb_batches=len(images)//batch_size
nb_test_batches=len(test_images)//batch_size
#get a generator for test and training imafes
generator_train=generate_data(images)
generator_test=generate_data(test_images)
#load from the checkpoint if it exists orthewise instantiate the model
model=GAE()
model,start_epoch,results_train,results_test=load_model(model)
model.cuda()
for epoch in range(start_epoch,iters):
    last_batch=False
    model.train()
    for batch in range(nb_batches):
        if batch==nb_batches-1:
            last_batch=True
        train_images=generate_batch(generator_train)
        train_images=t.tensor(train_images).float().permute([0,3,1,2]).cuda()
        model(train_images,epoch,batch,results_train,last_batch,is_training=True,save_outputs=True,save_all=False)
    last_batch=False
    with t.no_grad():
        model.eval()
        for test_batch in range(nb_test_batches):
            if test_batch==nb_test_batches-1:
                last_batch=True
            test_images=generate_batch(generator_test)
            test_images=t.tensor(test_images).float().permute([0,3,1,2]).cuda()
            model(test_images,epoch,test_batch,results_test,last_batch,is_training=False,save_outputs=True,save_all=False)
