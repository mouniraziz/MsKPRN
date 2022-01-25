import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import task_generator as tg
import os
import math
import argparse
import scipy as sp
import scipy.stats
import time
import models

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--support_num_per_class",type = int, default = 5)
parser.add_argument("-b","--query_num_per_class",type = int, default = 10)
parser.add_argument("-e","--episode",type = int, default= 500000)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-v","--val_episode", type = int, default = 300)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
args = parser.parse_args()


# Hyper Parameters
METHOD = "MsKPRN_Models"
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SUPPORT_NUM_PER_CLASS = args.support_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODES = args.episode
TEST_EPISODE = args.test_episode
VAL_EPISODE = args.val_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
	
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m = np.mean(a)
    s = scipy.stats.sem(a)
    h = s * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

def weights_init_kaiming(m):
	classname = m.__class__.__name__
	if classname.find('Conv2d') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm2d') != -1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


def main():
    metatrain_folders,metaval_folders,metatest_folders = tg.mini_imagenet_folders()

    best_accuracy = 0.0
    best_h = 0.0
    EPISODE=0
    
    print("init neural networks")
    feature_encoder = models.FeatureEncoder().apply(weights_init_kaiming).cuda(GPU)
    kron_rel_net= models.KronRelationNets().cuda(GPU).apply(weights_init_kaiming).cuda(GPU)
    
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = ReduceLROnPlateau(feature_encoder_optim,mode="max",factor=0.5,patience=2,verbose=True)
    kron_rel_net_optim = torch.optim.Adam(kron_rel_net.parameters(),lr=LEARNING_RATE)
    kron_rel_net_scheduler = ReduceLROnPlateau(kron_rel_net_optim,mode="max",factor=0.5,patience=2,verbose=True)
    

    if os.path.exists(str(METHOD + "/miniImagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str(METHOD + "/miniImagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl")))
        print("load feature encoder success")
        print("load feature encoder success", file=F_txt)
    if os.path.exists(str(METHOD + "/miniImagenet_kron_rel_net_" + str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl")):
        kron_rel_net.load_state_dict(torch.load(str(METHOD + "/miniImagenet_kron_rel_net_" + str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl")))
        print("load kron relation network success")
        print("load kron relation network success", file=F_txt)
    if os.path.exists(METHOD +"/checkpoint_"+ str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl"):
        print("resuming training... ")
        checkpoint = torch.load(METHOD +"/checkpoint_"+ str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl")
        best_accuracy = checkpoint['best_accuracy']
        best_h = checkpoint['best_h']
        EPISODE = checkpoint['episode']
    if os.path.exists(METHOD) == False:
        os.system('mkdir ' + METHOD)
    F_txt = open(METHOD +"/training_info_"+ str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.txt", 'a')

    dim1= 19
    dim2= 14
    dim3= 10
    print_training = False

    for episode in range(EPISODE,EPISODES):
        if(episode!=0):

            if(print_training==True):
                print("Training...")
                print("Training...", file=F_txt)
                print_training=False

            # init dataset
            task = tg.MiniImagenetTask(metatrain_folders,CLASS_NUM,SUPPORT_NUM_PER_CLASS,QUERY_NUM_PER_CLASS)
            support_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SUPPORT_NUM_PER_CLASS,split="train",shuffle=False)
            query_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=QUERY_NUM_PER_CLASS,split="test",shuffle=True,train_query_argue=True)

            # generate support data and query_data
            supports_84, supports_64, supports_48, support_labels = support_dataloader.__iter__().next() #25*3*84*84
            queries_84, queries_64, queries_48, query_labels = query_dataloader.__iter__().next()

            # generate features
            support_features_84 = feature_encoder(Variable(supports_84).cuda(GPU)) # 25*64*dim*dim
            support_features_84 = support_features_84.view(CLASS_NUM,SUPPORT_NUM_PER_CLASS,FEATURE_DIM,dim1,dim1).sum(1)
            query_features_84 = feature_encoder(Variable(queries_84).cuda(GPU))
            
            # form the QURY_NUM_PER_CLASSxCLASS_NUM relation pairs
            support_features_ext_84 = support_features_84.unsqueeze(0).repeat(QUERY_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
            support_features_ext_84= support_features_ext_84.view(QUERY_NUM_PER_CLASS*CLASS_NUM*CLASS_NUM,FEATURE_DIM,dim1,dim1)
            query_features_ext_84 = query_features_84.unsqueeze(1).repeat(1,CLASS_NUM,1,1,1)
            query_features_ext_84 = query_features_ext_84.reshape(QUERY_NUM_PER_CLASS*CLASS_NUM*CLASS_NUM,FEATURE_DIM,dim1,dim1)

            # generate features
            support_features_64 = feature_encoder(Variable(supports_64).cuda(GPU)) # 25*64*dim*dim
            support_features_64 = support_features_64.view(CLASS_NUM,SUPPORT_NUM_PER_CLASS,FEATURE_DIM,dim2,dim2).sum(1)
            query_features_64 = feature_encoder(Variable(queries_64).cuda(GPU))
            
            # form the QURY_NUM_PER_CLASSxCLASS_NUM relation pairs
            support_features_ext_64 = support_features_64.unsqueeze(0).repeat(QUERY_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
            support_features_ext_64= support_features_ext_64.view(QUERY_NUM_PER_CLASS*CLASS_NUM*CLASS_NUM,FEATURE_DIM,dim2,dim2)
            query_features_ext_64 = query_features_64.unsqueeze(1).repeat(1,CLASS_NUM,1,1,1)
            query_features_ext_64 = query_features_ext_64.reshape(QUERY_NUM_PER_CLASS*CLASS_NUM*CLASS_NUM,FEATURE_DIM,dim2,dim2)

            # generate features
            support_features_48 = feature_encoder(Variable(supports_48).cuda(GPU)) # 25*64*dim*dim
            support_features_48 = support_features_48.view(CLASS_NUM,SUPPORT_NUM_PER_CLASS,FEATURE_DIM,dim3,dim3).sum(1)
            query_features_48 = feature_encoder(Variable(queries_48).cuda(GPU))
            
            # form the QURY_NUM_PER_CLASSxCLASS_NUM relation pairs
            support_features_ext_48 = support_features_48.unsqueeze(0).repeat(QUERY_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
            support_features_ext_48= support_features_ext_48.view(QUERY_NUM_PER_CLASS*CLASS_NUM*CLASS_NUM,FEATURE_DIM,dim3,dim3)
            query_features_ext_48 = query_features_48.unsqueeze(1).repeat(1,CLASS_NUM,1,1,1)
            query_features_ext_48 = query_features_ext_48.reshape(QUERY_NUM_PER_CLASS*CLASS_NUM*CLASS_NUM,FEATURE_DIM,dim3,dim3)


            scores_84 = kron_rel_net(query_features_ext_84,support_features_ext_84)
            scores_64 = kron_rel_net(query_features_ext_64,support_features_ext_64)
            scores_48 = kron_rel_net(query_features_ext_48,support_features_ext_48)
            
            # define the loss function
            mse = nn.MSELoss().cuda(GPU)
            one_hot_labels = Variable(torch.zeros(QUERY_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, query_labels.view(-1,1), 1).cuda(GPU))
            loss =  mse(scores_84,one_hot_labels) + mse(scores_64,one_hot_labels) + mse(scores_48,one_hot_labels) 

            feature_encoder.zero_grad()
            kron_rel_net.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(),0.5)
            torch.nn.utils.clip_grad_norm_(kron_rel_net.parameters(),0.5)

            feature_encoder_optim.step()
            kron_rel_net_optim.step()

            if (episode+1)%100 == 0:
              print("episode:",episode+1,"loss",loss.item())
              print("episode:",episode+1,"loss",loss.item(), file=F_txt)

        if (episode+1)%5000 == 0:
            print("Validation...")
            print("Validation...", file= F_txt)
            val_accuracies = []
            with torch.no_grad():
                for i in range(VAL_EPISODE):
                    total_rewards = 0
                    task = tg.MiniImagenetTask(metaval_folders,CLASS_NUM,SUPPORT_NUM_PER_CLASS,15)
                    support_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SUPPORT_NUM_PER_CLASS,split="train",shuffle=False)
                    num_per_class = 5
                    query_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=num_per_class,split="test",shuffle=False)
                    support_images_84,support_images_64,support_images_48,support_labels = support_dataloader.__iter__().next()
                    for query_images_84,query_images_64,query_images_48,query_labels in query_dataloader:
                        query_size = query_labels.shape[0]
                        
                        # generate features
                        support_features_84 = feature_encoder(Variable(support_images_84).cuda(GPU)) # 25*64*dim*dim
                        support_features_84 = support_features_84.view(CLASS_NUM,SUPPORT_NUM_PER_CLASS,FEATURE_DIM,dim1,dim1).sum(1)
                        query_features_84 = feature_encoder(Variable(query_images_84).cuda(GPU))
                        
                        # form the QURY_NUM_PER_CLASSxCLASS_NUM relation pairs
                        support_features_ext_84 = support_features_84.unsqueeze(0).repeat(num_per_class*CLASS_NUM,1,1,1,1)
                        support_features_ext_84= support_features_ext_84.view(num_per_class*CLASS_NUM*CLASS_NUM,FEATURE_DIM,dim1,dim1)
                        query_features_ext_84 = query_features_84.unsqueeze(1).repeat(1,CLASS_NUM,1,1,1)
                        query_features_ext_84 = query_features_ext_84.reshape(num_per_class*CLASS_NUM*CLASS_NUM,FEATURE_DIM,dim1,dim1)

                        # generate features
                        support_features_64 = feature_encoder(Variable(support_images_64).cuda(GPU)) # 25*64*dim*dim
                        support_features_64 = support_features_64.view(CLASS_NUM,SUPPORT_NUM_PER_CLASS,FEATURE_DIM,dim2,dim2).sum(1)
                        query_features_64 = feature_encoder(Variable(query_images_64).cuda(GPU))
                        
                        # form the QURY_NUM_PER_CLASSxCLASS_NUM relation pairs
                        support_features_ext_64 = support_features_64.unsqueeze(0).repeat(num_per_class*CLASS_NUM,1,1,1,1)
                        support_features_ext_64= support_features_ext_64.view(num_per_class*CLASS_NUM*CLASS_NUM,FEATURE_DIM,dim2,dim2)
                        query_features_ext_64 = query_features_64.unsqueeze(1).repeat(1,CLASS_NUM,1,1,1)
                        query_features_ext_64 = query_features_ext_64.reshape(num_per_class*CLASS_NUM*CLASS_NUM,FEATURE_DIM,dim2,dim2)

                        # generate features
                        support_features_48 = feature_encoder(Variable(support_images_48).cuda(GPU)) # 25*64*dim*dim
                        support_features_48 = support_features_48.view(CLASS_NUM,SUPPORT_NUM_PER_CLASS,FEATURE_DIM,dim3,dim3).sum(1)
                        query_features_48 = feature_encoder(Variable(query_images_48).cuda(GPU))
                        
                        # form the QURY_NUM_PER_CLASSxCLASS_NUM relation pairs
                        support_features_ext_48 = support_features_48.unsqueeze(0).repeat(num_per_class*CLASS_NUM,1,1,1,1)
                        support_features_ext_48= support_features_ext_48.view(num_per_class*CLASS_NUM*CLASS_NUM,FEATURE_DIM,dim3,dim3)
                        query_features_ext_48 = query_features_48.unsqueeze(1).repeat(1,CLASS_NUM,1,1,1)
                        query_features_ext_48 = query_features_ext_48.reshape(num_per_class*CLASS_NUM*CLASS_NUM,FEATURE_DIM,dim3,dim3)

                        scores_84 = kron_rel_net(query_features_ext_84,support_features_ext_84)
                        scores_64 = kron_rel_net(query_features_ext_64,support_features_ext_64)
                        scores_48 = kron_rel_net(query_features_ext_48,support_features_ext_48)
                        
                        scores = scores_84+ scores_64 + scores_48
                         
                        _,predict_labels = torch.max(scores.data,1)
                        rewards = [1 if predict_labels[j]==query_labels[j].cuda(GPU) else 0 for j in range(query_size)]
                        
                        total_rewards += np.sum(rewards)

                    val_accuracy = total_rewards/1.0/CLASS_NUM/15
                    val_accuracies.append(val_accuracy)

                accuracy_val,h = mean_confidence_interval(val_accuracies)
                print("val accuracy :",accuracy_val,"h:",h)
                print("val accuracy :",accuracy_val,"h:",h, file=F_txt)
                feature_encoder_scheduler.step(accuracy_val)
                kron_rel_net_scheduler.step(accuracy_val)
        

        if (episode)%2500 == 0:
            print("Testing...")
            print("Testing...", file=F_txt)
            accuracies = []
            with torch.no_grad():
                for i in range(TEST_EPISODE):
                    total_rewards = 0
                    task = tg.MiniImagenetTask(metatest_folders,CLASS_NUM,SUPPORT_NUM_PER_CLASS,15)
                    support_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=SUPPORT_NUM_PER_CLASS,split="train",shuffle=False)
                    num_per_class = 5
                    query_dataloader = tg.get_mini_imagenet_data_loader(task,num_per_class=num_per_class,split="test",shuffle=False)
                    support_images_84,support_images_64,support_images_48,support_labels = support_dataloader.__iter__().next()
                    for query_images_84,query_images_64,query_images_48,query_labels in query_dataloader:
                        query_size = query_labels.shape[0]
                        
                        # generate features
                        support_features_84 = feature_encoder(Variable(support_images_84).cuda(GPU)) # 25*64*dim*dim
                        support_features_84 = support_features_84.view(CLASS_NUM,SUPPORT_NUM_PER_CLASS,FEATURE_DIM,dim1,dim1).sum(1)
                        query_features_84 = feature_encoder(Variable(query_images_84).cuda(GPU))
                        
                        # form the QURY_NUM_PER_CLASSxCLASS_NUM relation pairs
                        support_features_ext_84 = support_features_84.unsqueeze(0).repeat(num_per_class*CLASS_NUM,1,1,1,1)
                        support_features_ext_84= support_features_ext_84.view(num_per_class*CLASS_NUM*CLASS_NUM,FEATURE_DIM,dim1,dim1)
                        query_features_ext_84 = query_features_84.unsqueeze(1).repeat(1,CLASS_NUM,1,1,1)
                        query_features_ext_84 = query_features_ext_84.reshape(num_per_class*CLASS_NUM*CLASS_NUM,FEATURE_DIM,dim1,dim1)

                        # generate features
                        support_features_64 = feature_encoder(Variable(support_images_64).cuda(GPU)) # 25*64*dim*dim
                        support_features_64 = support_features_64.view(CLASS_NUM,SUPPORT_NUM_PER_CLASS,FEATURE_DIM,dim2,dim2).sum(1)
                        query_features_64 = feature_encoder(Variable(query_images_64).cuda(GPU))
                        
                        # form the QURY_NUM_PER_CLASSxCLASS_NUM relation pairs
                        support_features_ext_64 = support_features_64.unsqueeze(0).repeat(num_per_class*CLASS_NUM,1,1,1,1)
                        support_features_ext_64= support_features_ext_64.view(num_per_class*CLASS_NUM*CLASS_NUM,FEATURE_DIM,dim2,dim2)
                        query_features_ext_64 = query_features_64.unsqueeze(1).repeat(1,CLASS_NUM,1,1,1)
                        query_features_ext_64 = query_features_ext_64.reshape(num_per_class*CLASS_NUM*CLASS_NUM,FEATURE_DIM,dim2,dim2)

                        # generate features
                        support_features_48 = feature_encoder(Variable(support_images_48).cuda(GPU)) # 25*64*dim*dim
                        support_features_48 = support_features_48.view(CLASS_NUM,SUPPORT_NUM_PER_CLASS,FEATURE_DIM,dim3,dim3).sum(1)
                        query_features_48 = feature_encoder(Variable(query_images_48).cuda(GPU))
                        
                        # form the QURY_NUM_PER_CLASSxCLASS_NUM relation pairs
                        support_features_ext_48 = support_features_48.unsqueeze(0).repeat(num_per_class*CLASS_NUM,1,1,1,1)
                        support_features_ext_48= support_features_ext_48.view(num_per_class*CLASS_NUM*CLASS_NUM,FEATURE_DIM,dim3,dim3)
                        query_features_ext_48 = query_features_48.unsqueeze(1).repeat(1,CLASS_NUM,1,1,1)
                        query_features_ext_48 = query_features_ext_48.reshape(num_per_class*CLASS_NUM*CLASS_NUM,FEATURE_DIM,dim3,dim3)


                        scores_84 = kron_rel_net(query_features_ext_84,support_features_ext_84)
                        scores_64 = kron_rel_net(query_features_ext_64,support_features_ext_64)
                        scores_48 = kron_rel_net(query_features_ext_48,support_features_ext_48)
                        
                        scores = scores_84+ scores_64 + scores_48  

                        _,predict_labels = torch.max(scores.data,1)
                        rewards = [1 if predict_labels[j]==query_labels[j].cuda(GPU) else 0 for j in range(query_size)]
                        
                        total_rewards += np.sum(rewards)

                    accuracy = total_rewards/1.0/CLASS_NUM/15
                    accuracies.append(accuracy)

                test_accuracy,h = mean_confidence_interval(accuracies)
                print("test accuracy :",test_accuracy,"h:",h)
                print("best accuracy :",best_accuracy,"h:",best_h)
                print("test accuracy :",test_accuracy,"h:",h, file = F_txt)
                print("best accuracy :",best_accuracy,"h:",best_h, file = F_txt)

                if test_accuracy > best_accuracy:
                    # save networks
                    torch.save(feature_encoder.state_dict(),str(METHOD + "/miniImagenet_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl"))
                    torch.save(kron_rel_net.state_dict(),str(METHOD + "/miniImagenet_kron_rel_net_" + str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl"))
                    torch.save({'best_accuracy': test_accuracy, 'best_h': best_h, 'episode': episode }, METHOD +"/checkpoint_"+ str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl")
                    print("save networks for episode:",episode)
                    print("save networks for episode:",episode, file = F_txt)

                    best_accuracy = test_accuracy
                    best_h = h
                print_training= True
                F_txt.flush()
                os.fsync(F_txt.fileno())
                torch.save({'best_accuracy': test_accuracy, 'best_h': best_h, 'episode': episode }, METHOD +"/checkpoint_"+ str(CLASS_NUM) +"way_" + str(SUPPORT_NUM_PER_CLASS) +"shot.pkl")

    F_txt.close()

if __name__ == '__main__':
    main()
