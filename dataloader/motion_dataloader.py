import numpy as np
import pickle
from PIL import Image
import time
import shutil
import random
import argparse

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from split_train_test_video import *

import json
import csv
import h5py

 
def make_dataset(split_file, split, root, num_classes=65):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    for vid in data.keys():
        video_idx =int(vid.split('_')[2])
        #only read validation 901-909
        #if video_idx>910 or video_idx<901:
        #    continue
        if data[vid]['subset'] != split:
            continue

        if not os.path.exists(os.path.join(root,'x/', vid)):
            continue
        
        num_frames = int(len(os.listdir(os.path.join(root,'x/', vid,)))) #x,y flows and images are not put in the same folder
        label = np.zeros((num_frames,num_classes), np.float32)

        fps = num_frames/data[vid]['duration']
        for ann in data[vid]['actions']:
            for fr in range(0,num_frames):#,1):
                if fr/fps > ann[1] and fr/fps < ann[2]:#ann[1]:start, ann[2]:end
                    label[fr, ann[0]-1] = 1 # binary classification, class index -1 to make 0 indexed
        dataset.append((vid, label, data[vid]['duration'],num_frames))
        i += 1
    
    return dataset

# make_dataset('multithumos.json', 'training', '/ssd2/thumos/val_i3d_rgb')

class Thumos_flow(Dataset):

    def __init__(self, dic, root, mode,in_channel=10,transform=None):
 
        self.keys = dic.keys()
        self.values=dic.values()
        #self.split_file = split_file
        #self.batch_size = batch_size
        self.root = root
        self.transform = transform
        self.in_channel = in_channel
        self.img_rows=224
        self.img_cols=224
	self.mode = mode

    def load_flow(self,video_name, index):
        flow = torch.FloatTensor(2*self.in_channel,self.img_rows,self.img_cols)
        i = int(index)-5 #center at index

        for j in range(self.in_channel):
            idx = i + j
            idx = str(idx)
            h_image = self.root +'x/'+video_name+'/frame'+str(index).zfill(6)+'.jpg'   
            v_image = self.root +'y/'+video_name+'/frame'+str(index).zfill(6)+'.jpg'  
        
            imgH=(Image.open(h_image))
            imgV=(Image.open(v_image))

            H = self.transform(imgH)
            V = self.transform(imgV)

            flow[2*(j-1),:,:] = H
            flow[2*(j-1)+1,:,:] = V      
            imgH.close()
            imgV.close()  
        return flow

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        video_name, fr_idx = self.keys[idx].split(' ')
        #entry = self.data[index]
        
        
        if self.mode=='train': 
            rand_idx = random.randint(0, 9) #***NOTE: add randomness to the training set
            flow=self.load_flow(video_name,str(rand_idx+int(fr_idx)))
            label = self.values[index][rand_idx,:]#label of this frame
        else:
            flow=self.load_flow(video_name,fr_idx)
            label = self.values[index]
            clipname = video_name+'-'+str(int(int(fr_idx)/5))

        if self.mode=='train':
            return (flow,label)
        else:
            return (clipname, flow, label)

    def __len__(self):
        return len(self.keys)


class motion_dataset(Dataset):  
    def __init__(self, dic, in_channel, root_dir, mode, transform=None):
        #Generate a 16 Frame clip  #Note: 10 frame, no 16
        self.keys=dic.keys()
        self.values=dic.values()
        self.root_dir = root_dir
        self.transform = transform
        self.mode=mode
        self.in_channel = in_channel
        self.img_rows=224
        self.img_cols=224

    def stackopf(self):
        name = 'v_'+self.video
        u = self.root_dir+ 'u/' + name
        v = self.root_dir+ 'v/'+ name
        
        flow = torch.FloatTensor(2*self.in_channel,self.img_rows,self.img_cols)
        i = int(self.clips_idx)


        for j in range(self.in_channel):
            idx = i + j
            idx = str(idx)
            frame_idx = 'frame'+ idx.zfill(6)
            h_image = u +'/' + frame_idx +'.jpg'
            v_image = v +'/' + frame_idx +'.jpg'
            
            imgH=(Image.open(h_image))
            imgV=(Image.open(v_image))

            H = self.transform(imgH)
            V = self.transform(imgV)

            
            flow[2*(j-1),:,:] = H
            flow[2*(j-1)+1,:,:] = V      
            imgH.close()
            imgV.close()  
        return flow

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        #print ('mode:',self.mode,'calling Dataset:__getitem__ @ idx=%d'%idx)
        
        if self.mode == 'train':

            self.video,self.clips_idx = self.keys[idx].split('-')
            #self.video, nb_clips = self.keys[idx].split('-')
            #self.clips_idx = random.randint(1,int(nb_clips))#take a random clip from the video
        elif self.mode == 'val':
            self.video,self.clips_idx = self.keys[idx].split('-')
        else:
            raise ValueError('There are only train and val mode')

        label = self.values[idx]
        label = int(label)-1 
        data = self.stackopf()
        clipname = self.video +'-' + str(int(int(self.clips_idx)/10))

        if self.mode == 'train':
            sample = (clipname,data,label)
        elif self.mode == 'val':
            sample = (clipname,data,label)
            #sample = (self.video,data,label)
        else:
            raise ValueError('There are only train and val mode')
        return sample


class Motion_DataLoader():
    def __init__(self, BATCH_SIZE, num_workers, in_channel,  path, ucf_list, ucf_split,dataset='ucf'):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        self.frame_count={}
        self.in_channel = in_channel
        self.data_path=path
        self.dataset =dataset
        # split the training and testing videos        
        if dataset=='ucf':
            splitter = UCF101_splitter(path=ucf_list,split=ucf_split)
            self.train_video, self.test_video = splitter.split_video()        
        else:#multithumos
            self.train_video = make_dataset(ucf_list, 'training', path)
            #self.test_video = make_dataset(ucf_list, 'training', path)
            self.test_video= make_dataset(ucf_list,  'testing', '/media/yinghan/TOSHIBA EXT/TH14_test_set/tvl1_flow')#TODO:Using same training/testing set(basketball videos- validation 901-910) now. Change to 'testing' later
        
    def load_frame_count(self):
        #print '==> Loading frame number of each video'

        if self.dataset=='ucf':
            with open('dataloader/dic/frame_count.pickle','rb') as file:
                dic_frame = pickle.load(file)
            file.close()

            for line in dic_frame :
                videoname = line.split('_',1)[1].split('.',1)[0]
                n,g = videoname.split('_',1)
                if n == 'HandStandPushups':
                    videoname = 'HandstandPushups_'+ g
                self.frame_count[videoname]=dic_frame[line] 

    def run(self):
        self.load_frame_count()
        self.get_training_dic()
        self.val_sample()
        train_loader = self.train()
        val_loader = self.val()

        return train_loader, val_loader, self.test_video
            
    def val_sample(self):
	#generate a sequence of representation of the clips
        self.dic_test_idx = {}
        if self.dataset=='ucf':
            for video in self.test_video:
                n,g = video.split('_',1)
                sampling_interval = int(self.in_channel) 
                sampling_num = int((self.frame_count[video]-10+1)/sampling_interval)
                #sampling_interval = int((self.frame_count[video]-10+1)/19)
                for index in range(sampling_num):
                    clip_idx = index*sampling_interval
                    key = video + '-' + str(clip_idx+1)
                    self.dic_test_idx[key] = self.test_video[video]
        else:
            for video,label,duration,nb_frame in self.test_video:
                interval = 5
                nb_frame = nb_frame - 10+1 
                sample_num = int(nb_frame/interval)
                for i in range(1,sample_num):
                    frame = i*interval
                    key = video+ ' '+str(frame+1)
                    self.dic_test_idx[key] = label[frame,:]  
             
    def get_training_dic(self):
        self.dic_video_train={}
        if self.dataset=='ucf':
            for video in self.train_video:
                sampling_interval = int(self.in_channel) 
                sampling_num = int((self.frame_count[video]-10+1)/sampling_interval)
            #sampling_interval = int((self.frame_count[video]-10+1)/19)
                for index in range(sampling_num):
                    clip_idx = index*sampling_interval
                    key = video + '-' + str(clip_idx+1)
                    self.dic_video_train[key] = self.train_video[video]
        else:
            for video,label,duration,nb_frame in self.train_video:
                interval = 10
                nb_frame = nb_frame- 10+1# 10 channels
                sample_num = int(nb_frame/interval)
                for i in range(1,sample_num):
                    frame = i*interval
                    key = video+ ' '+str(frame+1)
                    self.dic_video_train[key] = label[frame:frame+10,:] #binary classification of every class
            #nb_clips = self.frame_count[video]-10+1
            #key = video +'-' + str(nb_clips)
            #self.dic_video_train[key] = self.train_video[video] 
                            
    def train(self):
        if self.dataset=='ucf': 
            training_set = motion_dataset(dic=self.dic_video_train, in_channel=self.in_channel, root_dir=self.data_path,
            mode='train',
            transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            ]))
        else:
            training_set = Thumos_flow(dic=self.dic_video_train, root=self.data_path, mode='train',transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            ]))
        print '==> Training data :',len(training_set)#,' videos',training_set[1][1].size()

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
            )

        return train_loader

    def val(self):
        if self.dataset=='ucf': 
            validation_set = motion_dataset(dic= self.dic_test_idx, in_channel=self.in_channel, root_dir=self.data_path ,
            mode ='val',
            transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            ]))
        else:#TODO: change to test set '/media/yinghan/TOSHIBA EXT/TH14_test_set/tvl1_flow/'
            val_root = '/media/yinghan/TOSHIBA EXT/TH14_test_set/tvl1_flow/'
            validation_set = Thumos_flow(dic=self.dic_test_idx, root=val_root, mode ='val',transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            ]))
        print '==> Validation data :',len(validation_set)#,' frames',validation_set[1][1].size()
        #print validation_set[1]

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)

        return val_loader

if __name__ == '__main__':
    data_loader =Motion_DataLoader(BATCH_SIZE=1,num_workers=1,in_channel=10,
                                        path='/home/ubuntu/data/UCF101/tvl1_flow/',
                                        ucf_list='/home/ubuntu/cvlab/pytorch/ucf101_two_stream/github/UCF_list/',
                                        ucf_split='01'
                                        )
    train_loader,val_loader,test_video = data_loader.run()
    #print train_loader,val_loader
