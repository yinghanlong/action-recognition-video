import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from split_train_test_video import *
from skimage import io, color, exposure
import torch
from torch.utils.data.dataloader import default_collate
from sets import Set
import numpy as np
import json
import csv
import h5py

import os
import os.path

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))



def make_dataset(split_file, split, root, num_classes=65):
    dataset = {}
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    for vid in data.keys():
        video_idx =int(vid.split('_')[2])
        #only read validation video 901-909
        #if video_idx>910 or video_idx<901:
        #    continue
        if data[vid]['subset'] != split:
            continue

        #if not os.path.exists(os.path.join(root, vid)):
            #print 'Video not found at',os.path.join(root, vid)
            #continue
        
        video_fps_125=Set(['video_validation_0000413','video_validation_0000419','video_validation_0000484','video_validation_0000420','video_validation_0000666','video_validation_0000311','video_test_0000950','video_test_0001195','video_test_0001255','video_test_0001459','video_test_0001058'])
        video_fps_12 = Set(['video_test_0001207','video_validation_0000411'])
        fps = 15
        if vid in video_fps_125:
            fps = 12.5
        elif vid in video_fps_12:
            fps = 12
        
        num_frames= int(fps * data[vid]['duration'])
        #num_frames = int(len(os.listdir(os.path.join(root, vid)))) #x,y flows and images are not put in the same folder
        label = np.zeros((num_frames,num_classes), np.float32)

        #fps = num_frames/data[vid]['duration']
        #if fps<14.0:
        #    print(vid,'Warning: fps=',fps)
        for ann in data[vid]['actions']:
            for fr in range(0,num_frames):#,1):
                if fr/fps > ann[1] and fr/fps < ann[2]:#ann[1]:start, ann[2]:end
                    label[fr, ann[0]-1] = 1 # binary classification, class index -1 to make 0 indexed
        dataset[vid]= (label, data[vid]['duration'],num_frames)
        i += 1
    
    return dataset

# make_dataset('multithumos.json', 'training', '/ssd2/thumos/val_i3d_rgb')

class MultiThumos(Dataset):

    def __init__(self, dic, root, mode,transform=None):
 
        self.keys = dic.keys()
        self.values=dic.values()
        #self.split_file = split_file
        #self.batch_size = batch_size
        self.root = root
        self.mode = mode
        self.transform = transform

    def load_image(self,video_name, index):
        
        #path = self.root +video_name+'/img_'+str(index).zfill(5)  
        path = self.root +video_name+'/frame'+str(index).zfill(6)  
        img = Image.open(path +'.jpg')
        transformed_img = self.transform(img)
        img.close()

        return transformed_img

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        #print index,len(self.keys)
        video_name, fr_idx = self.keys[index].split(' ')
        '''
        if entry[0] in self.in_mem:
            feat = self.in_mem[entry[0]]
        else:
            feat = np.load(os.path.join(self.root, entry[0]+'.npy'))
            feat = feat.astype(np.float32)
            self.in_mem[entry[0]] = feat
        '''    
        #print(self.values[index].shape)
        if self.mode=='train': 
            rand_idx = random.randint(0, 9) #***NOTE: add randomness to the training set
            image=self.load_image(video_name,str(rand_idx+int(fr_idx)))
            label = self.values[index][rand_idx,:]#label of this frame
        else:
            image=self.load_image(video_name,fr_idx)
            label = self.values[index]
            clipname = video_name+'-'+str(int(int(fr_idx)/5))

        if self.mode=='train':
            return (image,label)
        else:
            return (clipname, image, label)

    def __len__(self):
        return len(self.keys)


#UCF 101
class spatial_dataset(Dataset):  
    def __init__(self, dic, root_dir, mode, transform=None):
 
        self.keys = dic.keys()
        self.values=dic.values()
        self.root_dir = root_dir
        self.mode =mode
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def load_ucf_image(self,video_name, index):
        if video_name.split('_')[0] == 'HandstandPushups':
            n,g = video_name.split('_',1)
            name = 'HandstandPushups_'+g
            path = self.root_dir + 'v_'+name+'/frame'+str(index).zfill(6)  
        else:
            path = self.root_dir + 'v_'+video_name+'/frame'+str(index).zfill(6)  
        
        img = Image.open(path +'.jpg')
        transformed_img = self.transform(img)
        #print 'img shape=',transformed_img.shape=3,224,224
        img.close()

        return transformed_img

    def __getitem__(self, idx):

        if self.mode == 'train':
            video_name, nb_clips = self.keys[idx].split(' ')
            nb_clips = int(nb_clips)
            #Yinghan:three randomly selected images are taken from the video
            #need to sample more frames
            clips = []
            
            #for i in range(int(nb_clips/10))
            #    clips.append(10*i)
            clips.append(random.randint(1, nb_clips/3))
            clips.append(random.randint(nb_clips/3, nb_clips*2/3))
            clips.append(random.randint(nb_clips*2/3, nb_clips+1))
            
        elif self.mode == 'val':
            video_name, index = self.keys[idx].split(' ')
            index =abs(int(index))
        else:
            raise ValueError('There are only train and val mode')

        label = self.values[idx]
        label = int(label)-1
        
        if self.mode=='train':
            data ={}
            for i in range(len(clips)):
                key = 'img'+str(i)
                index = clips[i]
                data[key] = self.load_ucf_image(video_name, index)
            #print 'img shape=',data[key].shape
            sample = (data, label)
        elif self.mode=='val':
            data = self.load_ucf_image(video_name,index)
            clipname = video_name+'-'+str(int(index/10))
            sample = (clipname, data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class spatial_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, path, ucf_list, ucf_split,dataset='ucf'):
        self.dataset = dataset
        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.data_path=path
        self.frame_count ={}
        #self.val_len={}
        # split the training and testing videos
        print 'dataset=',dataset
        if self.dataset=='ucf':
            splitter = UCF101_splitter(path=ucf_list,split=ucf_split)
            self.train_video, self.test_video = splitter.split_video()        
        else:#multithumos
            self.train_video = make_dataset(ucf_list, 'training', path)
            #self.test_video = make_dataset(ucf_list, 'training', path)
            self.test_video= make_dataset(ucf_list, 'testing', '/media/yinghan/dataset/TH14_test/rgb')#TODO Using same training/testing set now. Change to 'testing' later

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
        self.val_sample20()

        train_loader = None #self.train()
        val_loader = None #self.validate()

        return train_loader, val_loader, self.test_video, self.train_video,self.frame_count


    def get_training_dic(self):
        #print '==> Generate frame numbers of each training video'
        if self.dataset=='ucf':
            self.dic_training={}
            for video in self.train_video:
                nb_frame = self.frame_count[video]-10+1
                key = video+' '+ str(nb_frame)
                self.dic_training[key] = self.train_video[video]
        else:
            self.dic_training={}
            for video in self.train_video:
                nb_frame = self.train_video[video][2]-10+1
                sample_num = int(nb_frame/5)
                self.frame_count[video]= self.train_video[video][2]
                for i in range(sample_num):
                    frame = i*5
                    key = video+ ' '+str(frame+1)
                    self.dic_training[key] = self.train_video[video][0][frame,:] #TODO: for lstm
                    #self.dic_training[key] = self.train_video[video][0][frame:frame+10,:] #binary classification of every class

     #change to sample every 10 frames               
    def val_sample20(self):
        print '==> sampling testing frames'
        self.dic_testing={}

        if self.dataset=='ucf':
            for video in self.test_video:
                nb_frame = self.frame_count[video]-10+1
                sample_num = int(nb_frame/10)#int(nb_frame/19)
                #self.val_len[video] = sample_num 
                for i in range(sample_num):#range(19):
                    frame = i*10 #i*interval
                    key = video+ ' '+str(frame+1)
                    self.dic_testing[key] = self.test_video[video]     
        else:
            for video in self.test_video:
                interval = 5 #TODO:set to 5
                nb_frame = self.test_video[video][2]- interval +1
                sample_num = int(nb_frame/interval)
                self.frame_count[video]= self.test_video[video][2]
                #self.val_len[video] = sample_num 
                #print sample_num
                for i in range(sample_num):
                    frame = i*interval #+ random.randint(0, 9)
                    key = video+ ' '+str(frame+1)
                    self.dic_testing[key] = self.test_video[video][0][frame,:]   

    def train(self):
        if self.dataset=='ucf':        
            training_set = spatial_dataset(dic=self.dic_training, root_dir=self.data_path, mode='train', transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        else:
            training_set = MultiThumos(dic=self.dic_training, root=self.data_path, mode='train',transform = transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))

        print '==> Training data :',len(training_set),'frames'
        #print len(training_set[1][0])

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)
        return train_loader

    def validate(self):
        if self.dataset=='ucf':  
            validation_set = spatial_dataset(dic=self.dic_testing, root_dir=self.data_path, mode='val', transform = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        else:
            val_root='/media/yinghan/dataset/TH14_test/rgb/'
            validation_set = MultiThumos(dic=self.dic_testing, root=val_root, mode='val',transform = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        
        print '==> Validation data :',len(validation_set),'frames'
        #print validation_set[1][1].size()

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader





if __name__ == '__main__':
    
    dataloader = spatial_dataloader(BATCH_SIZE=1, num_workers=1, 
                                path='/home/ubuntu/data/UCF101/spatial_no_sampled/', 
                                ucf_list='/home/ubuntu/cvlab/pytorch/ucf101_two_stream/github/UCF_list/',
                                ucf_split='01',
				dataset = 'ucf')
    train_loader,val_loader,test_video = dataloader.run()
