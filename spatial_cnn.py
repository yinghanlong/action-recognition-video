import numpy as np
import pickle
import os
from PIL import Image
import time
from tqdm import tqdm
import shutil
from random import randint
import argparse

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import MultiStepLR

import dataloader
from utils import *
from network import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='UCF101 spatial stream on resnet101')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=25, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--extract', dest='extract', action='store_true', help='extract features')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--split', default='01', type=str, metavar='split_list', help='choose train/test list')
parser.add_argument('--dataset', default='ucf', type=str, metavar='dataset', help='Dataset')

def main():
    global arg
    arg = parser.parse_args()
    print arg

    if arg.dataset=='ucf':
        data_path='/home/yinghan/Documents/two-stream/jpegs_256/'
        list_path ='/home/yinghan/Documents/two-stream/UCF_list/'
    else:
        data_path='/home/yinghan/Downloads/TH14_validation_set/rgb/'
        list_path ='/home/yinghan/Documents/super-events/data/multithumos.json'
        
    

    #Prepare DataLoader
    data_loader = dataloader.spatial_dataloader(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=8,
                        path= data_path,
                        ucf_list =list_path,
                        ucf_split =arg.split, 
                        dataset = arg.dataset
                        )
    
    train_loader, test_loader, test_video, train_video,frame_count = data_loader.run()
    #Model 
    model = Spatial_CNN(
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        resume=arg.resume,
                        start_epoch=arg.start_epoch,
                        evaluate=arg.evaluate,
                        extract=arg.extract,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        test_video=test_video,
                        ucf_split = arg.split,
                        dataset = arg.dataset
    )
    #Training
    model.run()

class Spatial_CNN():
    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, train_loader, test_loader, test_video,ucf_split,dataset,extract):
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.batch_size=batch_size
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        self.extract=extract
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.best_prec1=0
        self.test_video=test_video
        self.ucf_split = ucf_split
        self.dataset = dataset

    def build_model(self):
        print ('==> Build model and setup loss and optimizer')
        #build model
        self.model = resnet101(pretrained= True, channel=3).cuda()

        #Loss function and optimizer
        if self.dataset=='ucf':
            self.criterion = nn.CrossEntropyLoss().cuda()
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=2,verbose=True)
        else:
            self.criterion = nn.BCEWithLogitsLoss().cuda()
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5,verbose=True)
            #self.scheduler = MultiStepLR(self.optimizer, [300, 800])
    
    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
                  .format(self.resume, checkpoint['epoch'], self.best_prec1))
                #manually set learning rate if the stored lr is too small
                for g in self.optimizer.param_groups:
                    g['lr'] = self.lr
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        if self.evaluate:
            self.epoch = 0
            self.nb_epochs = 0
            #self.extract_features('test')
            #self.extract_features('train')
            prec1, val_loss = self.validate_1epoch()
            return        
        if self.extract:
            self.epoch = 0
            self.nb_epochs = 0
            self.extract_features('test')
            #self.extract_features('train')
            #prec1, val_loss = self.validate_1epoch()
            return

    def run(self):
        self.build_model()
        #fine tuning on multithumos
        if self.dataset!='ucf':
            self.model.fc_custom=nn.Linear(512*4 , 65).cuda()
        self.resume_and_evaluate()
        #fine tuning on multithumos
        #if self.dataset!='ucf':
        #    self.model.fc_custom=nn.Linear(512*4 , 65).cuda()
        cudnn.benchmark = True
        
        for self.epoch in range(self.start_epoch, self.nb_epochs):
            self.train_1epoch()
            prec1, val_loss = self.validate_1epoch()
            is_best = val_loss< self.best_prec1 #prec1 > self.best_prec1
            #lr_scheduler
            self.scheduler.step(val_loss)
            # save model
            if is_best:
                self.best_prec1 = val_loss#prec1
                print('Saving model')
                with open('record/spatial/spatial_video_preds.pickle','wb') as f:
                    pickle.dump(self.dic_video_level_preds,f)
                f.close()
            
                save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer' : self.optimizer.state_dict()
                },is_best,'record/spatial/checkpoint'+self.dataset+'.pth.tar','record/spatial/'+self.dataset+'-rand.pth.tar')

    def train_1epoch(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        #switch to train mode
        self.model.train()    
        end = time.time()
        # mini-batch training
        progress = tqdm(self.train_loader)
        for i, (data_dict,label) in enumerate(progress):

    
            # measure data loading time
            data_time.update(time.time() - end)
            
            label = label.cuda(async=True)
            target_var = Variable(label).cuda()

            # compute output
            if self.dataset=='ucf':
                output = Variable(torch.zeros(len(data_dict['img1']),101).float()).cuda()
                #print 'data_dict shape=',len(data_dict)
                for i in range(len(data_dict)):
                    key = 'img'+str(i)
                    data = data_dict[key]
                    input_var = Variable(data).cuda()
                    #sum up the results based on three sampled frames
                    output += self.model(input_var)
            else:
                data_var = Variable(data_dict).cuda(async=True)            
                output = self.model(data_var)
                #loss = F.binary_cross_entropy_with_logits(output,target_var, size_average=False)

            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            
            if self.dataset=='ucf':
                prec1, prec5 = accuracy(output.data, label, topk=(1, 5))
                top1.update(prec1, data.size(0))
                top5.update(prec5, data.size(0))
            losses.update(loss.data, data_dict.size(0))
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Loss':[round(losses.avg,5)],
                'Prec@1':[round(top1.avg,4)],
                'Prec@5':[round(top5.avg,4)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        record_info(info, 'record/spatial/rgb_train.csv','train')

    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        video_top1=0
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds={}
        frame_level_preds={}
        end = time.time()
        progress = tqdm(self.test_loader)
        for i, (keys,data,label) in enumerate(progress):
            
            label = label.cuda(async=True)
            data_var = Variable(data, volatile=True).cuda(async=True)
            label_var = Variable(label, volatile=True).cuda(async=True)

            # compute output
            output = self.model(data_var)
            
            loss = self.criterion(output, label_var)
            losses.update(loss.data, data.size(0))    

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            #Calculate video level prediction

            if self.dataset=='ucf':
                preds = output.data.cpu().numpy()
                nb_data = preds.shape[0]
                for j in range(nb_data):
                    videoName,clip_idx = keys[j].split('-')
                    if videoName not in self.dic_video_level_preds.keys():
                        self.dic_video_level_preds[videoName] = preds[j,:]
                    else:
                        self.dic_video_level_preds[videoName] += preds[j,:]
                    frame_level_preds[keys[j]] = preds[j,:]

        if self.dataset=='ucf':                
            video_top1, video_top5, video_loss = self.frame2_video_level_accuracy()
        #else:
            #apm.add(torch.sigmoid(frame_level_preds), labels)

        with open('record/spatial/'+self.dataset+'_video_preds.pickle','wb') as f:
            pickle.dump(frame_level_preds,f)
        f.close()            

        info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Loss':[round(losses.avg,5)],
                'Prec@1':[round(top1.avg,3)],
                'Prec@5':[round(top5.avg,3)]}
        record_info(info, 'record/spatial/rgb_test.csv','test')
        return top1.avg, losses.avg#video_loss

    def extract_features(self,loader):
        print('==> Epoch:[{0}/{1}][extract features]'.format(self.epoch, self.nb_epochs))
        #remove the last layer
        modules = list(self.model.children())[:-1]
        self.model = nn.Sequential(*modules)
        for p in self.model.parameters():
            p.requires_grad = False       

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds={}
        frame_level_preds={}
        end = time.time()
        if loader=='test':
            progress = tqdm(self.test_loader)
        else:
            progress = tqdm(self.train_loader)
        for i, (keys,data,label) in enumerate(progress):
            #print keys
            label = label.cuda(async=True)
            data_var = Variable(data, volatile=True).cuda(async=True)
            label_var = Variable(label, volatile=True).cuda(async=True)

            # compute output
            output = self.model(data_var)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            #Calculate video level prediction
            preds = output.data.cpu().numpy()
            nb_data = preds.shape[0]
            for j in range(nb_data):
                videoName,clip_idx = keys[j].split('-')
                #if os.path.exists(os.path.join('/media/yinghan/dataset/',self.dataset,'/',videoName,'/clip',clip_idx+'.npy')):
                #    continue
                #np.save(os.path.join('/media/yinghan/dataset/',self.dataset,'/',videoName,'/',clip_idx),preds[j,:])
                #if videoName not in self.dic_video_level_preds.keys():
                #    self.dic_video_level_preds[videoName] = preds[j,:]
                #else:
                #    self.dic_video_level_preds[videoName] += preds[j,:]
                frame_level_preds[keys[j]] = preds[j,:]
        #video_top1, video_top5, video_loss = self.frame2_video_level_accuracy()

        with open('record/spatial/'+self.dataset+'_validation.pickle','wb') as f:
            pickle.dump(frame_level_preds,f)
        f.close()            

        #info = {'Epoch':[self.epoch],
        #        'Batch Time':[round(batch_time.avg,3)],
        #        'Loss':[round(video_loss,5)],
        #        'Prec@1':[round(video_top1,3)],
        #        'Prec@5':[round(video_top5,3)]}
        #record_info(info, 'record/spatial/rgb_test.csv','test')
        #return video_top1, video_loss

    def frame2_video_level_accuracy(self):
            
        correct = 0
        video_level_preds = np.zeros((len(self.dic_video_level_preds),101))
        video_level_labels = np.zeros(len(self.dic_video_level_preds))
        ii=0
        for name in sorted(self.dic_video_level_preds.keys()):
        
            preds = self.dic_video_level_preds[name]
            label = int(self.test_video[name])-1
                
            video_level_preds[ii,:] = preds
            video_level_labels[ii] = label
            ii+=1         
            if np.argmax(preds) == (label):
                correct+=1

        #top1 top5
        video_level_labels = torch.from_numpy(video_level_labels).long()
        video_level_preds = torch.from_numpy(video_level_preds).float()
            
        top1,top5 = accuracy(video_level_preds, video_level_labels, topk=(1,5))
        loss = self.criterion(Variable(video_level_preds).cuda(), Variable(video_level_labels).cuda())     
                            
        top1 = float(top1.numpy())
        top5 = float(top5.numpy())
            
        #print(' * Video level Prec@1 {top1:.3f}, Video level Prec@5 {top5:.3f}'.format(top1=top1, top5=top5))
        return top1,top5,loss.data.cpu().numpy()







if __name__=='__main__':
    main()
