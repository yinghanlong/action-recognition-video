# -*- coding: utf-8 -*-
#LSTM for action recognition

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

#from matplotlib import pyplot as plt
import pickle
import numpy as np
from utils import *
import dataloader
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models


import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import MultiStepLR

from utils import *
from network import *
from PIL import Image
import time
import tqdm
import shutil
from random import randint
import argparse
import random

from apmeter import APMeter
#Set GPU Device number
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='UCF101 motion stream on resnet101')
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--lr', default=1e-2, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--top5enhance', dest='top5enhance', action='store_true', help='enhance the result by top5 from previous frames')

parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
#Multi-InOut LSTM
parser.add_argument('--multi', default=1, type=int, metavar='MultiN', help='Use Multi-LSTM')
parser.add_argument('--split', default='01', type=str, metavar='split_list', help='choose train/test list')
parser.add_argument('--dataset', default='ucf', type=str, metavar='dataset', help='Dataset')

#frame resized to 224*224
#Input to CNN: video clips size = (8,20,224,224) =(batch_size,num_frames,height,width) 
#Original setting: for spatial cnn(rgb), num_frames=3,channel=3; for motion cnn(flow), channel=20
#Current setting: for spatial cnn(rgb), num_frames=10; for motion cnn(flow), num_clips=10,channel=20
#Output = 10 * 101 (num_clips * num_classes)


######################################################################
# Create the model:
class LSTM_model(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layer,num_classes,batch=1,enhance=0):
        super(LSTM_model, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layer = num_layer
        self.batch = batch
        self.enhance = enhance
        #build lstm model
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layer).cuda()
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, num_classes).cuda()#Output layer
        self.outmap = nn.Linear(6, num_classes,bias=False).cuda()
        #self.updatez = nn.Linear(num_classes, num_classes,bias=False).cuda()# z_i = g(f(y_i-1),z_i-1)
        self.updatez = nn.Linear(input_dim, input_dim,bias=False).cuda()
        self.updatex1 = nn.Linear(input_dim, input_dim,bias=False).cuda()
        self.updatex2 = nn.Linear(input_dim, input_dim,bias=False).cuda()
        #self.combine = nn.Linear(input_dim, input_dim,bias=False).cuda()
        self.combine = nn.Linear(2*num_classes, num_classes).cuda()
        self.hidden = self.init_hidden() #hidden states
        self.init_weight()

    def init_weight(self):
        #initialize weights to zero
        torch.nn.init.constant_(self.updatez.weight.data,0)
        #torch.nn.init.constant_(self.combine.weight.data,0)
        torch.nn.init.constant_(self.outmap.weight.data,0)

        #self.updatef1 = nn.Linear(input_dim, 1,bias=False).cuda()
        #self.updatef2 = nn.Linear(input_dim, 1,bias=False).cuda()
        #torch.nn.init.constant_(self.updatex1.weight.data,0.1)
        #torch.nn.init.constant_(self.updatex2.weight.data,0.1)
        #torch.nn.init.eye_(self.updatex1.weight.data)
        #torch.nn.init.eye_(self.updatex2.weight.data)
        #print('Initial weights:',self.outmap.weight.data)

    def init_hidden(self):#h0, c0 = h0.cuda(), c0.cuda()
        # (h0,c0)
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(self.num_layer, self.batch, self.hidden_dim).cuda(),
                torch.zeros(self.num_layer, self.batch, self.hidden_dim).cuda())

    def forward(self, video):#
        #input dimension: clip_idx,batch_size,input_dim
        length = video.shape[0]
        #adaptive length
        enhance_capa =20 #(int(length/10)+1)*5 #TODO: Multithumos=20, UCF=10
        enhance_cnt = 0
        TOPK = 2 #TODO: define significant as topk change. Multithumos=2, UCF=5
        #initialize
        if self.enhance:
            enhance_array= torch.zeros(enhance_capa,dtype=torch.int16)#index of x_significant
            wxk = []#torch.zeros(len(video), self.input_dim, dtype=torch.float32,requires_grad=True).cuda()
            gxk = []
            wxk.append(torch.zeros(self.input_dim).cuda())# input 0
            gxk.append(torch.zeros(self.input_dim).cuda())

        input_lstm = [] #torch.zeros(len(video),self.batch,self.input_dim).cuda()
        #lstm_out = {}#torch.zeros(len(video),self.batch,self.hidden_dim).cuda()
        tag_space = torch.zeros(len(video),self.num_classes).cuda()
        tag_scores = torch.zeros(len(video),self.num_classes).cuda()
        #24 stand, 25 run, 26 jump, 36 throw, 44 squat, 63 talktocamera
        general_list = [23,24,25,35,43,62]
        #init hidden state
        self.hidden2 = self.init_hidden() 
        for i in range(len(video)):
            #generate inputs
            if self.enhance and enhance_cnt>=enhance_capa:
            #if self.enhance and enhance_cnt>4: #UCF101
                wx =self.updatex1(video[i,:,:].view(self.input_dim)).tanh()#video[i,:,:].view(self.input_dim)
                f= torch.zeros([enhance_capa],dtype=torch.float32,requires_grad=True).cuda()
                g= []
                for j in range(0,enhance_capa):
                    g.append(gxk[enhance_array[j]])
                    f[j]=torch.dot(wx,wxk[enhance_array[j]]) # (wxi)T(wxk): dot-product similarity scalar
                
                C= torch.sum(f)#.abs()
                #print("correlation",f)
                if C==0:
                    C= 1 #Avoid divide by zero
                zi = torch.mul(g[0],f[0])
                for j in range(1,enhance_capa):
                    zi = zi + torch.mul(g[j],f[j])
        	#Original input + zi
                #input_xp = video[i,:,:].view(self.input_dim) +self.combine(zi)
                input_xp = video[i,:,:].view(self.input_dim) + zi/C
                #Concatenation:
                #input_xp=torch.cat((video[i,:,:].view(video.shape[2]),zi),0)
                input_lstm.append(input_xp.cuda())
            else:
                input_lstm.append( video[i,:,:].cuda())

            #LSTM
            if i%2==0 or self.training==False:#if evaluating, do not separate
                lstm_out, self.hidden = self.lstm(input_lstm[i].view(1, 1, -1), self.hidden)
            else:
                lstm_out, self.hidden2 = self.lstm(input_lstm[i].view(1, 1, -1), self.hidden2)
            
            output= self.hidden2tag(lstm_out.view(1,-1))
            tag_space[i,:]= output

            #TODO: hierarchical classifier            
            if i>=10:#self.enhance and enhance_cnt>=enhance_capa:
                #avg_out = torch.mul(tag_space[i-20,:],1.0).view(1,-1)
                #for j in range(1,enhance_capa):
                #    avg_out = avg_out + torch.mul(tag_space[i-20+j,:],f[j]/C).view(1,-1)
                
                avg_out= torch.mean(tag_space[i-10:i,:],dim=0,keepdim=True).cuda()
                general_out = torch.cat((avg_out[:,23:26].view(-1),avg_out[:,35],avg_out[:,43],avg_out[:,62]))#torch.zeros((6)).cuda()
                #24 stand, 25 run, 26 jump, 36 throw, 44 squat, 63 talktocamera
                aux_out=self.outmap(general_out.view(1,-1).tanh())
                tag_space[i,:]= tag_space[i,:] + aux_out

            #print tag_space[i,:].size() 
            tag_scores[i,:] = F.softmax(output,dim=1)
            
            if self.enhance:
                pred_value, pred_idx = tag_scores[i,:].data.topk(TOPK, 0, True, True)#tag_scores[i,:].data.topk(5, 0, True, True)

                if i>0:
                #update the enhance_array if the prediction changed with the last input
                  if (torch.all(torch.eq(prev_pred,pred_idx))!=1):
                    for j in range(0,enhance_capa-1):
                        enhance_array[j] = enhance_array[j+1]   #shift to left
                    enhance_array[enhance_capa-1] = i
                    enhance_cnt+=1
                    wxk.append(self.updatex2(video[i,:,:].view(self.input_dim)).tanh())#non-linear
                    gxk.append(self.updatez(video[i,:,:].view(self.input_dim)).tanh())
                    #print("Idx:",enhance_array)
                  else:
                    wxk.append(torch.zeros(self.input_dim).cuda())
                    gxk.append(torch.zeros(self.input_dim).cuda())


                prev_pred = pred_idx #keep the previous prediction result for comparison
                

        '''
        lstm_out, self.hidden = self.lstm(video, self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(video), -1))

        #***NOTE: CrossentropyLoss= log_softmax+nllloss
        #tag_scores = F.log_softmax(tag_space, dim=1)
        #tag_scores = tag_scores.view(self.batch,len(video), -1)
        #lstm_out.size(), tag_scores.size()
        '''
        tag_space =tag_space.t() 
        tag_space =tag_space.view(1,self.num_classes,-1)
        
        return tag_space #(batch_size, num_classes,seq_length)


class LSTM():

    def __init__(self, nb_epochs, lr, input_dim, hidden_dim, num_layer,num_classes,multi,evaluate,enhance,
train_data,train_labels,validation_data,validation_labels,resume,batch=1,dataset='ucf'):
        self.multi=multi
        self.hidden_dim = hidden_dim
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.evaluate=evaluate
        self.train_data=train_data
        self.train_labels=train_labels
        self.validation_data=validation_data
        self.validation_labels=validation_labels
        self.best_prec1=0
        self.num_classes = num_classes
        self.batch = batch
        self.resume =resume
        #init model
        self.enhance = enhance
        self.model = LSTM_model(input_dim, hidden_dim, num_layer,num_classes,batch,enhance)
        self.dataset = dataset
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        if dataset=='ucf':
            self.criterion = nn.CrossEntropyLoss().cuda()
            lr_patience = 2
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=lr_patience,verbose=True)
        else:
            self.criterion = nn.BCEWithLogitsLoss().cuda()
            lr_patience = 10
            self.scheduler = MultiStepLR(self.optimizer, [30])
            # Combines sigmoid with F.binary_cross_entropy_with_logits().cuda()


    def run(self):
        #self.build_model()
        self.enhance_array = {}
        self.resume_and_evaluate()
        cudnn.benchmark = True
        
        for self.epoch in range(self.nb_epochs):
            self.train_1epoch()
            prec1, val_loss = self.validate_1epoch()
            is_best = prec1 > self.best_prec1
            #lr_scheduler
            if self.dataset=='ucf':
            	self.scheduler.step(val_loss)
            else:
            	self.scheduler.step()
            # save model
            if is_best:
                self.best_prec1 = prec1
                #with open('record/lstm/lstm_video_preds.pickle','wb') as f:
                #    pickle.dump(self.dic_video_level_preds,f)
                #f.close() 
            	if self.enhance:
                    save_checkpoint({
                    'epoch': self.epoch,
                    'state_dict': self.model.state_dict(),
                    'best_prec1': self.best_prec1,
                    'optimizer' : self.optimizer.state_dict()
                    },is_best,'record/checkpoint.pth.tar','record/'+self.dataset+'_enhance.pth.tar')
                else:
                    save_checkpoint({
                    'epoch': self.epoch,
                    'state_dict': self.model.state_dict(),
                    'best_prec1': self.best_prec1,
                    'optimizer' : self.optimizer.state_dict()
                    },is_best,'record/checkpoint.pth.tar','record/'+self.dataset+'_best.pth.tar')
    
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
            self.nb_epochs = 0
            self.epoch = 0
            prec1, val_loss = self.validate_1epoch()
            return

    def train_1epoch(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
        print('Num of training examples:',len(self.train_data))
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        self.dic_video_level_preds={}
        apm = APMeter()
        #switch to train mode
        self.model.train()    
        end = time.time()          
        #if self.epoch ==0:
            #self.model.init_weight()#initialize enhance weights to zero

        #random shuffle the dataset
        print("Shuffling dataset")
        keys= list(self.train_data.keys())
        random.shuffle(keys)
        for name in keys:#self.train_data.keys(): 
        # clear out the hidden state of the LSTM

            data_time.update(time.time() - end)
            self.model.zero_grad()
            self.model.hidden = self.model.init_hidden()
            #if self.enhance:
            #    data = torch.cat((self.train_data[name],self.enhance_array[name]),2)  
            #else:
            data = self.train_data[name]

        # Forward propagation for all frames in the video
            cnn_out = Variable(data).cuda()
            targets = Variable(self.train_labels[name]).cuda()
           # Step 3. Run our forward pass.
            tag_scores = self.model(cnn_out)
            
            # Step 4. Compute the loss
            # tag_scores=(batch_size,num_classes,seq_length), target=(batch_size,seq_length)
            loss = self.criterion(tag_scores, targets)
                
            # measure accuracy and record loss
            losses.update(loss.data, self.train_data[name].size(0))
            # compute gradient and do SGD step
            #Backpropagation through time after all frames are forward propagated
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #Calculate video level prediction
            preds = tag_scores.data.cpu().numpy()
            preds = preds.reshape((self.num_classes,-1))
            preds = np.array(np.transpose(preds))
            output=torch.from_numpy(preds).float()#sequence_length x num_classes
            if self.dataset=='multithumos':
                output = torch.sigmoid(output)
                #Average precision meter
                labels = self.train_labels[name].view(self.num_classes,-1)
                labels = labels.t()#shape=(seq_length,num_classes)
                apm.add(output, labels)
                topn=10
            else:
                output = F.softmax(output, dim=1)#F.log_softmax(output, dim=1)
                preds = output.numpy()
                prec1, prec5 = accuracy(output, targets.cpu(), topk=(1, 5))
                top1.update(prec1, self.train_data[name].size(0))
                top5.update(prec5, self.train_data[name].size(0))
                topn=5

    
            self.dic_video_level_preds[name] = np.average(preds,axis=0)
        #video_top1, video_top5, video_loss = self.frame2_video_level_accuracy(preds.shape[0],loss)
        if self.dataset=='ucf':
            info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Data Time':[round(data_time.avg,3)],
                'Loss':[round(losses.avg,5)],
                'Prec@1':[round(top1.avg,4)],
                'Prec@5':[round(top5.avg,4)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
            record_info(info, 'record/lstm/'+self.dataset+'opf_train.csv','train')
        else:
            #accuracy
            print 'MAP:', apm.value().mean()
            print 'loss:',round(losses.avg,5)# loss.data.cpu()
            print 'Learning Rate:',self.optimizer.param_groups[0]['lr']
    
    # Evaluate
    def validate_1epoch(self):

        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        print('Num of validation examples:',len(self.validation_data))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds={}
        apm = APMeter()
        end = time.time()
        correct = {}

        with torch.no_grad():

          for name in self.validation_data.keys(): 
            self.model.hidden = self.model.init_hidden()
            
            #data = data.sub_(127.353346189).div_(14.971742063)
            #label = self.labels[name].cuda(async=True)
            #if self.enhance:
            #    data = torch.cat((self.validation_data[name],self.enhance_array[name]),2)  
            #else:
            data = self.validation_data[name]
            data_var = Variable(data).cuda(async=True)
            label_var = Variable(self.validation_labels[name]).cuda(async=True)

            # compute output (batch_size,num_classes,seq_length)
            output = self.model(data_var)

            #Calculate video level prediction
            preds = output.data.cpu().numpy()
            preds = preds.reshape((self.num_classes,-1))
            preds = np.transpose(preds)
            results=torch.from_numpy(preds).float() #[seq_length,num_class]
            if self.dataset=='multithumos':
                results = torch.sigmoid(results)
                #Average precision meter
                labels = self.validation_labels[name].view(self.num_classes,-1)
                labels = labels.t()
                apm.add(results, labels)
                topn=10
            else:
                results = F.softmax(results, dim=1)#F.log_softmax(results, dim=1)
                preds = results.numpy()
                prec1, prec5 = accuracy(results, label_var.cpu(), topk=(1, 5))
                top1.update(prec1, self.validation_data[name].size(0))
                top5.update(prec5, self.validation_data[name].size(0))
                topn=5
            #print 'prediction shape:',preds.shape#= sequence_length * num_classes
            '''
            correct[name]=0
            if self.dataset=='multithumos':
              for i in range(preds.shape[0]):
                if  self.validation_labels[name][0,np.argmax(results[i,:].numpy()),i]==1:
                    #print name,np.argmax(results[i,:].numpy())
                    correct[name]+=1
            '''
            batch_time.update(time.time() - end)
            # measure accuracy and record loss
            # tag_scores=(batch_size,num_classes,seq_length), target=(batch_size,seq_length)
            loss = self.criterion(output, label_var)
            losses.update(loss.data, self.validation_data[name].size(0))

            #video level predicts
            self.dic_video_level_preds[name] = np.average(preds,axis=0)#np.sum(preds,axis=1)

        #print correct
        '''
        with open('record/lstm/lstm_video_preds.pickle','wb') as f:
            pickle.dump(self.dic_video_level_preds,f)
        f.close() 
        '''

        #Frame to video level accuracy
        if self.dataset =='ucf':
            video_top1, video_top5, video_loss = self.frame2_video_level_accuracy(preds.shape[0])
            info = {'Epoch':[self.epoch],
                'Batch Time':[round(batch_time.avg,3)],
                'Loss':[round(losses.avg,5)],
                'Prec@1':[round(video_top1,3)],
                'Prec@5':[round(video_top5,3)]
                }
            record_info(info, 'record/lstm/'+self.dataset+'opf_test.csv','test')
            return video_top1, video_loss
        else:
            #print average precision
            precision=np.zeros(6)
            precision[0]= apm.value()[1]
            precision[1:6]=apm.value()[44:49]
            #print 'AP of class basketball', ':', apm.value()[1],';', apm.value()[44],';', apm.value()[45],';', apm.value()[46],';', apm.value()[47],';', apm.value()[48],';'
            #print 'AP of class run/walk', ':', apm.value()[20],';', apm.value()[24]
            print 'average=',np.average(precision)
            print 'mAP=',apm.value().mean()
            print 'AP=',apm.value()
            print 'loss=',round(losses.avg,5)#loss.data.cpu()
            return apm.value().mean(), losses.avg#loss

    def frame2_video_level_accuracy(self,clip_num,loss=0):
     
        correct = 0
        video_level_preds = np.zeros((len(self.dic_video_level_preds),101))
        video_level_labels = np.zeros(len(self.dic_video_level_preds))
        ii=0
        for key in sorted(self.dic_video_level_preds.keys()):
            name = key.split('-',1)[0]

            preds = self.dic_video_level_preds[name]
            label = self.validation_labels[name]#int(self.test_video[name])-1
                
            video_level_preds[ii,:] = preds
            video_level_labels[ii] = label[0,0]#batch,seq_length
            ii+=1         
            #print name,', ',np.argmax(preds),' ',np.amax(preds),' ?=',label[0]
            if np.argmax(preds) == (label[0,0]):
                correct+=1
            #else:
                #print name,', ',np.amax(preds),' ',np.argmax(preds),'!=',label[0,0]
        #print 'correct num=',correct
        #top1 top5
        video_level_labels = torch.from_numpy(video_level_labels).long()
        video_level_preds = torch.from_numpy(video_level_preds).float()
        if loss==0: #validation
            loss = self.criterion(Variable(video_level_preds).cuda(), Variable(video_level_labels).cuda())   
            loss = loss.data.cpu().numpy() 
        top1,top5 = accuracy(video_level_preds, video_level_labels, topk=(1,5))     
                            
        top1 = float(top1.numpy())
        top5 = float(top5.numpy())
            
        return top1,top5,loss

#May use a confusion matrix to illustrate the results

#Load output of 2-stream CNN, size = 101
if __name__ == '__main__':
    global arg
    arg = parser.parse_args()
    print arg

    #training data - load features
    if arg.evaluate == False:
        rgb_preds='spatial/spatial_video_train.pickle'
        opf_preds = 'motion/motion_train.pickle'
        if arg.split!='01':
            rgb_preds='spatial/spatial_video_train'+arg.split+'.pickle'
            opf_preds = 'motion/motion_train'+arg.split+'.pickle'
        if arg.dataset!='ucf':  
            rgb_preds='spatial/multithumos_validation.pickle'
            opf_preds = 'motion/multithumos_validation.pickle'
        with open(rgb_preds,'rb') as f:
            rgb =pickle.load(f)
        f.close()
        with open(opf_preds,'rb') as f:
            opf =pickle.load(f)
        f.close()

    #validation data - load features
    rgb_valid='record/spatial/spatial_video_test.pickle'
    opf_valid = 'record/motion/motion_test.pickle'
    if arg.split!='01':
        rgb_valid='record/spatial/spatial_video_test'+arg.split+'.pickle'
        opf_valid = 'record/motion/motion_test'+arg.split+'.pickle'
    if arg.dataset!='ucf':  
        rgb_valid='spatial/multithumos_test.pickle'
        opf_valid = 'motion/multithumos_test.pickle'

    with open(rgb_valid,'rb') as f:
        rgb_v =pickle.load(f)
    f.close()
    with open(opf_valid,'rb') as f:
        opf_v =pickle.load(f)
    f.close()

    if arg.dataset=='ucf':
        data_path='/home/yinghan/Documents/two-stream/jpegs_256/'
        list_path ='/home/yinghan/Documents/two-stream/UCF_list/'
    else:
        data_path='/home/yinghan/Downloads/TH14_validation_set/rgb/'
        list_path ='multithumos.json'
      
#only need the labels
    dataloader = dataloader.spatial_dataloader(BATCH_SIZE=1, num_workers=1, 
                                    path=data_path, 
                                    ucf_list=list_path,
                                    ucf_split=arg.split,
                                    dataset=arg.dataset)
    train_loader,val_loader,test_video,train_video,frame_count = dataloader.run()


######################################################################
# Train the model:
    if arg.dataset=='ucf':
        num_classes = 101
        MAX_CLIP = 30
    else:#multithumos
        num_classes = 65
        MAX_CLIP = 300 #not used
    BATCH_SIZE =1 
#Training data frame_to_video   
    video_preds ={}
    video_labels ={}   
    countLen =[0,0,0,0,0,0,0]#0-5, 5-10, 10-15, 15-20, 20-25, 25-30,>30  #0-20,20-40,40-60,60-80,80-100,100-120,120+
    #print len(rgb),' ',len(opf)
    if arg.evaluate == False:
      for name in sorted(opf.keys()):   
        r = rgb[name]
        o = opf[name]

        videoName, clip_idx = name.split('-')
        clip_idx = int(clip_idx)
        if arg.dataset=='ucf':
            label = int(train_video[videoName]-1)
        else:
            label = torch.from_numpy(train_video[videoName][0]).float()#label[frame_idx,num_classes]  
          
        if clip_idx<MAX_CLIP or arg.dataset!='ucf':
            #initialize
            if videoName not in video_preds.keys():
                nb_frame = frame_count[videoName]-10+1
                sample_num = int(nb_frame/5)
                if sample_num> MAX_CLIP  and arg.dataset=='ucf':
                    sample_num= MAX_CLIP
                #count num of samples at a specific length
                
                if sample_num<=600:
                    countLen[sample_num/100]+=1
                else:
                    countLen[6]+=1
                
                video_preds[videoName] = torch.zeros((sample_num,BATCH_SIZE, 2*len(r))).cuda()#preds[j,:]
                if arg.dataset=='ucf':
                    video_labels[videoName] = torch.zeros((BATCH_SIZE,sample_num),dtype=torch.long)#preds[j,:]
                else:
                    video_labels[videoName] = torch.zeros((BATCH_SIZE,num_classes,sample_num),dtype=torch.float)
            #set input to lstm and labels
            video_preds[videoName][clip_idx,0,0:len(r)] = torch.squeeze(torch.from_numpy(r)).cuda()
            video_preds[videoName][clip_idx,0,len(r):2*len(r)] = torch.squeeze(torch.from_numpy(o)).cuda() #[clip_idx,batch,input_dim] 
            if arg.dataset=='ucf':  
                video_labels[videoName][:,clip_idx] = label
            else:
                frame_idx=clip_idx*5
                video_labels[videoName][0,:,clip_idx] = label[frame_idx,:]
                #print('check label length:', frame_idx, clip_idx)

    print("Training length count:", countLen)
    #multi input
    if arg.multi!=1:
        for videoName in video_preds.keys():
            for i in range(video_preds[videoName].shape[0]-arg.multi):
                video_preds[videoName][i,0,:] = torch.mean(video_preds[videoName][i:i+arg.multi,0,:],0,True)

#validation
    validation_preds ={}
    validation_labels ={}   
    countLen =[0,0,0,0,0,0,0]#0-5, 5-10, 10-15, 15-20, 20-25, 25-30,>30
    print len(rgb_v),' ',len(opf_v)
    for name in sorted(opf_v.keys()):   
        r = rgb_v[name]
        o = opf_v[name]

        videoName, clip_idx = name.split('-')
        clip_idx = int(clip_idx)
        
        if arg.dataset=='ucf':
            label = int(test_video[videoName]-1)
        else:
            label = torch.from_numpy(test_video[videoName][0]).float()#label[frame_idx,num_classes] 
                    

        if videoName not in validation_preds.keys():
            nb_frame = frame_count[videoName]-10+1
            sample_num = int(nb_frame/5)
            
            validation_preds[videoName] = torch.zeros((sample_num,BATCH_SIZE, 2*len(r))).cuda()#preds[j,:]
            if arg.dataset=='ucf':
                validation_labels[videoName] = torch.zeros((BATCH_SIZE,sample_num),dtype=torch.long)
            else:
                validation_labels[videoName] = torch.zeros((BATCH_SIZE,num_classes,sample_num),dtype=torch.float)
        #set validation set inputs and labels
        validation_preds[videoName][clip_idx,0,0:len(r)] = torch.squeeze(torch.from_numpy(r)).cuda()
        validation_preds[videoName][clip_idx,0,len(r):2*len(r)] = torch.squeeze(torch.from_numpy(o)).cuda()
        if arg.dataset=='ucf':
            validation_labels[videoName][:,clip_idx] = label
        else:
            frame_idx=clip_idx*5
            validation_labels[videoName][0,:,clip_idx] = label[frame_idx,:]

    #print("Testing length count:", countLen)
    #multi input
    if arg.multi!=1:
        for videoName in validation_preds.keys():
            for i in range(validation_preds[videoName].shape[0]-arg.multi):
                validation_preds[videoName][i,0,:] = torch.mean(validation_preds[videoName][i:i+arg.multi,0,:],0,True)

    

    HIDDEN_DIM = 512 # num of memory units
    NUM_LAYER = 1 # stacked layers: didn't improve performance if set to 5

    if arg.top5enhance:
        INPUT_DIM = 2*len(r) # dimension of input from CNN
    else:
        INPUT_DIM = 2*len(r)
    print 'INPUT DIMEMSION=', INPUT_DIM

    model = LSTM(input_dim=INPUT_DIM, 
                        hidden_dim=HIDDEN_DIM, 
                        num_layer=NUM_LAYER, 
                        num_classes=num_classes,
                        resume=arg.resume,
                        # Data 
                        train_data=video_preds,
                        train_labels=video_labels,
                        validation_data=validation_preds,
                        validation_labels=validation_labels,
                        multi = arg.multi,
                        evaluate=arg.evaluate,
                        enhance = arg.top5enhance,
                        # Hyper-parameter
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch = BATCH_SIZE,
                        dataset = arg.dataset
                        )

    model.run()
