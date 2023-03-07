import copy
import csv
import os
import pickle
import pdb
import soundfile as sf
import numpy as np
import random
from scipy import signal
import pickle as pkl
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import soundfile as sf
import librosa

class RDDataset(Dataset):

    def __init__(self, args, mode='train', transforms=None):

        self.rgb_path = '/home/yake_wei/data/ucf101-frames-1fps/video-set'
        self.flow_path = '/home/yake_wei/data/tvl1_flow/'
        self.stat_path = '/home/yake_wei/UCF101/data/class2id.txt'
        self.train_txt='/home/yake_wei/UCF101/data/trainlist01.txt'
        self.test_txt='/home/yake_wei/UCF101/data/testlist01.txt'


        self.mode = mode
        self.classes = []
        f = open(self.stat_path)
        line = f.readline()
        while line:
            self.classes.append(line.split(',')[-1].split('\n')[0])
            line = f.readline()
        f.close()

        if mode == 'train':
            csv_file = self.train_txt
        else:
            csv_file = self.test_txt
        
        '''
        self.label = []
        self.ro_files = []
        file_list = [lists for lists in os.listdir(self.rgb_path)]
        random.seed(0)
        random.shuffle(file_list)
        #print(file_list[100:120])
        file_num = len(file_list)
        for i in range(file_num):
            if os.path.exists(self.flow_path + '/u/' + file_list[i]):
                self.ro_files.append(file_list[i])
                self.label.append(file_list[i].split('_')[1])
        
        file_num = len(self.label)

        if mode == 'train':
        
            self.ro_files = self.ro_files[int(file_num*0.8):int(file_num*0.9)]
            self.label = self.label[int(file_num*0.8):int(file_num*0.9)]
            
        if mode == 'test':
        
            self.ro_files = self.ro_files[int(file_num*0.8):int(file_num*0.9)]
            self.label = self.label[int(file_num*0.8):int(file_num*0.9)]
        '''
        
        self.label = []
        self.ro_files = []
        label = []
        ro_files = []


        
        
        
        f = open(csv_file)
        line = f.readline()
        while line:
            ro_files.append(line.split('.')[0].split('/')[-1])
            label.append(line.split('.')[0].split('/')[0])
            line = f.readline()
        f.close()
        

        index = [i for i in range(len(label))]
        random.shuffle(index)

        for i in range(len(ro_files)):
            self.ro_files.append(ro_files[index[i]])
            self.label.append(label[index[i]])
                
            
        
        

        file_num = len(self.ro_files)
        print( file_num,'data ready')
        
        #print(self.ro_files)
        #print(self.label)

        
        self.transforms = transforms
        self._init_atransform()


    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.ro_files)

    def __getitem__(self, idx):
    
        if self.label[idx] != 'HandstandPushups':
            rgb = '/home/andong_deng/dataset/UCF101/ucf101-frames-1fps/video-set/' + self.ro_files[idx]
        else:
            rgb = '/home/andong_deng/dataset/UCF101/ucf101-frames-1fps/video-set/' + self.ro_files[idx].split('HandstandPushups')[0] + 'HandStandPushups' + self.ro_files[idx].split('HandstandPushups')[-1]
        
        flow = '/home/andong_deng/dataset/UCF101/ucf101-flow/tvl1_flow/'
        
        
        if self.mode == 'train':

            rgb_transf = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            flow_transf = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            
        else:
            rgb_transf = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            flow_transf = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
            ])

        
        # rgb
        pick_num = 3
        file_num = 6 #len([lists for lists in os.listdir(rgb)])
        seg = int(file_num/pick_num)
        path1 = []
        image = []
        image_arr = []
        t = [0]*pick_num

        for i in range(pick_num):
            t[i] = random.randint(i*seg+1,i*seg+seg) if file_num >= 6 else 1
            if t[i]==10:
                t[i] -= 1
            if self.mode == 'test':
                t[i] = seg * i + 1
            path1.append('frame_0000'+ str(t[i]) + '.jpg')
            image.append(Image.open(rgb + "/" + path1[i]).convert('RGB'))
            image_arr.append(rgb_transf(image[i]))
            image_arr[i] = image_arr[i].unsqueeze(1).float()
            if i==0:
                image_n = copy.copy(image_arr[i])
            else:
                image_n = torch.cat((image_n, image_arr[i]), 1)
                

        #flow
        # u = flow_transf(Image.open(flow + 'u/' + self.ro_files[idx] + '/frame000003.jpg').convert('L'))
        # v = flow_transf(Image.open(flow + 'v/' + self.ro_files[idx] + '/frame000003.jpg').convert('L'))
        # flow_n = torch.cat((u,v),0)

        pick_num_F= 16
        file_num_F= min(len([lists for lists in os.listdir(flow + 'u/' + self.ro_files[idx])]),1000)
        seg_F = int(file_num_F/pick_num_F)
        path2 = []
        
        t_F = [0]*pick_num_F

        for i in range(pick_num_F):
            t_F[i] = random.randint(i*seg_F+1,i*seg_F+seg_F) if file_num_F >= 16 else 1
            
            if self.mode == 'test':
                t_F[i] = seg_F * i + 1
            if t_F[i]==0:
                path2.append('frame000001.jpg')
            if t_F[i]<10:
                path2.append('frame00000'+ str(t_F[i]) + '.jpg')
            elif t_F[i]>=10 and t_F[i]<100:
                path2.append('frame0000'+ str(t_F[i]) + '.jpg')
            elif t_F[i]>=100 and t_F[i]<1000:
                path2.append('frame000'+ str(t_F[i]) + '.jpg')
            elif t_F[i]>=1000:
                path2.append('frame00'+ str(t_F[i]) + '.jpg')

            
            
            # print(self.ro_files[idx])
            # print(type(self.ro_files[idx]))
            # print(path2[i])
            # print(type(path2[i]))

            #print(flow + 'u/' + self.ro_files[idx] + '/'+str(path2[i]))

            


            # if isinstance(flow,str) and isinstance(self.ro_files[idx],str) and isinstance(path2[i],str):
            #     print(idx)
            #     print("normal")
                
            # else:
            #     print(idx)
            #     print(flow)
            #     print(self.ro_files[idx])  
            #     print(path2[i])
            
                    
            vs=flow+'v/'+self.ro_files[idx]+"/" +str(path2[i])
            
            us=flow+'u/' +self.ro_files[idx] + "/" + str(path2[i])
             
            
            #OF_u.append(Image.open(flow + 'u/' + self.ro_files[idx] + '/frame000003.jpg').convert('L'))
           
                        #Image.open(flow + 'u/' + self.ro_files[idx] + '/frame000003.jpg').convert('L')
            u=flow_transf(Image.open(us).convert('L')).unsqueeze(1).float()
            v=flow_transf(Image.open(vs).convert('L')).unsqueeze(1).float()      
            
            flow_= torch.cat((u,v),0)

            
            
            if i==0:
                flow_n = copy.copy(flow_)
            else:
                flow_n = torch.cat((flow_n, flow_), 1)




        labels = self.classes.index(self.label[idx])

        return  image_n, flow_n, labels, self.ro_files[idx]