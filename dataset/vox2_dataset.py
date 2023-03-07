#! /usr/bin/python
# -*- encoding: utf-8 -*-

from tkinter.tix import IMAGE
import torch
import csv
import numpy
import random
import pdb
import os
import threading
import time
import math
import glob
import soundfile
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.distributed as dist
from PIL import Image
import copy

def round_down(num, divisor):
    return num - (num%divisor)

def worker_init_fn(worker_id):
    numpy.random.seed(numpy.random.get_state()[1][0] + worker_id)


def loadWAV(filename, max_frames, evalmode=True, num_eval=1):

    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage    = max_audio - audiosize + 1 
        audio       = numpy.pad(audio, (0, shortage), 'wrap')
        audiosize   = audio.shape[0]

    if evalmode:
        startframe = numpy.linspace(0,audiosize-max_audio,num=num_eval)
    else:
        startframe = numpy.array([numpy.int64(random.random()*(audiosize-max_audio))])
    
    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])

    feat = numpy.stack(feats,axis=0).astype(numpy.float)

    return feat
    

class train_dataset_loader(Dataset):
    def __init__(self, max_frames, **kwargs):

        # self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames = max_frames)
        self.audio_path = '/home/ruize_xu/data/vox2-audio/train'
        self.video_path = '/home/ruize_xu/data/vox2-png-2fps/train'
        self.train_list = '/home/ruize_xu/ruoxuan/CD/train_vox2.csv'
        self.max_frames = max_frames
        # self.musan_path = musan_path
        # self.rir_path   = rir_path
        # self.augment    = augment

        id_set = set()
        id_list = []
        self.png_num = []
        self.audio_list = []
        self.video_list = []
        
        # Read training files
        with open(self.train_list) as dataset_file:
            csv_reader = csv.reader(dataset_file)
            for item in csv_reader:
                audio_name = self.audio_path+item[0]
                video_name = self.video_path+item[1]
                self.audio_list.append(audio_name)
                self.video_list.append(video_name)
                self.png_num.append(int(item[3]))
                id_list.append(item[2])
                id_set.add(item[2])


        # Make a dictionary of ID names and ID indices
        dictkeys = list(id_set)
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }

        # Parse the training list into file names and ID indices
        self.data_label = []
        for id in id_list:
            self.data_label.append(dictkeys[id])


    def __getitem__(self, index):

        # feat = []
        # print(self.audio_list[index])
        audio = loadWAV(self.audio_list[index], self.max_frames, evalmode=False)

        pick_name = []
        if self.png_num[index]>=8:
            pick_name = ['/0000002.png','/0000006.png']
        elif self.png_num[index]==7:
            pick_name = ['/0000002.png','/0000005.png']
        elif self.png_num[index]==6:
            pick_name = ['/0000002.png','/0000004.png']
        elif self.png_num[index]==5:
            pick_name = ['/0000001.png','/0000003.png']
        else:
            pick_name = ['/0000000.png','/0000002.png']


        transf = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        image = []
        image_arr = []

        for i,name in enumerate(pick_name):
            
            # path1.append('frame_0000'+ str(t[i]) + '.jpg')
            # image.append(Image.open(path + "/" + path1[i]).convert('RGB'))
            # path1.append(path + '.jpg')
            image.append(Image.open(self.video_list[index] + name).convert('RGB'))
            
            image_arr.append(transf(image[i]))
            image_arr[i] = image_arr[i].unsqueeze(1).float()
            if i==0:
                image_n = copy.copy(image_arr[i])
            else:
                image_n = torch.cat((image_n, image_arr[i]), 1)


                    
        return torch.FloatTensor(audio),image_n, self.data_label[index]

    def __len__(self):
        return len(self.audio_list)



class test_dataset_loader(Dataset):
    def __init__(self, eval_frames, num_eval, **kwargs):
        self.max_frames = eval_frames
        self.num_eval   = num_eval
        self.audio_path = '/home/ruize_xu/data/vox2-audio/test'
        self.video_path = '/home/ruize_xu/data/vox2-png-2fps/test'
        self.test_list = '/home/ruize_xu/ruoxuan/CD/test_pairs.csv'

        self.audio_list1 = []
        self.video_list1 = []

        self.audio_list2 = []
        self.video_list2 = []

        self.labels = []

        with open(self.test_list) as f:
            csv_reader = csv.reader(f)
            for item in csv_reader:

                self.audio_list1.append(self.audio_path+item[1])
                self.audio_list2.append(self.audio_path+item[2])
                
                id_1 = item[1].split('/')[1]
                vedio_1 = item[1].split('/')[2]
                utter_1 = item[1].split('/')[3].split('.')[0]+'.txt#000.mp4'
                v1 = id_1 + '#' + vedio_1 +'#'+utter_1 + '/0000002.png'
                
                id_2 = item[2].split('/')[1]
                vedio_2 = item[2].split('/')[2]
                utter_2 = item[2].split('/')[3].split('.')[0]+'.txt#000.mp4'
                v2 = id_2 + '#' + vedio_2 +'#'+utter_2 + '/0000002.png'

                self.video_list1.append(self.video_path + '/' + v1)
                self.video_list2.append(self.video_path + '/' + v2)

                self.labels.append(item[0])


    def __getitem__(self, index):
        transf = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        audio1 = loadWAV(self.audio_list1[index], self.max_frames, evalmode=True, num_eval=self.num_eval)
        audio2 = loadWAV(self.audio_list2[index], self.max_frames, evalmode=True, num_eval=self.num_eval)

        image1 = transf(Image.open(self.video_list1[index]).convert('RGB')).unsqueeze(1).float()
        image2 = transf(Image.open(self.video_list2[index]).convert('RGB')).unsqueeze(1).float()
        

        return (torch.FloatTensor(audio1),image1), (torch.FloatTensor(audio2),image2), int(self.labels[index])

    def __len__(self):
        return len(self.audio_list1)


class train_dataset_sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size, distributed, seed, **kwargs):

        self.data_label         = data_source.data_label
        self.nPerSpeaker        = nPerSpeaker
        self.max_seg_per_spk    = max_seg_per_spk
        self.batch_size         = batch_size
        self.epoch              = 0
        self.seed               = seed
        self.distributed        = distributed
        
    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_label), generator=g).tolist()

        data_dict = {}

        # Sort into dictionary of file indices for each ID
        for index in indices:
            speaker_label = self.data_label[index]
            if not (speaker_label in data_dict):
                data_dict[speaker_label] = []
            data_dict[speaker_label].append(index)


        ## Group file indices for each class
        dictkeys = list(data_dict.keys())
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []
        
        for findex, key in enumerate(dictkeys):
            data    = data_dict[key]
            numSeg  = round_down(min(len(data),self.max_seg_per_spk),self.nPerSpeaker)
            
            rp      = lol(numpy.arange(numSeg),self.nPerSpeaker)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        ## Mix data in random order
        mixid           = torch.randperm(len(flattened_label), generator=g).tolist()
        mixlabel        = []
        mixmap          = []

        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = round_down(len(mixlabel), self.batch_size)
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        mixed_list = [flattened_list[i] for i in mixmap]

        ## Divide data to each GPU
        if self.distributed:
            total_size  = round_down(len(mixed_list), self.batch_size * dist.get_world_size()) 
            start_index = int ( ( dist.get_rank()     ) / dist.get_world_size() * total_size )
            end_index   = int ( ( dist.get_rank() + 1 ) / dist.get_world_size() * total_size )
            self.num_samples = end_index - start_index
            # print(mixed_list[start_index:end_index])
            return iter(mixed_list[start_index:end_index])
        else:
            total_size = round_down(len(mixed_list), self.batch_size)
            self.num_samples = total_size
            return iter(mixed_list[:total_size])

    
    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class train_dataset_loader_ddp(Dataset):
    def __init__(self, max_frames, **kwargs):

        # self.augment_wav = AugmentWAV(musan_path=musan_path, rir_path=rir_path, max_frames = max_frames)
        self.audio_path = '/data/users/public/vox_audio/vox/voxceleb2'
        self.video_path = '/data/users/public/vox2/vox2-png-2fps/train'
        self.train_list = '/home/ruize_xu/ruoxuan/CD/train_vox2.csv'
        self.max_frames = max_frames
        # self.musan_path = musan_path
        # self.rir_path   = rir_path
        # self.augment    = augment

        id_set = set()
        id_list = []
        self.png_num = []
        self.audio_list = []
        self.video_list = []
        
        # Read training files
        with open(self.train_list) as dataset_file:
            csv_reader = csv.reader(dataset_file)
            for item in csv_reader:
                audio_name = self.audio_path+item[0]
                video_name = self.video_path+item[1]
                self.audio_list.append(audio_name)
                self.video_list.append(video_name)
                self.png_num.append(int(item[3]))
                id_list.append(item[2])
                id_set.add(item[2])


        # Make a dictionary of ID names and ID indices
        dictkeys = list(id_set)
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }

        # Parse the training list into file names and ID indices
        self.data_label = []
        for id in id_list:
            self.data_label.append(dictkeys[id])


    def __getitem__(self, indices):

        # feat = []
        # print(self.audio_list[index])
        transf = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        feat = []
        all_image = []

        for index in indices:
            audio = loadWAV(self.audio_list[index], self.max_frames, evalmode=False)
            feat.append(audio)

            pick_name = []
            if self.png_num[index]>=8:
                pick_name = ['/0000002.png','/0000006.png']
            elif self.png_num[index]==7:
                pick_name = ['/0000002.png','/0000005.png']
            elif self.png_num[index]==6:
                pick_name = ['/0000002.png','/0000004.png']
            elif self.png_num[index]==5:
                pick_name = ['/0000001.png','/0000003.png']
            else:
                pick_name = ['/0000000.png','/0000002.png']


            for i,name in enumerate(pick_name):
                
                # path1.append('frame_0000'+ str(t[i]) + '.jpg')
                # image.append(Image.open(path + "/" + path1[i]).convert('RGB'))
                # path1.append(path + '.jpg')
                open_image = Image.open(self.video_list[index] + name).convert('RGB')
                
                t_image = transf(open_image).unsqueeze(1).float()
                if i==0:
                    image_n = copy.copy(t_image)
                else:
                    image_n = torch.cat((image_n, t_image), 1)

            all_image.append(image_n)
            
        feat = numpy.concatenate(feat, axis=0)
        all_image = numpy.concatenate(all_image, axis=0)


        return torch.FloatTensor(feat),torch.FloatTensor(all_image), self.data_label[index]

    def __len__(self):
        return len(self.audio_list)


class vox1_test_dataset_loader(Dataset):
    def __init__(self, eval_frames, num_eval, **kwargs):
        self.max_frames = eval_frames
        self.num_eval   = num_eval
        self.audio_path = '/home/ruize_xu/data/vox1-audio/'
        self.video_path = '/home/ruize_xu/data/vox1-png/'
        self.test_list = '/home/ruize_xu/ruoxuan/CD/veri_test.csv'

        self.audio_list1 = []
        self.video_list1 = []

        self.audio_list2 = []
        self.video_list2 = []

        self.labels = []

        with open(self.test_list) as f:
            csv_reader = csv.reader(f)
            for item in csv_reader:

                self.audio_list1.append(self.audio_path+item[1])
                self.audio_list2.append(self.audio_path+item[2])
                
                id_1 = item[1].split('/')[0]
                vedio_1 = item[1].split('/')[1]
                utter_1 = item[1].split('/')[2].split('.')[0]+'.txt#000.mp4'
                v1 = id_1 + '#' + vedio_1 +'#'+utter_1 + '/0000002.png'
                
                id_2 = item[2].split('/')[0]
                vedio_2 = item[2].split('/')[1]
                utter_2 = item[2].split('/')[2].split('.')[0]+'.txt#000.mp4'
                v2 = id_2 + '#' + vedio_2 +'#'+utter_2 + '/0000002.png'

                self.video_list1.append(self.video_path + v1)
                self.video_list2.append(self.video_path + v2)

                self.labels.append(item[0])


    def __getitem__(self, index):
        transf = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        audio1 = loadWAV(self.audio_list1[index], self.max_frames, evalmode=True, num_eval=self.num_eval)
        audio2 = loadWAV(self.audio_list2[index], self.max_frames, evalmode=True, num_eval=self.num_eval)

        image1 = transf(Image.open(self.video_list1[index]).convert('RGB')).unsqueeze(1).float()
        image2 = transf(Image.open(self.video_list2[index]).convert('RGB')).unsqueeze(1).float()
        

        return (torch.FloatTensor(audio1),image1), (torch.FloatTensor(audio2),image2), int(self.labels[index])

    def __len__(self):
        return len(self.audio_list1)