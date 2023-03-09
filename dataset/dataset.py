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

def time_shift_spectrogram(spectrogram):
    nb_cols = spectrogram.shape[1]
    nb_shifts = np.random.randint(0, nb_cols)

    return np.roll(spectrogram, nb_shifts, axis=1)


class AVDataset(Dataset):

    def __init__(self, args, mode='train'):
        classes = []
        data = []
        data2class = {}
        self.mode = mode

        
        if args.dataset=='CREMAD':
            self.visual_feature_path = '/{PATH of CREMAD SAVEDIR}/AudioWAV/'
            self.audio_feature_path = '/{PATH of CREMAD SAVEDIR}/image/'
            self.stat_path = '../data/CREMAD/stat.csv'
            self.train_txt = '../data/CREMAD/train.csv'
            self.test_txt = '../data/CREMAD/test.csv'

        with open(self.stat_path) as f1:
            csv_reader = csv.reader(f1)
            for row in csv_reader:
                classes.append(row[0])
                
        if mode == 'train':
            csv_file = self.train_txt
        else:
            csv_file = self.test_txt

        with open(csv_file) as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(self.audio_feature_path, item[1] + '.pkl')
                visual_path = os.path.join(self.visual_feature_path, item[1])
                if os.path.exists(audio_path) and os.path.exists(visual_path):   
                    data.append(item[1])
                    data2class[item[1]] = item[0]
                else:
                    continue

        self.classes = sorted(classes)

        print(self.classes)
        self.data2class = data2class

        self.av_files = []
        for item in data:
            self.av_files.append(item)
        print('# of files = %d ' % len(self.av_files))
        print('# of classes = %d' % len(self.classes))

    def __len__(self):
        return len(self.av_files)

    def __getitem__(self, idx):
        av_file = self.av_files[idx]

        # Audio
        audio_path = os.path.join(self.audio_feature_path, av_file + '.pkl')
        spectrogram = pickle.load(open(audio_path, 'rb'))

        # Visual
        visual_path = os.path.join(self.visual_feature_path, av_file)
        file_num = len(os.listdir(visual_path))

        if self.mode == 'train':

            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        pick_num = 3
        seg = int(file_num / pick_num)
        path1 = []
        image = []
        image_arr = []
        t = [0] * pick_num

        for i in range(pick_num):
            t[i] = seg * i + 1
            path1.append('frame_0000' + str(t[i]) + '.jpg')
            image.append(Image.open(visual_path + "/" + path1[i]).convert('RGB'))
            image_arr.append(transform(image[i]))
            image_arr[i] = image_arr[i].unsqueeze(1).float()
            if i == 0:
                image_n = copy.copy(image_arr[i])
            else:
                image_n = torch.cat((image_n, image_arr[i]), 1)

        return spectrogram, image_n, self.classes.index(self.data2class[av_file]), av_file


class AVDataset_CD(Dataset):


    def __init__(self, args, mode='train', transforms=None):
        classes = []
        data = []
        data2class = {}

        self.mode=mode
        self.visual_path = '/home/xiaokang_peng/data/CREMA-D/image'
        self.audio_path = '/home/xiaokang_peng/data/CREMA-D/AudioWAV'
        self.stat_path = '/home/xiaokang_peng/emotion/data/stat.csv'
        self.train_txt = '/home/xiaokang_peng/emotion/data/train.csv'
        self.test_txt = '/home/xiaokang_peng/emotion/data/test.csv'
        if mode == 'train':
            csv_file = self.train_txt
        else:
            csv_file = self.test_txt

        with open(self.stat_path, encoding='UTF-8-sig') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                classes.append(row[0])

        with open(csv_file) as f:

            csv_reader = csv.reader(f)
            for item in csv_reader:
                #print(item)

                if item[1] in classes and os.path.exists(
                        self.audio_path + '/' + item[0] + '.wav') and os.path.exists(
                        self.visual_path + '/' + item[0] + '.jpg'):

                    data.append(item[0])
                    data2class[item[0]] = item[1]

                #print(args.audio_path + '/' + item[1] + '.wav')


        print('data load over')
        print(len(data))
        
        self.transforms = transforms
        self.classes = sorted(classes)

        print(self.classes)
        self.data2class = data2class


        self._init_atransform()

        self.av_files = []
        for item in data:
            self.av_files.append(item)
        print('# of files = %d ' % len(self.av_files))
        print('# of classes = %d' % len(self.classes))


    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.av_files)

    def __getitem__(self, idx):
        av_file = self.av_files[idx]

        # Audio

        samples, samplerate = sf.read(self.audio_path + '/'+ av_file + '.wav')
        resamples = np.tile(samples, 3)[:48000]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
        spectrogram = np.log(spectrogram + 1e-7)
        #spectrogram = time_shift_spectrogram(spectrogram)
        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram - mean, std + 1e-9)
        #np.save('/home/xiaokang_peng/avetry/ave_av/' + self.mode + '/audio_np/' + av_file + '.npy', spectrogram)
        #spectrogram = np.load('/home/xiaokang_peng/vggtry/vggall/' + self.mode + '/audio_np/' + av_file + '.npy')



        #Visual
        path = self.visual_path + '/' + av_file
        #file_num = len([lists for lists in os.listdir(path)])
        file_num = 1
        # print("file_num")
        # print(file_num)

        if self.mode == 'train':

            transf = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transf = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        pick_num = 1
        seg = int(file_num/pick_num)
        path1 = []
        image = []
        image_arr = []
        t = [0]*pick_num

        for i in range(pick_num):
            t[i] = random.randint(i*seg+1,i*seg+seg) if file_num > 6 else 1
            if t[i]==10:
                t[i] -= 1
            if self.mode == 'test':
                t[i] = seg * i + 1
            #path1.append('frame_0000'+ str(t[i]) + '.jpg')
            #image.append(Image.open(path + "/" + path1[i]).convert('RGB'))
            path1.append(path + '.jpg')
            image.append(Image.open(path1[i]).convert('RGB'))
            
            image_arr.append(transf(image[i]))
            image_arr[i] = image_arr[i].unsqueeze(1).float()
            if i==0:
                image_n = copy.copy(image_arr[i])
            else:
                image_n = torch.cat((image_n, image_arr[i]), 1)

        return spectrogram, image_n, self.classes.index(self.data2class[av_file]), av_file


class AVDataset_VOX1_Train(Dataset):


    def __init__(self,  transforms=None):
        classes = []
        visual_data = []
        audio_data = []
        id_labels = []
        label2class = {}

        self.visual_path = '/data/users/public/vox_video/face-video-preprocessing/vox-png/train'
        self.audio_path = '/data/users/public/vox_audio/vox/voxceleb1'
        self.stat_path = '/home/ruize_xu/vox/stat.csv'
        self.train_txt = '/home/ruize_xu/vox/train_vox1.csv'
        self.test_txt = '/home/ruize_xu/vox/train_vox1.csv'


        csv_file = self.train_txt



        with open(self.stat_path, encoding='UTF-8-sig') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                classes.append(row[0])
                label2class[row[0]] = row[1]




        with open(csv_file) as f:

            csv_reader = csv.reader(f)
            # print(csv_reader)
            for item in csv_reader:

                audio_data.append(item[0])
                visual_data.append(item[1])
                id_labels.append(item[2])

                #print(args.audio_path + '/' + item[1] + '.wav')


        print('data load over')
        
        self.transforms = transforms
        self.classes = classes
        self.visual_data = visual_data
        self.audio_data = audio_data
        self.labels = id_labels

        # print(self.classes)
        self.label2class = label2class


        self._init_atransform()

        print('# of files = %d ' % len(self.audio_data))
        print('# of classes = %d' % len(self.classes))


    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):
        a_file = self.audio_data[idx]

        # Audio

        samples, samplerate = sf.read(self.audio_path + a_file)
        resamples = np.tile(samples, 3)[:48000]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
        spectrogram = np.log(spectrogram + 1e-7)
        #spectrogram = time_shift_spectrogram(spectrogram)
        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram - mean, std + 1e-9)
        #np.save('/home/xiaokang_peng/avetry/ave_av/' + self.mode + '/audio_np/' + av_file + '.npy', spectrogram)
        #spectrogram = np.load('/home/xiaokang_peng/vggtry/vggall/' + self.mode + '/audio_np/' + av_file + '.npy')



        #Visual
        v_file = self.visual_data[idx]
        path = self.visual_path + v_file
        #file_num = len([lists for lists in os.listdir(path)])
        # file_num = 1

        transf = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


        # pick_num = 1
        # seg = int(file_num/pick_num)
        pick_name = ['/0000000.png','/0000024.png','/0000048.png']
        image = []
        image_arr = []
        # t = [0]*pick_num

        for i,name in enumerate(pick_name):
            
            # path1.append('frame_0000'+ str(t[i]) + '.jpg')
            # image.append(Image.open(path + "/" + path1[i]).convert('RGB'))
            # path1.append(path + '.jpg')
            image.append(Image.open(path + name).convert('RGB'))
            
            image_arr.append(transf(image[i]))
            image_arr[i] = image_arr[i].unsqueeze(1).float()
            if i==0:
                image_n = copy.copy(image_arr[i])
            else:
                image_n = torch.cat((image_n, image_arr[i]), 1)

        return spectrogram, image_n, int(self.label2class[self.labels[idx]])


class AVDataset_VOX1_Test(Dataset):


    def __init__(self,  transforms=None):
        visual_data_1 = []
        audio_data_1 = []
        visual_data_2 = []
        audio_data_2 = []
        labels = []

        self.visual_path = '/data/users/public/vox_video/face-video-preprocessing/vox-png/test/'
        self.audio_path = '/data/users/public/vox_audio/vox/voxceleb1/vox1_test_wav/wav/'
        self.test_txt = '/home/ruize_xu/vox/test_vox.csv'


        csv_file = self.test_txt


        with open(csv_file) as f:

            csv_reader = csv.reader(f)
            # print(csv_reader)
            for item in csv_reader:

                audio_data_1.append(item[1])
                audio_data_2.append(item[2])

                id_1 = item[1].split('/')[0]
                vedio_1 = item[1].split('/')[1]
                utter_1 = item[1].split('/')[2].split('.')[0]+'.txt#000.mp4'
                v1 = id_1 + '#' + vedio_1 +'#'+utter_1
                
                id_2 = item[2].split('/')[0]
                vedio_2 = item[2].split('/')[1]
                utter_2 = item[2].split('/')[2].split('.')[0]+'.txt#000.mp4'
                v2 = id_2 + '#' + vedio_2 +'#'+utter_2

                visual_data_1.append(v1)
                visual_data_2.append(v2)

                labels.append(item[0])

                #print(args.audio_path + '/' + item[1] + '.wav')


        print('data load over')
        
        self.transforms = transforms
        self.visual_data_1 = visual_data_1
        self.audio_data_1 = audio_data_1
        self.visual_data_2 = visual_data_2
        self.audio_data_2 = audio_data_2
        self.labels = labels



        self._init_atransform()

        print('# of files = %d ' % len(self.audio_data_1))


    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.audio_data_1)

    def __getitem__(self, idx):
        a_file_1 = self.audio_data_1[idx]
        a_file_1 = self.audio_data_2[idx]

        # Audio

        samples, samplerate = sf.read(self.audio_path + a_file_1)
        resamples = np.tile(samples, 3)[:48000]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
        spectrogram = np.log(spectrogram + 1e-7)
        #spectrogram = time_shift_spectrogram(spectrogram)
        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram1 = np.divide(spectrogram - mean, std + 1e-9)

        samples, samplerate = sf.read(self.audio_path + a_file_1)
        resamples = np.tile(samples, 3)[:48000]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
        spectrogram = np.log(spectrogram + 1e-7)
        #spectrogram = time_shift_spectrogram(spectrogram)
        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram2 = np.divide(spectrogram - mean, std + 1e-9)




        #Visual
        v_file_1 = self.visual_data_1[idx]
        v_file_2 = self.visual_data_2[idx]
        path1 = self.visual_path + v_file_1
        path2 = self.visual_path + v_file_2
        #file_num = len([lists for lists in os.listdir(path)])
        # file_num = 1


        transf = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # pick_num = 1
        # seg = int(file_num/pick_num)
        pick_name = ['/0000000.png','/0000024.png','/0000048.png']
        image_1 = []
        image_arr_1 = []
        image_2 = []
        image_arr_2 = []
        # t = [0]*pick_num

        for i,name in enumerate(pick_name):
            
            # path1.append('frame_0000'+ str(t[i]) + '.jpg')
            # image.append(Image.open(path + "/" + path1[i]).convert('RGB'))
            # path1.append(path + '.jpg')
            image_1.append(Image.open(path1 + name).convert('RGB'))
            image_2.append(Image.open(path2 + name).convert('RGB'))
            
            image_arr_1.append(transf(image_1[i]))
            image_arr_2.append(transf(image_2[i]))

            image_arr_1[i] = image_arr_1[i].unsqueeze(1).float()
            image_arr_2[i] = image_arr_2[i].unsqueeze(1).float()
            if i==0:
                image_n_1 = copy.copy(image_arr_1[i])
                image_n_2 = copy.copy(image_arr_2[i])
            else:
                image_n_1 = torch.cat((image_n_1, image_arr_1[i]), 1)
                image_n_2 = torch.cat((image_n_2, image_arr_2[i]), 1)

        return (spectrogram1, image_n_1), (spectrogram2, image_n_2), int(self.labels[idx])