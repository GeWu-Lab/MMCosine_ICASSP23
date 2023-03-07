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

        
        if args.dataset=='VGGSound':
            self.visual_feature_path = '/data/users/public/vggall/train/audio_np/'
            self.audio_feature_path = '/data/users/public/vggall/train/visual_all/visual/'
            self.stat_path = '/home/ruize_xu/vgg/vgg_noise/data/stat.csv'
            self.train_txt = '/home/ruize_xu/vgg/vgg_noise/data/train_all.csv'
            self.test_txt = '/home/ruize_xu/vgg/vgg_noise/data/test_all.csv'
        elif args.dataset=='AVE':
            self.visual_feature_path = '/home/ruize_xu/data/AVE_av/visual/'
            self.audio_feature_path = '/home/ruize_xu/data/AVE_av/audio/'
            self.stat_path = '/home/ruize_xu/ave_av/data/stat.csv'
            self.train_txt = '/home/ruize_xu/ave_av/data/train.csv'
            self.test_txt = '/home/ruize_xu/ave_av/data/test.csv'
        elif args.dataset=='KineticSound':
            if mode=='train':
                self.visual_feature_path = '/data/users/public/Kinetics-Sounds/ks-train-frame-1fps/ks-train-set/'
                self.audio_feature_path = '/data/users/public/Kinetics-Sounds/ks-train-frame-1fps/ks-train-set/'
            #self.stat_path = '/home/ruize_xu/ave_av/data/stat.csv'
            else:
                self.visual_feature_path = '/data/users/public/Kinetics-Sounds/ks-test-frame-1fps/test-set'
                self.audio_feature_path = '/data/users/public/Kinetics-Sounds/ks-test-frame-1fps/test-set'
    
            self.train_txt = '/home/ruize_xu/ks/data/train_1fps_path.txt'
            self.test_txt = '/home/ruize_xu/ave_av/data/test_1fps_path.txt'
        elif args.dataset=='CREMAD':
            self.visual_feature_path = '/home/xiaokang_peng/data/CREMA-D/AudioWAV/'
            self.audio_feature_path = '/home/xiaokang_peng/data/CREMA-D/image/'
            self.stat_path = '/home/xiaokang_peng/emotion/data/stat.csv'
            self.train_txt = '/home/xiaokang_peng/emotion/data/train.csv'
            self.test_txt = '/home/xiaokang_peng/emotion/data/test.csv'

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
                    if args.dataset == 'AVE':
                        # AVE, delete repeated labels
                        a = set(data)
                        if item[1] in a:
                            del data2class[item[1]]
                            data.remove(item[1])
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

class AVDataset_AVE(Dataset):

    def __init__(self, args, mode='train', transforms=None):
        classes = []
        data = []
        data2class = {}
        self.mode = mode
        self.visual_path = '/home/ruize_xu/data/AVE_av/visual'
        self.audio_path = '/home/ruize_xu/data/AVE_av/audio'
        self.stat_path = '/home/ruize_xu/ave_av/data/stat.csv'
        self.train_txt = '/home/ruize_xu/ave_av/data/train.csv'
        self.test_txt = '/home/ruize_xu/ave_av/data/test.csv'
        with open(self.stat_path, encoding='UTF-8-sig') as f1:
            csv_reader = csv.reader(f1)
            for row in csv_reader:
                classes.append(row[0])
        if mode == 'train':
            csv_file = self.train_txt
        else:
            csv_file = self.test_txt
        
        with open(csv_file) as f:

            csv_reader = csv.reader(f)
            for item in csv_reader:

                if item[0] in classes and os.path.exists(
                        self.audio_path + '/' + item[1] + '.wav') and os.path.exists(
                        self.visual_path + '/' + item[1]):

                    data.append(item[1])
                    data2class[item[1]] = item[0]

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
        resamples = np.tile(samples, 10)[:160000]
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
        file_num = len([lists for lists in os.listdir(path)])

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

        pick_num = 3
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
            path1.append('frame_0000'+ str(t[i]) + '.jpg')
            image.append(Image.open(path + "/" + path1[i]).convert('RGB'))
            image_arr.append(transf(image[i]))
            image_arr[i] = image_arr[i].unsqueeze(1).float()
            if i==0:
                image_n = copy.copy(image_arr[i])
            else:
                image_n = torch.cat((image_n, image_arr[i]), 1)

        return spectrogram, image_n, self.classes.index(self.data2class[av_file]), av_file

class AVDataset_KS(Dataset):

    def __init__(self, args, mode, transforms=None):
        classes = []
        data = []
        data2class = {}

        for i in range(31):
            classes.append(str(i))

        self.mode = mode
        if mode=='train':
            self.visual_path = '/data/users/public/Kinetics-Sounds/ks-train-frame-1fps/ks-train-set'
            self.audio_path = '/data/users/public/Kinetics-Sounds/ks-train-frame-1fps/ks-train-set'
            #self.stat_path = '/home/ruize_xu/ave_av/data/stat.csv'
        else:
            self.visual_path = '/data/users/public/Kinetics-Sounds/ks-test-frame-1fps/test-set'
            self.audio_path = '/data/users/public/Kinetics-Sounds/ks-test-frame-1fps/test-set'
    
        self.train_txt = '/home/ruize_xu/ks/data/train_1fps_path.txt'
        self.test_txt = '/home/ruize_xu/ks/data/test_1fps_path.txt'

        if mode == 'train':
            csv_file = self.train_txt
        else:
            csv_file = self.test_txt
                    
        with open(csv_file) as f:
            for line in f:
                item = line.split("\n")[0].split(" ")
                name = item[0].split("/")[-1]

                if item[-1] in classes :
                    if os.path.exists(self.visual_path + '/' + name):

                        data.append(name)
                        data2class[name] = item[-1]



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

        samples, samplerate = librosa.load(self.audio_path + '/'+ av_file + '/'+ av_file + '.wav')
        resamples = np.tile(samples, 10)[:160000]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
        spectrogram = np.log(spectrogram + 1e-7)

        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram - mean, std + 1e-9)
        
        if spectrogram.shape != (257,1004):
            spectrogram = np.random.rand(257,1004)
        #np.save('/home/xiaokang_peng/avetry/ave_av/' + self.mode + '/audio_np/' + av_file + '.npy', spectrogram)
        #spectrogram = np.load('/home/xiaokang_peng/vggtry/vggall/' + self.mode + '/audio_np/' + av_file + '.npy')



        #Visual
        path = self.visual_path + '/' + av_file
        file_num = len([lists for lists in os.listdir(path)])
        #file_num = 3

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

        pick_num = 3
        seg = int(file_num/pick_num)
        path1 = []
        image = []
        image_arr = []
        t = [0]*pick_num

        for i in range(pick_num):
            t[i] = random.randint(i*seg+1,i*seg+seg) if file_num > 6 else 1
            if t[i]>=10:
                t[i] = 9
            #if self.mode == 'test':
            #    t[i] = seg * i + 1
            path1.append('frame_0000'+ str(t[i]) + '.jpg')
            image.append(Image.open(path + "/" + path1[i]).convert('RGB'))
            #path1.append(path + '.jpg')
            #image.append(Image.open(path1[i]).convert('RGB'))
            
            image_arr.append(transf(image[i]))
            image_arr[i] = image_arr[i].unsqueeze(1).float()
            if i==0:
                image_n = copy.copy(image_arr[i])
            else:
                image_n = torch.cat((image_n, image_arr[i]), 1)
        #print(spectrogram.shape, image_n.size())
        #spectrogram = np.random.rand(257,1004)

        return spectrogram, image_n, self.classes.index(self.data2class[av_file]), av_file

class AVDataset_VGG(Dataset):

    def __init__(self, args, mode, transforms=None):
        classes = []
        data = []
        data2class = {}

        self.mode = mode
        self.visual_path = '/data/users/public/vggall/train/visual_all/visual'
        self.audio_path = '/data/users/public/vggall/train/audio_np'
        self.stat_path = '/home/ruize_xu/vgg/vgg_noise/data/stat.csv'
        self.train_txt = '/home/ruize_xu/vgg/vgg_noise/data/train_all.csv'
        self.test_txt = '/home/ruize_xu/vgg/vgg_noise/data/test_all.csv'

        if mode == 'train':
            csv_file = self.train_txt
        else:
            csv_file = self.test_txt

        with open(self.stat_path) as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                classes.append(row[0])

        with open(csv_file) as f:

            csv_reader = csv.reader(f)
            valid_a=0
            valid_v=0
            for item in csv_reader:
                

                if item[2] in classes and os.path.exists(
                        '/data/users/public/vggall/' + mode + '/audio_np/' + item[0] + '_' + item[1] + '.npy') and os.path.exists(
                        '/data/users/public/vggall/' + mode + '/visual_all/visual/' + item[0] + '_' + item[1]):

                    data.append(item[0] + '_' + item[1])
                    data2class[item[0] + '_' + item[1]] = item[2]
        print("valid_a",valid_a)
        print("valid_v",valid_v)

        # print(args.csv_path + args.test)
        print('data load over')
       
        self.transforms = transforms
        self.classes = sorted(classes)
        print(self.classes)

        print("now in",self.mode)
        self.data2class = data2class


        self._init_atransform()

        self.av_files = []
        for item in data:
            self.av_files.append(item)
        
        '''
        if self.mode == 'train':
            self.av_files = self.av_files[0:2000]
        else:
            self.av_files = self.av_files[0:500]
        '''
        
        
        
        

        print('# of files = %d ' % len(self.av_files))
        print('# of classes = %d' % len(self.classes))


    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.av_files)

    def __getitem__(self, idx):
        av_file = self.av_files[idx]

        # Audio
        '''
        samples, samplerate = sf.read(self.audio_path + '/'+ av_file + '.wav')
        resamples = np.tile(samples, 10)[:160000]
        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
        spectrogram = np.log(spectrogram + 1e-7)
        #spectrogram = time_shift_spectrogram(spectrogram)
        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram - mean, std + 1e-9)
        np.save('/home/xiaokang_peng/vggtry/vggall/' + self.mode + '/audio_np/' + av_file + '.npy', spectrogram)
        '''

        spectrogram = np.load('/data/users/public/vggall/' + self.mode + '/audio_np/' + av_file + '.npy')




        #Visual
        path = '/data/users/public/vggall/' + self.mode + '/visual_all/visual/' + av_file
        file_num = len([lists for lists in os.listdir(path)])

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

        pick_num = 3
        seg = int(file_num/pick_num)
        path1 = []
        image = []
        image_arr = []
        t = [0]*pick_num

        for i in range(pick_num):
            t[i] = random.randint(i*seg+1,i*seg+seg) if file_num > 6 else 1
            path1.append('frame_0000'+ str(t[i]) + '.jpg')
            image.append(Image.open(path + "/" + path1[i]).convert('RGB'))
            image_arr.append(transf(image[i]))
            image_arr[i] = image_arr[i].unsqueeze(1).float()
            if i==0:
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

class RDDataset_F1(Dataset):

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
        u = flow_transf(Image.open(flow + 'u/' + self.ro_files[idx] + '/frame000003.jpg').convert('L'))
        v = flow_transf(Image.open(flow + 'v/' + self.ro_files[idx] + '/frame000003.jpg').convert('L'))
        flow_n = torch.cat((u,v),0)

        # pick_num_F= 16
        # file_num_F= min(len([lists for lists in os.listdir(flow + 'u/' + self.ro_files[idx])]),1000)
        # seg_F = int(file_num_F/pick_num_F)
        # path2 = []
        
        # t_F = [0]*pick_num_F

        # for i in range(pick_num_F):
        #     t_F[i] = random.randint(i*seg_F+1,i*seg_F+seg_F) if file_num_F >= 16 else 1
            
        #     if self.mode == 'test':
        #         t_F[i] = seg_F * i + 1
        #     if t_F[i]==0:
        #         path2.append('frame000001.jpg')
        #     if t_F[i]<10:
        #         path2.append('frame00000'+ str(t_F[i]) + '.jpg')
        #     elif t_F[i]>=10 and t_F[i]<100:
        #         path2.append('frame0000'+ str(t_F[i]) + '.jpg')
        #     elif t_F[i]>=100 and t_F[i]<1000:
        #         path2.append('frame000'+ str(t_F[i]) + '.jpg')
        #     elif t_F[i]>=1000:
        #         path2.append('frame00'+ str(t_F[i]) + '.jpg')

            
            
        #     # print(self.ro_files[idx])
        #     # print(type(self.ro_files[idx]))
        #     # print(path2[i])
        #     # print(type(path2[i]))

        #     #print(flow + 'u/' + self.ro_files[idx] + '/'+str(path2[i]))

            


        #     # if isinstance(flow,str) and isinstance(self.ro_files[idx],str) and isinstance(path2[i],str):
        #     #     print(idx)
        #     #     print("normal")
                
        #     # else:
        #     #     print(idx)
        #     #     print(flow)
        #     #     print(self.ro_files[idx])  
        #     #     print(path2[i])
            
                    
        #     vs=flow+'v/'+self.ro_files[idx]+"/" +str(path2[i])
            
        #     us=flow+'u/' +self.ro_files[idx] + "/" + str(path2[i])
             
            
        #     #OF_u.append(Image.open(flow + 'u/' + self.ro_files[idx] + '/frame000003.jpg').convert('L'))
           
        #                 #Image.open(flow + 'u/' + self.ro_files[idx] + '/frame000003.jpg').convert('L')
        #     u=flow_transf(Image.open(us).convert('L')).unsqueeze(1).float()
        #     v=flow_transf(Image.open(vs).convert('L')).unsqueeze(1).float()      
            
            # flow_= torch.cat((u,v),0)

            
            
            # if i==0:
            #     flow_n = copy.copy(flow_)
            # else:
            #     flow_n = torch.cat((flow_n, flow_), 1)




        labels = self.classes.index(self.label[idx])

        return  image_n, flow_n, labels, self.ro_files[idx]


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