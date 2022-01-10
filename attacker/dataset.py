import torch
from torch.utils import data
from torchvision.transforms import transforms
from torch.utils.data import Dataset

import os
import csv
import numpy as np
import cv2
from PIL import Image
import librosa

# data augmentation
transform_method = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                             std=[0.5, 0.5, 0.5])
                   ])

class lie_data(Dataset):
    def __init__(self, data_path, phase):
        self.names = []
        lie_annotation = {1:'truth', 0:'lie', 'truth':1, 'lie':0}

        if phase == 'train':
            txt_path = os.path.join(data_path, 'Real_life_train.txt')
        elif phase == 'test':
            txt_path = os.path.join(data_path, 'Real_life_val.txt')

        # read txt - get corresponding folders
        self.folders = []
        with open(txt_path, 'r') as csv_in:
            csv_reader = csv.reader(csv_in, delimiter=',')
            for line in csv_reader:
                self.folders.append(line[0])

        self.data_list = []
        for folder in self.folders:
            label = int(lie_annotation[folder.split('_')[0]])
            audio_path = os.path.join(data_path, 'Real_life_audio', '{}.wav'.format(folder))
            self.data_list.append([audio_path, label])
            self.names.append(audio_path.split('/')[-1])

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        audio_path, label = self.data_list[index]

        y, sr = librosa.load(audio_path)
        melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=128)
        feature = cv2.resize(melspec, (224, 224), interpolation=cv2.INTER_NEAREST)
        feature = cv2.cvtColor(feature, cv2.COLOR_GRAY2BGR)
        feature = np.transpose(feature, [2, 0, 1])
        image = torch.from_numpy((feature.copy()))
        image = image.float()/255.0
        mean = torch.as_tensor([0.5, 0.5, 0.5], dtype=image.dtype, device=image.device)
        std = torch.as_tensor([0.5, 0.5, 0.5], dtype=image.dtype, device=image.device)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        image.sub_(mean).div_(std)

        return image, label, audio_path.split('/')[-1]
    
    def __getname__(self):
        return self.names

class AdvDataset(Dataset):
    def __init__(self, data_path, phase, image_path, csv_path):
        self.images = []
        self.labels = []
        self.names = []

        if phase == 'train':
            txt_path = os.path.join(data_path, 'Real_life_train.txt')
        elif phase == 'test':
            txt_path = os.path.join(data_path, 'Real_life_val.txt')

        # read txt - get corresponding folders
        self.folders = []
        with open(txt_path, 'r') as csv_in:
            csv_reader = csv.reader(csv_in, delimiter=',')
            for line in csv_reader:
                self.folders.append(line[0])
        
        # read csv - get frames from each folder
        self.folder_images = {}
        with open(csv_path, 'r') as csv_in:
            csv_reader = csv.reader(csv_in, delimiter=',')
            for line in csv_reader:
                folder = line[0]
                images = line[1:]
                if folder not in self.folder_images:
                    self.folder_images[folder] = []
                self.folder_images[folder].append(images)
        
        # extract images for the task
        for folder in self.folders:
            for video in self.folder_images[folder]:
                image_series = []
                image_series_names = []
                for image_name in video:
                    image_series.append(os.path.join(image_path, folder, image_name))
                    image_series_names.append(image_name)
                self.images.append(image_series)
                self.names.append(image_series_names)
                if 'lie' in folder:
                    self.labels.append(0)
                elif 'truth' in folder:
                    self.labels.append(1)

        self.transform = transform_method
    def __getitem__(self, idx):
        images = []
        for image_name in self.images[idx]:
            image = self.transform(Image.open(image_name))
            images.append(image)
        label = self.labels[idx]

        return images, label, self.images[idx]
    def __getname__(self):
        return self.images
    def __len__(self):
        return len(self.images)