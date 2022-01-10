from librosa.core import audio
import librosa

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

import numpy as np
import os

from PIL import Image
import cv2

lie_annotation = {1:'truth', 0:'lie', 'truth':1, 'lie':0}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

class lie_data(Dataset):
    def __init__(self, opt, data_list):
        self.data_path = opt.data_path
        self.dataset_name = opt.dataset_name
        self.image_folder = opt.image_folder
        self.transform = transform

        self.data_list = []
        for data in data_list:
            folder_name = data[0]
            if self.dataset_name == 'Real_life':
                label = int(lie_annotation[folder_name.split('_')[0]])
            elif self.dataset_name == 'Bag_of_lies':
                label = int(folder_name.split('_')[1])

            for image_folder in self.image_folder:
                tmp_list = []
                for frame in data[1:]:
                    frame_path = os.path.join(self.data_path, image_folder, folder_name, frame)
                    tmp_list.append(frame_path)
                audio_path = os.path.join(self.data_path, self.dataset_name + '_audio', folder_name + '.wav')
                self.data_list.append([tmp_list, audio_path, label])
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        image_list, audio_path, label = self.data_list[index]
        for idx, image_path in enumerate(image_list):
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0)
            
            if idx == 0:
                output_list = image
            else:
                output_list = torch.cat((output_list, image), dim=0)

        y, sr= librosa.load(audio_path)
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

        return output_list, image, label

class lie_multi_modal_test(Dataset):
    def __init__(self, opt, data_list):
        self.data_path = opt.data_path
        self.dataset_name = opt.dataset_name
        self.image_folder = opt.image_folder
        self.transform = transform

        self.data_list = []
        for data in data_list:
            folder_name = data[0]
            if self.dataset_name == 'Real_life':
                label = int(lie_annotation[folder_name.split('_')[0]])
            elif self.dataset_name == 'Bag_of_lies':
                label = int(folder_name.split('_')[1])

            tmp_list = []
            for frame in data[1:]:
                frame_path = os.path.join(self.data_path, self.image_folder[0], folder_name, frame)
                tmp_list.append(frame_path)
            audio_path = os.path.join(self.data_path, self.image_folder[1], folder_name + '.png')
            
            self.data_list.append([tmp_list, audio_path, label])
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        image_list, audio_path, label = self.data_list[index]
        for idx, image_path in enumerate(image_list):
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0)
            
            if idx == 0:
                output_list = image
            else:
                output_list = torch.cat((output_list, image), dim=0)

        melspec = cv2.imread(audio_path)
        feature = cv2.resize(melspec, (224, 224), interpolation=cv2.INTER_NEAREST)
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

        return output_list, image, label

if __name__ == '__main__':
    pass
