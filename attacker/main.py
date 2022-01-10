'''
Modified Date: 2022/01/11
Author: Gi-Luen Huang
mail: come880412@gmail.com
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import argparse

from model import ensembleNet, ensembleNet_spec
from dataset import AdvDataset, lie_data
from utils import video_attack, audio_attack, create_model
from Trainer import pretrain_image, pretrain_spec

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--n_cpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.001)

parser.add_argument('--input_type', type=str, default='image', help= 'image/audio')
parser.add_argument('--action', type=str, default='generate_train_image_adv', help='pretrain_image/pretrain_spec/generate_train_image_adv/generate_test_image_adv/generate_train_spec_adv/generate_test_spec_adv')
parser.add_argument('--backbone', type=str, default='resnet18', help='alexnet/vgg16/resnet18/resnet50/inception_v3')
parser.add_argument('--attack', type=str, default='fgsm', help= 'fgsm/ifgsm/mifgsm')
parser.add_argument('--data_path', type=str, default='../../dataset')
parser.add_argument('--image_path', type=str, default='../../dataset/Real_life')
parser.add_argument('--csv_path', type=str, default='../../dataset/Real_life.csv')
parser.add_argument('--save_adv_path', type=str, default='../../dataset/Real_life_adv')
args = parser.parse_args()

# assign gpu to use
device = torch.device(0)

# the mean and std are the calculated statistics from cifar_100 dataset
data_mean = (0.5, 0.5, 0.5)    # mean for the three channels of cifar_100 images
data_std = (0.5, 0.5, 0.5)    # std for the three channels of cifar_100 images

# convert mean and std to 3-dimensional tensors for future operations
mean = torch.tensor(data_mean).to(device).view(3, 1, 1)
std = torch.tensor(data_std).to(device).view(3, 1, 1)

epsilon = 8 / 255 / std    # 8
alpha = 0.8 / 255 / std    # 0.8

if __name__ == '__main__':
    if args.action == 'pretrain_image':
        # read data
        train_set = AdvDataset(args.data_path, 'train', args.image_path, args.csv_path)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=True)

        valid_set = AdvDataset(args.data_path, 'test', args.image_path, args.csv_path)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu, drop_last=False)

        # create model
        model = create_model(backbone=args.backbone, input_type=args.input_type)
        model = model.to(device)

        # define loss function, optimizer, scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        # start training
        print('start pretraining...')
        pretrain_image(args, args.epochs, model, criterion, optimizer, scheduler, train_loader, valid_loader, device)
    elif args.action == 'pretrain_spec':
        # read data
        train_set = lie_data(args.data_path, 'train')
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu, drop_last=True)

        valid_set = lie_data(args.data_path, 'test')
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu, drop_last=False)

        # create model
        model = create_model(backbone=args.backbone, input_type=args.input_type)
        model = model.to(device)

        # define loss function, optimizer, scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        # start training
        print('start pretraining...')
        pretrain_spec(args, args.epochs, model, criterion, optimizer, scheduler, train_loader, valid_loader, device)
    elif args.action == 'generate_train_image_adv':
        train_set = AdvDataset(args.data_path, 'train', args.image_path, args.csv_path)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.n_cpu, drop_last=False)

        attack_alg = video_attack(args, epsilon, alpha, mean, std, device)
        # create model
        if args.backbone == 'ensemble':
            model = ensembleNet('image_model')
        else:
            model = create_model(backbone=args.backbone, input_type=args.input_type)
            state_dict = torch.load('./image_model/{}.pth'.format(args.backbone))
            model.load_state_dict(state_dict)
        
        model = model.to(device)

        # define loss function
        criterion = nn.CrossEntropyLoss()

        # start generating adversarial examples
        print('start generating adversarial examples...')
        adv_examples, attack_acc, attack_loss = attack_alg.gen_adv_examples(model, train_loader, args.attack, criterion)
        adv_names = train_set.__getname__()
        
        attack_alg.save_adv_images(adv_examples, adv_names)
    elif args.action == 'generate_test_image_adv':
        valid_set = AdvDataset(args.data_path, 'test', args.image_path, args.csv_path)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False)

        attack_alg = video_attack(args, epsilon, alpha, mean, std, device)
        # create model
        if args.backbone == 'ensemble':
            model = ensembleNet('image_model')
        else:
            model = create_model(backbone=args.backbone, input_type=args.input_type)
            state_dict = torch.load('./image_model/{}.pth'.format(args.backbone))
            model.load_state_dict(state_dict)
        
        model = model.to(device)

        # define loss function
        criterion = nn.CrossEntropyLoss()

        # start generating adversarial examples
        print('start generating adversarial examples...')
        adv_examples, attack_acc, attack_loss = attack_alg.gen_adv_examples(model, valid_loader, args.attack, criterion)
        adv_names = valid_set.__getname__()
        
        attack_alg.save_adv_images(adv_examples, adv_names)
    elif args.action == 'generate_train_spec_adv':
        train_set = lie_data(args.data_path, 'train')
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False)

        attack_alg = audio_attack(args, epsilon, alpha, mean, std, device)
        # create model
        if args.backbone == 'ensemble':
            model = ensembleNet_spec('spec_model')
        else:
            model = create_model(backbone=args.backbone, input_type=args.input_type)
            state_dict = torch.load('./spec_model/{}.pth'.format(args.backbone))
            model.load_state_dict(state_dict)
        
        model = model.to(device)

        # define loss function
        criterion = nn.CrossEntropyLoss()

        # start generating adversarial examples
        print('start generating adversarial examples...')
        adv_examples, attack_acc, attack_loss = attack_alg.gen_adv_examples(model, train_loader, args.attack, criterion)
        adv_names = train_set.__getname__()
        
        attack_alg.save_adv_images(adv_examples, adv_names)
    elif args.action == 'generate_test_spec_adv':
        valid_set = lie_data(args.data_path, 'test')
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False)

        attack_alg = audio_attack(args, epsilon, alpha, mean, std, device)
        # create model
        if args.backbone == 'ensemble':
            model = ensembleNet_spec('spec_model')
        else:
            model = create_model(backbone=args.backbone, input_type=args.input_type)
            state_dict = torch.load('./spec_model/{}.pth'.format(args.backbone))
            model.load_state_dict(state_dict)
        
        model = model.to(device)

        # define loss function
        criterion = nn.CrossEntropyLoss()

        # start generating adversarial examples
        print('start generating adversarial examples...')
        adv_examples, attack_acc, attack_loss = attack_alg.gen_adv_examples(model, valid_loader, args.attack, criterion)
        adv_names = valid_set.__getname__()
        
        attack_alg.save_adv_images(adv_examples, adv_names)
