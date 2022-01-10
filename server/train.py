'''
Modified Date: 2022/01/11
Author: Gi-Luen Huang
mail: come880412@gmail.com
'''

import numpy as np
import argparse
import tqdm
import os
import random

from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch
from tensorboardX import SummaryWriter

from dataset import lie_data
from utils import train_val_split, get_cosine_schedule_with_warmup
from model import CNN_GRU, CNN_Transformer, multi_modal_GRU, multi_modal_Transformer


def train(opt, model, train_loader, val_loader):
    writer = SummaryWriter('runs/%s' % opt.saved_name)
    cuda = True if torch.cuda.is_available() else False
    criterion = nn.CrossEntropyLoss()

    # Load optimizer
    if opt.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=3e-4)
    elif opt.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=3e-4)
    
    # Load scheduler
    if opt.scheduler == 'step':
        print('Use StepLR scheduler!')
        scheduler = StepLR(optimizer, gamma=opt.gamma, step_size=opt.lr_decay_epoch)
    elif opt.scheduler == 'warmup':
        total_steps = len(train_loader) * opt.n_epochs
        warmup_steps = total_steps * opt.warmup_ratio
        print('Use warmup scheduler! Total_steps: %d \t Warmup_steps: %d' % (total_steps, warmup_steps))
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if cuda:
        criterion = criterion.cuda()
        model = model.cuda()

    # Load model to continue training
    if opt.saved_model:
        print('load pretrained model!')
        model.load_state_dict(torch.load(opt.saved_model))

    train_update = 0
    max_acc = 0.

    print('Start training!')
    for epoch in range(opt.initial_epoch, opt.n_epochs):
        pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="Train[%d/%d]" % (epoch, opt.n_epochs), unit=" uttr")
        model.train()

        label_total = 0
        acc = 0.
        correct_total = 0
        loss_total = 0.

        for frame, audio, label in train_loader:
            lr = optimizer.param_groups[0]['lr']
            if cuda:
                frame = frame.cuda()
                audio = audio.cuda()
                label = label.cuda()
            
            if opt.multi_modal:
                pred = model(frame, audio)
            else:
                pred = model(frame)

            optimizer.zero_grad()
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)

            correct = np.sum(np.equal(label.cpu().numpy(), pred_label))
            label_total += len(pred_label)
            correct_total += correct
            acc = (correct_total / label_total) * 100

            loss_total += loss
            writer.add_scalar('training_loss', loss, train_update)
            writer.add_scalar('training_acc', acc, train_update)

            # Print log
            pbar.update()
            pbar.set_postfix(
            loss=f"{loss_total:.4f}",
            Accuracy=f"{acc:.2f}%",
            lr = f"{lr:.6f}"
            )
            train_update += 1

            if opt.scheduler == 'warmup':
                scheduler.step()
        pbar.close()

        # Validation & Save model
        val_acc = validation(opt, model, val_loader, writer)
        if val_acc >= max_acc:
            print('save model!')
            max_acc = val_acc
            torch.save(model.state_dict(), '%s/%s/model_epoch%d_acc%.2f.pth' % (opt.save_model_path, opt.saved_name, epoch, max_acc))

        if opt.scheduler == 'step':
            scheduler.step()
    print('best acc:', max_acc)

def validation(opt, model, val_loader, writer):
    model.eval()
    cuda = True if torch.cuda.is_available() else False
    criterion = torch.nn.CrossEntropyLoss()

    if cuda:
        criterion = criterion.cuda()

    pbar = tqdm.tqdm(total=len(val_loader), ncols=0, desc="val", unit=" step")

    label_total = 0
    correct_total = 0
    loss_total = 0.
    for frame, audio, label in val_loader:
        with torch.no_grad():
            if cuda:
                frame = frame.cuda()
                label = label.cuda()
                audio = audio.cuda()

            if opt.multi_modal:
                pred = model(frame, audio)
            else:
                pred = model(frame)

            loss = criterion(pred, label)

            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)

            correct = np.sum(np.equal(label.cpu().numpy(), pred_label))
            label_total += len(pred_label)
            correct_total += correct
            acc = (correct_total / label_total) * 100

            loss_total += loss
            pbar.update()
            pbar.set_postfix(
            loss=f"{loss_total:.4f}",
            Accuracy=f"{acc:.2f}%"
            )
    pbar.close()

    return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--initial_epoch", type=int, default=0, help="Start epoch")

    # Scheduler parameters
    parser.add_argument("--lr_decay_epoch", type=int, default=25, help="Start to learning rate decay")
    parser.add_argument("--gamma", type=float, default=0.412, help="decay factor of StepLR")
    parser.add_argument("--warmup_ratio", type=float, default=0.06, help="warmup ratio")
    parser.add_argument('--scheduler', type=str, default='step', help='warmup/step')

    # Model parameters
    parser.add_argument('--Sequential_processing', type=str, default='Transformer', help='Transformer/GRU')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Feature extractor(resnet18/resnext50)')
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size of GRU or Transformer")
    parser.add_argument("--num_layer", type=int, default=1, help="number of layers of GRU or Transformer")
    parser.add_argument('--multi_modal', action = 'store_true', help='whether to use multi_modal')

    # Data information
    parser.add_argument('--data_path', type=str, default='../../dataset/', help='path to train')
    parser.add_argument('--dataset_name', type=str, default='Real_life', help='Real_life/Bag_of_lies')
    parser.add_argument('--image_folder', type=str, default=['Real_life'], help='Real_life/Bag_of_lies')
    parser.add_argument('--saved_model', type=str, default='', help='path to model to continue training')

    # Optimizer parameters
    parser.add_argument('--opt', type=str, default='adam', help='Optimizer(adam/sgd)')
    parser.add_argument("--lr", type=float, default=3e-4, help="adam: learning rate")

    parser.add_argument("--batch_size", type=int, default=4, help="batch_size")
    parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu workers')
    
    parser.add_argument('--saved_name', type=str, default='Real_life_GRU', help='The name you want to save')
    parser.add_argument("--save_model_path", type=str, default='./saved_models/multi_modal', help="Path to saved_model")
    opt = parser.parse_args()
    
    os.makedirs('%s/%s' % (opt.save_model_path, opt.saved_name), exist_ok=True)

    # Set random seed
    seed = 412
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # load data
    train_data_list, valid_data_list = train_val_split(opt.data_path, os.path.join(opt.data_path, opt.dataset_name + '.csv'), opt.dataset_name)
    
    train_data = lie_data(opt, train_data_list)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
    val_data = lie_data(opt, valid_data_list)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

    print('Number of training data:', len(train_data))
    print('Number of validation data:', len(val_data))

    # Load model
    if opt.Sequential_processing == 'GRU':
        if opt.multi_modal:
            print('Use multi_modal with CNN_GRU!')
            model = multi_modal_GRU(opt).cuda()
        else:
            print('Use CNN_GRU model!!')
            model = CNN_GRU(opt)

    elif opt.Sequential_processing == 'Transformer':
        if opt.multi_modal:
            print('Use multi_modal with CNN_Transformer!')
            model = multi_modal_Transformer(opt).cuda()
        else:
            print('Use CNN_Transformer model!!')
            model = CNN_Transformer(opt)

    print('Start training!!')
    train(opt, model, train_loader, val_loader)

    