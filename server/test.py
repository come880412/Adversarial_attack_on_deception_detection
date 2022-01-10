import numpy as np
import argparse
import tqdm
import os
import random

from dataset import lie_data, lie_multi_modal_test
from utils import train_val_split
from model import CNN_GRU, CNN_Transformer, multi_modal_GRU, multi_modal_Transformer

from torch.utils.data import DataLoader
import torch

def test(opt, model, val_loader):
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
    parser.add_argument('--Sequential_processing', type=str, default='Transformer', help='Transformer/GRU')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Feature extractor(resnet18/resnext50)')
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size of GRU or Transformer")
    parser.add_argument("--num_layer", type=int, default=1, help="number of layers of GRU or Transformer")
    parser.add_argument('--multi_modal', action = 'store_true', help='whether to use multi_modal')
    parser.add_argument('--saved_model', type=str, default='./saved_models/multi_modal/Real_life_Transformer/model_epoch34_acc90.91.pth', help='path to model to continue training')
    
    parser.add_argument('--data_path', type=str, default='../dataset/', help='path to train')
    parser.add_argument('--dataset_name', type=str, default='Real_life', help='Real_life/Bag_of_lies')
    parser.add_argument('--image_folder', type=list, default=['Real_life_fgsm_resnet18', 'Real_life_audio_fgsm_resnet18'], help='Real_life/Bag_of_lies')

    parser.add_argument("--batch_size", type=int, default=4, help="batch_size")
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu workers')

    opt = parser.parse_args()
    
    # Set random seed
    seed = 412
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Load data
    _, valid_data_list = train_val_split(os.path.join(opt.data_path, opt.dataset_name + '.csv'), opt.dataset_name)
    if opt.multi_modal:
        val_data = lie_multi_modal_test(opt, valid_data_list)
        val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    else:
        val_data = lie_data(opt, valid_data_list)
        val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)
    print('Number of validation data:', len(val_data))

    # Load model
    if opt.Sequential_processing == 'GRU':
        if opt.multi_modal:
            print('Use multi_modal with CNN_GRU!')
            model = multi_modal_GRU(opt).cuda()
        else:
            print('Use CNN_GRU model!!')
            model = CNN_GRU(opt).cuda()

    elif opt.Sequential_processing == 'Transformer':
        if opt.multi_modal:
            print('Use multi_modal with CNN_Transformer!')
            model = multi_modal_Transformer(opt).cuda()
        else:
            print('Use CNN_Transformer model!!')
            model = CNN_Transformer(opt).cuda()

    print('Load pretrained model from: ', opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model))

    print('Start testing!')
    test(opt, model, val_loader)