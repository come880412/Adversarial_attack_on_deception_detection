'''
Modified Date: 2022/01/11
Author: Gi-Luen Huang
mail: come880412@gmail.com
'''

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

from utils import Smoother

class multi_modal_GRU(nn.Module):
    def __init__(self, opt):
        super(multi_modal_GRU, self).__init__()
        pretrained_model = models.resnet18(pretrained=True)
        self.audio_FE = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.frame_FE = nn.Sequential(*list(pretrained_model.children())[:-1])

        self.d_model = 512
        self.num_layer = opt.num_layer
        self.hidden_size = opt.hidden_size


        self.GRU = nn.GRU(self.d_model, self.hidden_size, self.num_layer, dropout=0, bidirectional=True, batch_first = True)
        self.Classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 2)

        )

    def init_hidden(self, batch_size):
        if torch.cuda.is_available():
            return (Variable(torch.zeros(2 * self.num_layer, batch_size, self.hidden_size)).cuda())
        else:
            return (Variable(torch.zeros(2 * self.num_layer, batch_size, self.hidden_size)))

    def forward(self, input_visual, input_audio):
        hidden1 = self.init_hidden(input_visual.shape[0])
        
        for frame_idx in range(input_visual.shape[1]):
            frame = input_visual[:,frame_idx,:,:,:]
            frame = self.frame_FE(frame).squeeze()
            if frame.dim() == 1: # If batch_size = 1
                frame = frame.unsqueeze(0)
            frame = frame.unsqueeze(1) # pad length
            if frame_idx == 0:
                frame_out = frame
            else:
                frame_out = torch.cat((frame_out, frame), dim=1)
        # input = (batch_size, length, embeddings)
        _, h_1 = self.GRU(frame_out, hidden1)
        frame_feat = torch.sum(h_1, dim=0)
        audio_feat = self.audio_FE(input_audio).squeeze()

        fusion_feat = torch.cat((frame_feat, audio_feat), dim=1)

        out = self.Classifier(fusion_feat)

        return out

class multi_modal_Transformer(nn.Module):
    def __init__(self, opt, n_classes=2, dropout=0.1):
        super(multi_modal_Transformer, self).__init__()
        pretrained_model = models.resnet18(pretrained=True)
        self.audio_FE = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.frame_FE = nn.Sequential(*list(pretrained_model.children())[:-1])

        self.d_model = 512
        self.num_layer = opt.num_layer
        self.hidden_size = opt.hidden_size


        self.encoder_layer = Smoother(
        d_model=self.d_model, dim_feedforward=1024, nhead=2, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        # Project the the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Linear(self.d_model, n_classes)
        self.Classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 2)

        )

    def forward(self, input_visual, input_audio):
        for frame_idx in range(input_visual.shape[1]):
            frame = input_visual[:,frame_idx,:,:,:]
            frame = self.frame_FE(frame).squeeze()
            if frame.dim() == 1: # If batch_size = 1
                frame = frame.unsqueeze(0)
            frame = frame.unsqueeze(1) # pad length
            if frame_idx == 0:
                out = frame
            else:
                out = torch.cat((out, frame), dim=1)
        # out: (length, batch size, d_model)
        out = out.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.encoder(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)
        # mean pooling
        frame_feat = out.mean(dim=1)
        audio_feat = self.audio_FE(input_audio).squeeze()

        fusion_feat = torch.cat((frame_feat, audio_feat), dim=1)

        out = self.Classifier(fusion_feat)

        return out
 
class CNN_GRU(nn.Module):
    def __init__(self, opt):
        super(CNN_GRU, self).__init__()
        if opt.backbone == 'resnet18':
            pretrained_model = models.resnet18(pretrained=True)
            self.d_model = 512
        elif opt.backbone == 'resnext50':
            pretrained_model = models.resnext50_32x4d(pretrained=True)
            self.d_model = 2048

        self.num_layer = opt.num_layer
        self.hidden_size = opt.hidden_size
        self.Feature_Extractor = nn.Sequential(*list(pretrained_model.children())[:-1])

        self.GRU = nn.GRU(self.d_model, self.hidden_size, self.num_layer, dropout=0, bidirectional=True, batch_first = True)

        # Fully-connected layer
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.BatchNorm1d(self.hidden_size//2),
            nn.ReLU(True),
            nn.Linear(self.hidden_size//2, self.hidden_size//4),
            nn.BatchNorm1d(self.hidden_size//4),
            nn.ReLU(True),
            nn.Linear(self.hidden_size//4, 2)
            )
        # self.fc = nn.Linear(self.hidden_size, 2)
        

    def init_hidden(self, batch_size):
        if torch.cuda.is_available():
            return (Variable(torch.zeros(2 * self.num_layer, batch_size, self.hidden_size)).cuda())
        else:
            return (Variable(torch.zeros(2 * self.num_layer, batch_size, self.hidden_size)))

    def forward(self, input):
        hidden1 = self.init_hidden(input.shape[0])
        
        for frame_idx in range(input.shape[1]):
            frame = input[:,frame_idx,:,:,:]
            frame = self.Feature_Extractor(frame).squeeze()
            if frame.dim() == 1: # If batch_size = 1
                frame = frame.unsqueeze(0)
            frame = frame.unsqueeze(1) # pad length
            if frame_idx == 0:
                frame_out = frame
            else:
                frame_out = torch.cat((frame_out, frame), dim=1)
        # input = (batch_size, length, embeddings)
        out, h_1 = self.GRU(frame_out, hidden1)
        h_1 = torch.sum(h_1, dim=0)
        out = self.fc(h_1)
        return out

class CNN_Transformer(nn.Module):
  def __init__(self, opt, n_classes=2, dropout=0.1):
    super().__init__()
    if opt.backbone == 'resnet18':
        pretrained_model = models.resnet18(pretrained=True)
        self.d_model = 512
    elif opt.backbone == 'resnext50':
        pretrained_model = models.resnext50_32x4d(pretrained=True)
        self.d_model = 2048

    self.Feature_Extractor = nn.Sequential(*list(pretrained_model.children())[:-1])
    self.encoder_layer = Smoother(
      d_model=self.d_model, dim_feedforward=1024, nhead=2, dropout=dropout
    )
    self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

    # Project the the dimension of features from d_model into speaker nums.
    self.pred_layer = nn.Linear(self.d_model, n_classes)

  def forward(self, input):
    for frame_idx in range(input.shape[1]):
        frame = input[:,frame_idx,:,:,:]
        frame = self.Feature_Extractor(frame).squeeze()
        if frame.dim() == 1: # If batch_size = 1
            frame = frame.unsqueeze(0)
        frame = frame.unsqueeze(1) # pad length
        if frame_idx == 0:
            out = frame
        else:
            out = torch.cat((out, frame), dim=1)
    # out: (length, batch size, d_model)
    out = out.permute(1, 0, 2)
    # The encoder layer expect features in the shape of (length, batch size, d_model).
    out = self.encoder(out)
    # out: (batch size, length, d_model)
    out = out.transpose(0, 1)
    # mean pooling
    stats = out.mean(dim=1)

    # out: (batch, n_spks)
    out = self.pred_layer(stats)
    return out

if __name__ == '__main__':
    pass
    
