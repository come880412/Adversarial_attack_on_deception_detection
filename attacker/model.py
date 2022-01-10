'''
Modified Date: 2022/01/11
Author: Gi-Luen Huang
mail: come880412@gmail.com
'''

import torch
import torch.nn as nn
import torchvision


class Model(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

        if backbone == 'resnet18':
            self.model = torchvision.models.resnet18(pretrained=True)
            self.model.fc = nn.Linear(512, 2)
        elif backbone == 'alexnet':
            self.model = torchvision.models.alexnet(pretrained=True)
            self.model.classifier[6] = nn.Linear(4096, 2)
        elif backbone == 'vgg16':
            self.model = torchvision.models.vgg16(pretrained=True)
            self.model.classifier[6] = nn.Linear(4096, 2)
        elif backbone == 'resnet50':
            self.model = torchvision.models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(2048, 2)
        elif backbone == 'resnext50_32x4d':
            self.model = torchvision.models.resnext50_32x4d(pretrained=True)
            self.model.fc = nn.Linear(2048, 2)
        
    def forward(self, images):
        # images: 10 x batch_size x 3 x H x W
        feature_pool = []
        for pointer in range(len(images)):
            image = images[pointer]    # batch_size x 3 x H x W
            output = self.model(image)    # batch_size x 2
            output = output.unsqueeze(1)    # batch_size x 1 x 2
            feature_pool.append(output)
        
        feature_pool = torch.cat(feature_pool, dim=1)
        out = feature_pool.mean(1)
        return out

class Model_spec(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

        if backbone == 'resnet18':
            self.model = torchvision.models.resnet18(pretrained=True)
            self.model.fc = nn.Linear(512, 2)
        elif backbone == 'alexnet':
            self.model = torchvision.models.alexnet(pretrained=True)
            self.model.classifier[6] = nn.Linear(4096, 2)
        elif backbone == 'vgg16':
            self.model = torchvision.models.vgg16(pretrained=True)
            self.model.classifier[6] = nn.Linear(4096, 2)
        elif backbone == 'resnet50':
            self.model = torchvision.models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(2048, 2)
        elif backbone == 'resnext50_32x4d':
            self.model = torchvision.models.resnext50_32x4d(pretrained=True)
            self.model.fc = nn.Linear(2048, 2)
        
    def forward(self, images):
        # images: batch_size x 3 x H x W
        output = self.model(images)    # batch_size x 2
        return output

class ensembleNet(nn.Module):
    def __init__(self, save_folder):
        super().__init__()

        self.resnet18 = torchvision.models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Linear(512, 2)
        state_dict = torch.load('./{}/resnet18.pth'.format(save_folder))
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '')] = state_dict.pop(key)
        self.resnet18.load_state_dict(state_dict)

        self.alexnet = torchvision.models.alexnet(pretrained=False)
        self.alexnet.classifier[6] = nn.Linear(4096, 2)
        state_dict = torch.load('./{}/alexnet.pth'.format(save_folder))
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '')] = state_dict.pop(key)
        self.alexnet.load_state_dict(state_dict)

        self.vgg16 = torchvision.models.vgg16(pretrained=False)
        self.vgg16.classifier[6] = nn.Linear(4096, 2)
        state_dict = torch.load('./{}/vgg16.pth'.format(save_folder))
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '')] = state_dict.pop(key)
        self.vgg16.load_state_dict(state_dict)

        self.resnet50 = torchvision.models.resnet50(pretrained=False)
        self.resnet50.fc = nn.Linear(2048, 2)
        state_dict = torch.load('./{}/resnet50.pth'.format(save_folder))
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '')] = state_dict.pop(key)
        self.resnet50.load_state_dict(state_dict)

        self.resnext50 = torchvision.models.resnext50_32x4d(pretrained=False)
        self.resnext50.fc = nn.Linear(2048, 2)
        state_dict = torch.load('./{}/resnext50_32x4d.pth'.format(save_folder))
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '')] = state_dict.pop(key)
        self.resnext50.load_state_dict(state_dict)

        self.models = nn.ModuleList([self.alexnet, self.vgg16, self.resnet18, self.resnet50, self.resnext50])

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, images):
        # images: 10 x batch_size x 3 x H x W
        feature_pool = []
        for pointer in range(len(images)):
            image = images[pointer]    # batch_size x 3 x H x W

            # ensemble
            for id, model in enumerate(self.models):
                output = model(image)    # batch_size x 2
                prob = self.softmax(output)
                prob = prob.unsqueeze(2)
                if id == 0:
                    out = prob    # batch_size x 2 x 1
                else:
                    out = torch.cat((out, prob), dim=2)
                
            out = out.sum(dim=2)
            out = out.unsqueeze(1)
            feature_pool.append(out)
        
        feature_pool = torch.cat(feature_pool, dim=1)
        out = feature_pool.mean(1)
        return out

class ensembleNet_spec(nn.Module):
    def __init__(self, save_folder):
        super().__init__()

        self.resnet18 = torchvision.models.resnet18(pretrained=False)
        self.resnet18.fc = nn.Linear(512, 2)
        state_dict = torch.load('./{}/resnet18.pth'.format(save_folder))
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '')] = state_dict.pop(key)
        self.resnet18.load_state_dict(state_dict)

        self.alexnet = torchvision.models.alexnet(pretrained=False)
        self.alexnet.classifier[6] = nn.Linear(4096, 2)
        state_dict = torch.load('./{}/alexnet.pth'.format(save_folder))
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '')] = state_dict.pop(key)
        self.alexnet.load_state_dict(state_dict)

        self.vgg16 = torchvision.models.vgg16(pretrained=False)
        self.vgg16.classifier[6] = nn.Linear(4096, 2)
        state_dict = torch.load('./{}/vgg16.pth'.format(save_folder))
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '')] = state_dict.pop(key)
        self.vgg16.load_state_dict(state_dict)

        self.resnet50 = torchvision.models.resnet50(pretrained=False)
        self.resnet50.fc = nn.Linear(2048, 2)
        state_dict = torch.load('./{}/resnet50.pth'.format(save_folder))
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '')] = state_dict.pop(key)
        self.resnet50.load_state_dict(state_dict)

        self.resnext50 = torchvision.models.resnext50_32x4d(pretrained=False)
        self.resnext50.fc = nn.Linear(2048, 2)
        state_dict = torch.load('./{}/resnext50_32x4d.pth'.format(save_folder))
        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '')] = state_dict.pop(key)
        self.resnext50.load_state_dict(state_dict)

        self.models = nn.ModuleList([self.alexnet, self.vgg16, self.resnet18, self.resnet50, self.resnext50])

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, images):
        # ensemble
        for id, model in enumerate(self.models):
            output = model(images)    # batch_size x 2
            prob = self.softmax(output)
            prob = prob.unsqueeze(1)
            if id == 0:
                out = prob    # batch_size x 1 x 2 
            else:
                out = torch.cat((out, prob), dim=1)
        
        out = out.mean(1)

        return out