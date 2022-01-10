'''
Modified Date: 2022/01/11
Author: Gi-Luen Huang
mail: come880412@gmail.com
'''

import torch

import os
import numpy as np
from PIL import Image
from torchvision.transforms.autoaugment import TrivialAugmentWide
import tqdm

from model import Model, Model_spec

def create_model(backbone, input_type):
    if input_type == 'image':
        model = Model(backbone)
    elif input_type == 'audio':
        model = Model_spec(backbone)

    return model

class audio_attack():
    def __init__(self, args, epsilon, alpha, mean, std, device):
        self.epsilon = epsilon
        self.alpha = alpha
        self.mean = mean
        self.std = std
        self.device = device
        self.args = args

    def fgsm(self, model, x, y, loss_fn):
        x_adv = x.detach().clone()    # initialize x_adv as original benign image x
        x_adv.requires_grad = True    # need to obtain gradient of x_adv, thus set required grad
        loss = loss_fn(model(x_adv), y)    # calculate loss
        loss.backward()    # calculate gradient
        # fgsm: use gradient ascent on x_adv to maximize loss
        x_adv = x_adv + self.epsilon * x_adv.grad.detach().sign()
        return x_adv

    def ifgsm(self, model, x, y, loss_fn, num_iter=20):
        x_adv = x.detach().clone()    # initialize x_adv as original benign image x
        # write a loop of num_iter to represent the iterative times
        for i in range(num_iter):
            x_adv = self.fgsm(model, x_adv, y, loss_fn, self.alpha)    # call fgsm with (epsilon = alpha) to obtain new x_adv
            x_adv = torch.max(torch.min(x_adv, x + self.epsilon), x - self.epsilon)    # clip new x_adv back to [x-epsilon, x+epsilon]
        return x_adv


    def poison(self, model, x, y, loss_fn, decay=1.0, momentum=None):
        x_adv = x.detach().clone()
        x_adv.requires_grad = True
        loss = loss_fn(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=False, create_graph=False)[0]
        grad /= torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        grad += momentum * decay
        momentum = grad
        x_adv = x_adv + self.epsilon * grad.detach().sign()
        return x_adv, momentum


    def mi_fgsm(self, model, x, y, loss_fn, num_iter=50):
        x_adv = x.detach().clone()    # initialize x_adv as original benign image x
        # write a loop of num_iter to represent the iterative times
        momentum = torch.zeros_like(x_adv, device=x.device)
        for i in range(num_iter):
            x_adv, momentum = self.poison(model, x_adv, y, loss_fn, self.alpha, decay=0.7, momentum=momentum)    # call fgsm with (epsilon = alpha) to obtain new x_adv
            x_adv = torch.max(torch.min(x_adv, x + self.epsilon), x - self.epsilon)    # clip new x_adv back to [x-epsilon, x+epsilon]
        return x_adv
    
    # perform adversarial attack and generate adversarial examples
    def gen_adv_examples(self, model, loader, attack, loss_fn):
        model.eval()
        train_acc, train_loss = 0.0, 0.0
        for i, (x, y, _) in tqdm.tqdm(enumerate(loader)):
            x, y = x.to(self.device), y.to(self.device)
            x_adv = attack(model, x, y, loss_fn)    # obtain adversarial examples
            yp = model(x_adv)
            loss = loss_fn(yp, y)
            train_acc += (yp.argmax(dim=1) == y).sum().item()
            train_loss += loss.item() * x.shape[0]
            # store adversarial examples
            adv_ex = ((x_adv) * self.std + self.mean).clamp(0, 1)    # to 0-1 scale
            adv_ex = (adv_ex * 255).clamp(0, 255)    # 0-255 scale
            adv_ex = adv_ex.detach().cpu().data.numpy().round()    # round to remove decimal part
            adv_ex = adv_ex.transpose((0, 2, 3, 1))    # transpose (bs, C, H, W) back to (bs, H, W, C)
            adv_examples = adv_ex if i == 0 else np.r_[adv_examples, adv_ex]
        return adv_examples, train_acc / len(loader.dataset), train_loss / len(loader.dataset)


    def save_adv_images(self, adv_examples, adv_names):
        os.makedirs(os.path.join(self.args.data_path, 'Real_life_audio_{}_{}'.format(self.args.attack, self.args.backbone)), exist_ok=True)
        for example, name in zip(adv_examples, adv_names):
            im = Image.fromarray(example.astype(np.uint8))
            im.save(os.path.join(self.args.data_path, 'Real_life_audio_{}_{}'.format(self.args.attack, self.args.backbone), name.replace('.wav', '.png')))

class video_attack():
    def __init__(self, args, epsilon, alpha, mean, std, device):
        self.epsilon = epsilon
        self.alpha = alpha
        self.mean = mean
        self.std = std
        self.device = device
        self.args = args
    
    def fgsm(self, model, x, y, loss_fn):
        x_adv = [x_.detach().clone() for x_ in x]
        for id in range(len(x_adv)):
            x_adv[id].requires_grad = True
        
        loss = loss_fn(model(x_adv), y)
        loss.backward()

        # fgsm: use gradient ascent on x_adv to maximize loss
        for id in range(len(x_adv)):
            x_adv[id] = x_adv[id] + self.epsilon * x_adv[id].grad.detach().sign()

        return x_adv


    def ifgsm(self, model, x, y, loss_fn, num_iter=20):
        x_adv = [x_.detach().clone() for x_ in x]    # initialize x_adv as original benign image x
        # write a loop of num_iter to represent the iterative times
        for i in range(num_iter):
            x_adv = self.fgsm(model, x_adv, y, loss_fn, self.alpha)    # call fgsm with (epsilon = alpha) to obtain new x_adv
            x_adv = [torch.max(torch.min(x_adv[id], x[id] + self.epsilon), x[id] - self.epsilon) for id in range(len(x_adv))]

        return x_adv


    def poison(self, model, x, y, loss_fn, decay=1.0, momentums=None):
        x_adv = [x_.detach().clone() for x_ in x]
        for id in range(len(x_adv)):
            x_adv[id].requires_grad = True
        loss = loss_fn(model(x_adv), y)

        grads = []
        for id in range(len(x_adv)):
            if id != (len(x_adv)-1):
                grad = torch.autograd.grad(loss, x_adv[id], retain_graph=True, create_graph=False)[0]
            else:
                grad = torch.autograd.grad(loss, x_adv[id], retain_graph=False, create_graph=False)[0]
            grads.append(grad)

        grads = [grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True) for grad in grads]
        for id in range(len(grads)):
            grads[id] = grads[id] + momentums[id] * decay
        
        momentums = grads
        for id in range(len(x_adv)):
            x_adv[id] = x_adv[id] + self.epsilon * grads[id].detach().sign()

        return x_adv, momentums


    def mi_fgsm(self, model, x, y, loss_fn, num_iter=50):
        x_adv = [x_.detach().clone() for x_ in x]    # initialize x_adv as original benign image x
        # write a loop of num_iter to represent the iterative times
        momentums = [torch.zeros_like(x_adv_, device=x_adv_.device) for x_adv_ in x_adv]
        for i in range(num_iter):
            x_adv, momentums = self.poison(model, x_adv, y, loss_fn, self.alpha, decay=0.7, momentums=momentums)    # call fgsm with (epsilon = alpha) to obtain new x_adv
            x_adv = [torch.max(torch.min(x_adv[id], x[id] + self.epsilon), x[id] - self.epsilon) for id in range(len(x_adv))]    # clip new x_adv back to [x-epsilon, x+epsilon]
        return x_adv


    # perform adversarial attack and generate adversarial examples
    def gen_adv_examples(self, model, loader, attack, loss_fn):
        model.eval()
        adv_examples = []
        train_acc, train_loss = 0.0, 0.0
        for i, (x, y, _) in tqdm.tqdm(enumerate(loader)):
            for id in range(len(x)):
                x[id] = torch.FloatTensor(x[id])
                x[id] = x[id].to(self.device)
            
            y = torch.LongTensor(y)
            y = y.to(self.device)

            if attack == 'fgsm':
                x_adv = self.fgsm(model, x, y, loss_fn)
            if attack == 'ifgsm':
                x_adv = self.ifgsm(model, x, y, loss_fn)
            if attack == 'mifgsm':
                x_adv = self.mi_fgsm(model, x, y, loss_fn)

            yp = model(x_adv)
            loss = loss_fn(yp, y)
            train_acc += (yp.argmax(dim=1) == y).sum().item()
            train_loss += loss.item() * x[0].shape[0]

            # store adversarial examples
            adv_ex = [((x_adv_element) * self.std + self.mean).clamp(0, 1) for x_adv_element in x_adv]
            adv_ex = [(adv_ex_element * 255).clamp(0, 255) for adv_ex_element in adv_ex]
            for id in range(len(adv_ex)):
                adv_ex[id] = adv_ex[id].detach().cpu().data.numpy().round()
            
            for id in range(len(adv_ex)):
                adv_ex[id] = adv_ex[id].transpose((0, 2, 3, 1)).squeeze()
            
            adv_examples.append(adv_ex)

        return adv_examples, train_acc / len(loader.dataset), train_loss / len(loader.dataset)

    def save_adv_images(self, adv_examples, adv_names):
        # adv_examples: 11 x 10 x 224 x 224 x 3
        # adv_names: 11 x 10
        os.makedirs(os.path.join(self.args.data_path, 'Real_life_{}_{}'.format(self.args.attack, self.args.backbone)), exist_ok=True)
        for batch_id in range(len(adv_examples)):
            for frame_id in range(10):
                adv_image = adv_examples[batch_id][frame_id]
                adv_name = adv_names[batch_id][frame_id].replace('Real_life', 'Real_life_{}_{}'.format(self.args.attack, self.args.backbone))
                im = Image.fromarray(adv_image.astype(np.uint8))    # image pixel value should be unsigned int
                im.save(adv_name)