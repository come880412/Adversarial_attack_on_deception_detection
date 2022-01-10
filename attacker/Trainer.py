import torch
import numpy as np
import tqdm
import os

def pretrain_image(args, epochs, model, criterion, optimizer, scheduler, train_loader, valid_loader, device):
    save_threshold = 0
    for epoch in range(epochs):
        # set model to train mode
        model.train()
        for images, labels, image_names in tqdm.tqdm(train_loader):
            # set image names to list
            for id in range(len(image_names)):
                image_names[id] = list(image_names[id])
            
            image_names = np.array(image_names)    # 10 x batch_size

            for id in range(len(images)):
                images[id] = torch.FloatTensor(images[id])
                images[id] = images[id].to(device)
            labels = torch.LongTensor(labels)
            labels = labels.to(device)

            out = model(images)

            optimizer.zero_grad()
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        
        # set model to eval mode
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels, image_names in valid_loader:
                # set image names to list
                for id in range(len(image_names)):
                    image_names[id] = list(image_names[id])
                
                image_names = np.array(image_names)    # 10 x batch_size

                for id in range(len(images)):
                    images[id] = torch.FloatTensor(images[id])
                    images[id] = images[id].to(device)
                labels = torch.LongTensor(labels)
                labels = labels.to(device)

                out = model(images)
                pred = out.max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()
        
        valid_acc = (correct / len(valid_loader.dataset))
        print('epoch: {}/{} valid_acc: {}'.format(epoch+1, epochs, valid_acc))
        if valid_acc >= save_threshold:
            os.makedirs('./image_model', exist_ok=True)
            save_threshold = valid_acc
            torch.save(model.state_dict(), './image_model/{}.pth'.format(args.backbone))
        
        scheduler.step()

def pretrain_spec(args, epochs, model, criterion, optimizer, scheduler, train_loader, valid_loader, device):
    save_threshold = 0
    for epoch in range(epochs):
        # set model to train mode
        model.train()
        for images, labels, image_names in tqdm.tqdm(train_loader):
            images = torch.FloatTensor(images)
            labels = torch.LongTensor(labels)

            images = images.to(device)
            labels = labels.to(device)

            out = model(images)

            optimizer.zero_grad()
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        
        # set model to eval mode
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels, image_names in valid_loader:
                images = torch.FloatTensor(images)
                labels = torch.LongTensor(labels)

                images = images.to(device)
                labels = labels.to(device)

                out = model(images)
                pred = out.max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()
        
        valid_acc = (correct / len(valid_loader.dataset))
        print('epoch: {}/{} valid_acc: {}'.format(epoch+1, epochs, valid_acc))
        if valid_acc >= save_threshold:
            os.makedirs('./spec_model', exist_ok=True)
            save_threshold = valid_acc
            torch.save(model.state_dict(), './spec_model/{}.pth'.format(args.backbone))
        
        scheduler.step()