import glob
import json
import os
import sys

import torch
from torchvision import datasets, transforms, models

from helper.constant import chkptdir


def check_point():
    if len(glob.glob(chkptdir + '/*.pth')) > 0:
        checkpt = max(glob.glob(chkptdir + '/*.pth'), key=os.path.getctime)
    else:
        checkpt = None
        print('\n*** no saved checkpoint to load ... exiting\n')
        sys.exit(1)

    return checkpt


def load_categories():
    with open('./cat_to_name.json', 'r') as f:
        return json.load(f)


def transform_load(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # define transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
    }

    image_datasets = {k: datasets.ImageFolder(os.path.join(args.data_dir, k), transform=data_transforms[k])
                      for k in ['train', 'valid', 'test']}

    dataloaders = {k: torch.utils.data.DataLoader(image_datasets[k], batch_size=64, shuffle=True)
                   for k in ['train', 'valid', 'test']}

    return dataloaders, image_datasets


def load_checkpoint(device, args):
    if device.type == 'cuda':
        print('*** loading chkpt', args.checkpoint, ' in cuda ...\n')
        checkpoint = torch.load(args.checkpoint)
    else:
        print('*** loading chkpt', args.checkpoint, ' in cpu ...\n')
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)

    model = models.__dict__[checkpoint['arch']](pretrained=True)
    if checkpoint['arch'] == 'resnet18':
        model.fc = checkpoint['fc']
        print('architecture:', checkpoint['arch'], '\nmodel.fc:\n', model.fc, '\n')
    else:
        model.classifier = checkpoint['classifier']
        print('architecture:', checkpoint['arch'], '\nmodel.classifier:\n', model.classifier, '\n')
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model, checkpoint['arch']
