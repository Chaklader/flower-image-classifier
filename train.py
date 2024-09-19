#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a new network on a dataset of images.

This script allows users to train a neural network on an image dataset with the following features:
- Prints out training loss, validation loss, and validation accuracy as the network trains
- Allows users to choose from at least two different architectures available from torchvision.models
- Allows users to set hyperparameters for learning rate, number of hidden units, and training epochs
- Allows users to choose training the model on a GPU

Usage examples:
1. Use data_dir 'flowers':
   python train.py flowers

2. Use save_dir 'checkpoints' to save checkpoint:
   python train.py --save_dir checkpoints

3. Use densenet161 and hidden_units '1000, 500':
   python train.py --arch densenet161 -hu '1000, 500'

4. Set epochs to 10:
   python train.py -e 10

5. Set learning rate to 0.002 and dropout to 0.3:
   python train.py -lr 0.002 -dout 0.3

6. Train in GPU mode (subject to device capability):
   python train.py --gpu

For more options and information, use:
python train.py --help
"""

import os
from datetime import datetime

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from tqdm import tqdm

from helper.common_utils import print_device_info, set_device_for_gpu
from helper.input import get_train_input_args
from helper.load import transform_load
from helper.validate import validate_train_args



# def test(model, dataloaders, criterion):
#     print('*** validating testset ...\n')
#     model.cpu()
#     model.eval()
#
#     test_loss = 0
#     total = 0
#     match = 0
#
#     start_time = datetime.now()
#
#     with torch.no_grad():
#         for images, labels in iter(dataloaders['test']):
#             model, images, labels = model.to(device), images.to(device), labels.to(device)
#
#             output = model.forward(images)
#             test_loss += criterion(output, labels).item()
#             total += images.shape[0]
#             equality = labels.data == torch.max(output, 1)[1]
#             match += equality.sum().item()
#
#     model.test_accuracy = match / total * 100
#     print('Test Loss: {:.3f}'.format(test_loss / len(dataloaders['test'])),
#           'Test Accuracy: {:.2f}%'.format(model.test_accuracy))
#
#     elapsed = datetime.now() - start_time
#     print('\n*** test validation done ! \nElapsed time[hh:mm:ss.ms]: {}'.format(elapsed))
#
def test(model, dataloaders, criterion, device):
    print('\n*** Evaluating model on test set ***\n')
    model.to(device)
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    start_time = datetime.now()

    with torch.no_grad():
        for images, labels in tqdm(dataloaders['test'], desc='Testing'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / total
    accuracy = 100 * correct / total

    print(f'Test Loss: {avg_loss:.4f}')
    print(f'Test Accuracy: {accuracy:.2f}%')

    elapsed = datetime.now() - start_time
    print(f'\n*** Test evaluation completed ***')
    print(f'Elapsed time: {elapsed}')

    return avg_loss, accuracy


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for inputs, labels in tqdm(dataloader, desc='Training'):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        # noinspection PyTypeChecker
        running_corrects += torch.sum(preds == labels.data).item()
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc

# def validate(model, dataloaders, criterion):
#     valid_loss = 0
#     accuracy = 0
#
#     for images, labels in iter(dataloaders['valid']):
#         images, labels = images.to(device), labels.to(device)
#
#         output = model.forward(images)
#         valid_loss += criterion(output, labels).item()
#         ps = torch.exp(output)
#         equality = (labels.data == ps.max(dim=1)[1])
#         accuracy += equality.type(torch.FloatTensor).mean()
#
#     return valid_loss, accuracy
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Validating'):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            # noinspection PyTypeChecker
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc

# training
# def train(model, dataloaders, optimizer, criterion, epochs=2, print_freq=20, lr=0.001):
#     if torch.cuda.is_available():
#         print('*** training classifier in GPU mode ...\n')
#     else:
#         print('*** training classifier in CPU mode ...\n')
#
#     model.to(device)
#     start_time = datetime.now()
#
#     print('epochs:', epochs, ', print_freq:', print_freq, ', lr:', lr, '\n')
#
#     steps = 0
#
#     for e in range(epochs):
#         model.train()
#         running_loss = 0
#         for images, labels in iter(dataloaders['train']):
#             steps += 1
#
#             images, labels = images.to(device), labels.to(device)
#
#             optimizer.zero_grad()
#
#             output = model.forward(images)
#             loss = criterion(output, labels)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#
#             if steps % print_freq == 0:
#                 model.eval()
#
#                 with torch.no_grad():
#                     valid_loss, accuracy = validate(model, dataloaders, criterion)
#
#                 print('Epoch: {}/{}..'.format(e + 1, epochs),
#                       'Training Loss: {:.3f}..'.format(running_loss / print_freq),
#                       'Validation Loss: {:.3f}..'.format(valid_loss / len(dataloaders['valid'])),
#                       'Validation Accuracy: {:.3f}%'.format(accuracy / len(dataloaders['valid']) * 100)
#                       )
#                 running_loss = 0
#
#                 model.train()
#
#     elapsed = datetime.now() - start_time
#
#     print('\n*** classifier training done ! \nElapsed time[hh:mm:ss.ms]: {}'.format(elapsed))
#
#     return model
def train(model, dataloaders, optimizer, criterion, epochs=3, print_freq=20, lr=0.001, device='cpu', patience=5):
    model.to(device)
    start_time = datetime.now()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)

    print(f'Epochs: {epochs}, Print frequency: {print_freq}, Initial LR: {lr}\n')

    best_val_loss = float('inf')
    best_model_wts = model.state_dict()
    no_improve = 0

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], optimizer, criterion, device)
        val_loss, val_acc = validate(model, dataloaders['valid'], criterion, device)

        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Valid Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve == patience:
            print("Early stopping")
            break

        print()

    elapsed = datetime.now() - start_time
    print(f'\nTraining complete in {elapsed}')
    print(f'Best val Loss: {best_val_loss:4f}')

    model.load_state_dict(best_model_wts)
    return model


# def build_classifier(model, args):
#     # Freeze parameters so we don't backprop through them
#     for param in model.parameters():
#         param.requires_grad = False
#
#     in_size = {
#         'densenet121': 1024,
#         'densenet161': 2208,
#         'vgg16': 25088,
#     }
#
#     hid_size = {
#         'densenet121': [500],
#         'densenet161': [1000, 500],
#         'vgg16': [4096, 4096, 1000],
#     }
#
#     if args.z_dpout:
#         p = args.z_dpout
#     else:
#         p = 0.5
#
#     output_size = len(dataloaders['train'].dataset.classes)
#     relu = nn.ReLU()
#     dropout = nn.Dropout(p)
#     output = nn.LogSoftmax(dim=1)
#
#     if args.z_hid:
#         h_list = args.z_hid.split(',')
#         h_list = list(map(int, h_list))  # convert list from string to int
#     else:
#         h_list = hid_size[args.z_arch]
#
#     h_layers = [nn.Linear(in_size[args.z_arch], h_list[0])]
#     h_layers.append(relu)
#     if args.z_arch[:3] == 'vgg':
#         h_layers.append(dropout)
#
#     if len(h_list) > 1:
#         h_sz = zip(h_list[:-1], h_list[1:])
#         for h1, h2 in h_sz:
#             h_layers.append(nn.Linear(h1, h2))
#             h_layers.append(relu)
#             if args.z_arch[:3] == 'vgg':
#                 h_layers.append(dropout)
#
#     last = nn.Linear(h_list[-1], output_size)
#     h_layers.append(last)
#     h_layers.append(output)
#
#     print(h_layers)
#     model.classifier = nn.Sequential(*h_layers)
#
#     return model

def build_classifier(model, args, dataloaders):
    in_size = {
        'densenet121': 1024,
        'densenet161': 2208,
        'vgg16': 25088,
    }

    default_hidden_sizes = {
        'densenet121': [500],
        'densenet161': [1000, 500],
        'vgg16': [4096, 4096, 1000],
    }

    if args.arch not in in_size:
        raise ValueError(f"Unsupported architecture: {args.arch}")

    output_size = len(dataloaders['train'].dataset.classes)

    hidden_layers = [int(units) for units in args.hidden_units.split(',')] if args.hidden_units else \
        default_hidden_sizes[args.arch]

    layers = []
    layer_sizes = [in_size[args.arch]] + hidden_layers

    for i in range(len(layer_sizes) - 1):
        layers.extend([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
            nn.ReLU(),
            nn.Dropout(args.dropout)
        ])

    layers.extend([
        nn.Linear(layer_sizes[-1], output_size),
        nn.LogSoftmax(dim=1)
    ])

    model.classifier = nn.Sequential(*layers)

    print("\nClassifier architecture:")
    for i, layer in enumerate(model.classifier):
        print(f"  Layer {i}: {layer}")

    return model


def configure_model_for_training(args, dataloaders, model):
    if args.arch == 'resnet18':
        model.fc = nn.Linear(model.fc.in_features, len(dataloaders['train'].dataset.classes))
        print(f'\n*** model architecture: {args.arch}')
        print(f'*** fc:\n{model.fc}\n')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    else:
        model = build_classifier(model, args, dataloaders)
        print(f'\n*** model architecture: {args.arch}')
        print(f'*** Classifier:\n{model.classifier}\n')
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate)
    return criterion, model, optimizer


def main():
    args = get_train_input_args()
    device = set_device_for_gpu()

    print('\n*** command line arguments ***')
    print(f'architecture: {args.arch}\ndata dir: {args.data_dir}\nchkpt dir: {args.save_dir}'
          f'\nlearning rate: {args.learning_rate}\ndropout: {args.dropout}'
          f'\nhidden layer: {args.hidden_units}\nepochs: {args.epochs}\nGPU mode: {args.gpu}\n')

    validate_train_args(args)
    print_device_info(device, args)

    dataloaders, image_datasets = transform_load(args)
    model = models.__dict__[args.arch](pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    criterion, model, optimizer = configure_model_for_training(args, dataloaders, model)

    model = model.cpu()
    model.class_to_idx = dataloaders['train'].dataset.class_to_idx

    os.makedirs(args.save_dir, exist_ok=True)

    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer': optimizer.state_dict(),
        'arch': args.arch,
        'lrate': args.learning_rate,
        'epochs': args.epochs,
        'fc' if args.arch == 'resnet18' else 'classifier': model.fc if args.arch == 'resnet18' else model.classifier
    }

    chkpt = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.arch}.pth"
    checkpt = os.path.join(args.save_dir, chkpt)

    torch.save(checkpoint, checkpt)
    print(f'\n*** checkpoint: {chkpt}, saved to: {os.path.dirname(checkpt)}\n')

    print("Starting training...")
    model = train(model, dataloaders, optimizer, criterion, args.epochs, 40, args.learning_rate, device)
    print("Training completed.")

    print("Starting testing...")
    test_loss, test_accuracy = test(model, dataloaders, criterion, device)
    print(f"Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_accuracy:.2f}%")
    print("Testing completed.")


if __name__ == "__main__":
    main()
