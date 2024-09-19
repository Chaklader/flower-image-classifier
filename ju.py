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


# Configure model for training
# def build_classifier(model, args, dataloaders):
#     in_size = {
#         'densenet121': 1024,
#         'densenet161': 2208,
#         'vgg16': 25088,
#     }
#
#     if args.arch not in in_size:
#         raise ValueError(f"Unsupported architecture: {args.arch}")
#
#     output_size = len(dataloaders['train'].dataset.classes)
#
#     hidden_layers = [int(units) for units in args.hidden_units.split(',')] if args.hidden_units else [500]
#
#     layers = []
#     layer_sizes = [in_size[args.arch]] + hidden_layers
#
#     for i in range(len(layer_sizes) - 1):
#         layers.extend([
#             nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
#             nn.ReLU(),
#             nn.Dropout(args.dropout)
#         ])
#
#     layers.extend([
#         nn.Linear(layer_sizes[-1], output_size),
#         nn.LogSoftmax(dim=1)
#     ])
#
#     model.classifier = nn.Sequential(*layers)
#
#     print("\nClassifier architecture:")
#     for i, layer in enumerate(model.classifier):
#         print(f"  Layer {i}: {layer}")
#
#     return model

# def configure_model_for_training(args, dataloaders, model):
#     if args.arch == 'resnet18':
#         model.fc = nn.Linear(model.fc.in_features, len(dataloaders['train'].dataset.classes))
#         print(f'\n*** model architecture: {args.arch}')
#         print(f'*** fc:\n{model.fc}\n')
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
#     else:
#         model = build_classifier(model, args, dataloaders)
#         print(f'\n*** model architecture: {args.arch}')
#         print(f'*** Classifier:\n{model.classifier}\n')
#         criterion = nn.NLLLoss()
#         optimizer = optim.Adam(model.classifier.parameters(), args.learning_rate)
#     return criterion, model, optimizer
#
# criterion, model, optimizer = configure_model_for_training(args, dataloaders, model)

# def train_one_epoch(model, dataloader, optimizer, criterion, device):
#     model.train()
#     running_loss = 0.0
#     running_corrects = 0
#     total_samples = 0
#
#     for inputs, labels in tqdm(dataloader, desc='Training'):
#         inputs, labels = inputs.to(device), labels.to(device)
#
#         optimizer.zero_grad()
#
#         outputs = model(inputs)
#         _, preds = torch.max(outputs, 1)
#         loss = criterion(outputs, labels)
#
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item() * inputs.size(0)
#         # noinspection PyTypeChecker
#         running_corrects += torch.sum(preds == labels.data).item()
#         total_samples += inputs.size(0)
#
#     epoch_loss = running_loss / total_samples
#     epoch_acc = running_corrects / total_samples
#
#     return epoch_loss, epoch_acc
#
#
# def validate(model, dataloader, criterion, device):
#     model.eval()
#     running_loss = 0.0
#     running_corrects = 0
#     total_samples = 0
#
#     with torch.no_grad():
#         for inputs, labels in tqdm(dataloader, desc='Validating'):
#             inputs, labels = inputs.to(device), labels.to(device)
#
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             loss = criterion(outputs, labels)
#
#             running_loss += loss.item() * inputs.size(0)
#             # noinspection PyTypeChecker
#             running_corrects += torch.sum(preds == labels.data).item()
#             total_samples += inputs.size(0)
#
#     epoch_loss = running_loss / total_samples
#     epoch_acc = running_corrects / total_samples
#
#     return epoch_loss, epoch_acc
#
# def train(model, dataloaders, optimizer, criterion, epochs=3, print_freq=20, lr=0.001, device='cpu', patience=5):
#     model.to(device)
#     start_time = datetime.now()
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)
#
#     print(f'Epochs: {epochs}, Print frequency: {print_freq}, Initial LR: {lr}\n')
#
#     best_val_loss = float('inf')
#     best_model_wts = model.state_dict()
#     no_improve = 0
#
#     for epoch in range(epochs):
#         print(f'Epoch {epoch + 1}/{epochs}')
#         print('-' * 10)
#
#         train_loss, train_acc = train_one_epoch(model, dataloaders['train'], optimizer, criterion, device)
#         val_loss, val_acc = validate(model, dataloaders['valid'], criterion, device)
#
#         print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
#         print(f'Valid Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
#
#         scheduler.step(val_loss)
#
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_model_wts = model.state_dict()
#             no_improve = 0
#         else:
#             no_improve += 1
#
#         if no_improve == patience:
#             print("Early stopping")
#             break
#
#         print()
#
#     elapsed = datetime.now() - start_time
#     print(f'\nTraining complete in {elapsed}')
#     print(f'Best val Loss: {best_val_loss:4f}')
#
#     model.load_state_dict(best_model_wts)
#     return model

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