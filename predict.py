#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Classifier Prediction Script

This script reads an image and a checkpoint, then predicts the most likely image class
and its associated probability.

Features:
- Loads a JSON file that maps class values to category names
- Prints out top k classes and probabilities
- Supports GPU prediction (if available)

Usage:
  python predict.py [OPTIONS] [CHECKPOINT]

Arguments:
  CHECKPOINT  Path to the saved checkpoint (default: latest in 'chksav')

Options:
  --img_pth PATH           Path to the image (default: 'flowers/test/91/image_08061.jpg')
  --category_names FILE    Path to category name JSON mapper file (default: 'cat_to_name.json')
  --top_k N                Number of top classes to print (default: 1)
  --gpu                    Use GPU for prediction if available
  --help                   Show this help message and exit

Examples:
  python predict.py chksav/chkpt.pth
  python predict.py --top_k 4 --gpu
  python predict.py --img_pth flowers/test/91/image_08061.jpg --category_names cat_to_name.json
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

from helper.common_utils import print_device_info, set_device_for_gpu
from helper.constant import test_dir
from helper.image import process_image, pick_a_pic
from helper.input import get_predict_input_args
from helper.load import load_categories, transform_load, load_checkpoint
from helper.validate import validate_predict_args


def show_classifier(topk_names, topk_probs, img_path):
    """
    Display the image and a bar chart of top class probabilities.
    """
    img = Image.open(img_path)
    img_name = topk_names[0]

    fig, (ax1, ax2) = plt.subplots(figsize=(10, 4), ncols=2)
    ax1.set_title(img_name)
    ax1.imshow(img)
    ax1.axis('off')

    y_pos = np.arange(len(topk_probs))
    ax2.barh(y_pos, topk_probs)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(topk_names)
    ax2.invert_yaxis()
    ax2.set_title('Class Probability')

    plt.tight_layout()
    plt.show()


# def predict(image_path, cat_to_name, model, topk=5):
#     ''' Predict the class (or classes) of an image using a trained deep learning model.
#     '''
#
#     # TODO: Implement the code to predict the class from an image file
#     model.cpu()
#     model.eval()
#
#     pil_img = Image.open(image_path)
#     image = process_image(pil_img)
#     image = torch.FloatTensor(image)
#
#     model, image = model.to(device), image.to(device)
#
#     print('\nori image.shape:', image.shape)
#     image.unsqueeze_(0)  # add a new dimension in pos 0
#     print('new image.shape:', image.shape, '\n')
#
#     output = model.forward(image)
#
#     # get the top k classes of prob
#     ps = torch.exp(output).data[0]
#     topk_prob, topk_idx = ps.topk(topk)
#
#     # bring back to cpu and convert to numpy
#     topk_probs = topk_prob.cpu().numpy()
#     topk_idxs = topk_idx.cpu().numpy()
#
#     # map topk_idx to classes in model.class_to_idx
#     idx_class = {i: k for k, i in model.class_to_idx.items()}
#     topk_classes = [idx_class[i] for i in topk_idxs]
#
#     # map class to class name
#     topk_names = [cat_to_name[i] for i in topk_classes]
#
#     print('*** Top ', topk, ' classes ***')
#     print('class names:   ', topk_names)
#     print('classes:       ', topk_classes)
#     print('probabilities: ', topk_probs)
#
#     return topk_classes, topk_names, topk_probs

def predict(model, arch, image_path, topk=5, device='cpu'):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """
    model.eval().to(device)
    cat_to_name = load_categories()

    image = process_image(Image.open(image_path))
    image = torch.FloatTensor(image).unsqueeze_(0).to(device)

    with torch.no_grad():
        output = model(image)
        ps = F.softmax(output, dim=1) if arch == 'resnet18' else torch.exp(output)
        topk_prob, topk_idx = ps.topk(topk)

    topk_probs = topk_prob.cpu().numpy()[0]
    topk_idxs = topk_idx.cpu().numpy()[0]

    idx_to_class = {i: k for k, i in model.class_to_idx.items()}
    topk_classes = [idx_to_class[i] for i in topk_idxs]
    topk_names = [cat_to_name[i] for i in topk_classes]

    print(f'*** Top {topk} classes ***')
    print('Class names:   ', topk_names)
    print('Classes:       ', topk_classes)
    print('Probabilities: ', topk_probs)

    return topk_classes, topk_names, topk_probs


def main():
    device = set_device_for_gpu()
    start_time = datetime.now()

    args = get_predict_input_args()
    print('\n*** Command line arguments ***')
    print(f'Checkpoint: {args.checkpoint}\nImage path: {args.img_pth}'
          f'\nCategory names mapper file: {args.category_names}\nNumber of top k: {args.top_k}'
          f'\nGPU mode: {args.gpu}\n')

    validate_predict_args(args)
    print_device_info(device, args)

    dataloaders, _ = transform_load(args)
    model, arch = load_checkpoint(device, args)

    elapsed = datetime.now() - start_time
    print(f'\n*** Prediction setup done!\nElapsed time: {elapsed}\n')

    args = pick_a_pic(dataloaders, test_dir, 'test', args)
    image_path = args.z_rndimgpth
    print(f'Image path: {image_path}\n')

    # Predict the top classes and probabilities for the selected image
    print("Predicting top classes and probabilities...")
    _, topk_names, topk_probs = predict(model, arch, image_path, args.top_k, device)
    print("Prediction complete.")

    # Display the image and a bar chart of the top class probabilities
    print("Displaying classification results...")
    show_classifier(topk_names, topk_probs, image_path)
    print("Classification results displayed. Check the pop-up window.")


if __name__ == "__main__":
    main()
