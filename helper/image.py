import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image`
# function works, running the output through this function should return the original image (except for the cropped out
# portions).

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax

def pick_a_pic(dataloaders, dset_dir, dset_type, args):

    args.z_imgcls = np.random.choice(dataloaders[dset_type].dataset.classes)
    args.z_rndimg = np.random.choice(os.listdir(dset_dir + '/' + args.z_imgcls))
    args.z_rndimgpth = dset_dir + '/' + args.z_imgcls + '/' + args.z_rndimg

    return args


def process_image(image: Image.Image) -> np.ndarray:
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model.

    Args:
        image (PIL.Image.Image): The input PIL image.

    Returns:
        np.ndarray: Processed image as a NumPy array.

    The function performs the following steps:
    1. Resizes the image to have a short side of 256 pixels while maintaining the aspect ratio.
    2. Crops the center 224x224 portion of the image.
    3. Normalizes the image values.
    4. Transposes the image to have the color channel in the first position.
    """
    short_side = 256
    crop_size = 224

    width, height = image.size
    aspect_ratio = width / height

    if height < width:
        new_height = short_side
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = short_side
        new_height = int(new_width / aspect_ratio)

    resized_img = image.resize((new_width, new_height))

    left = (new_width - crop_size) // 2
    top = (new_height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    cropped_img = resized_img.crop((left, top, right, bottom))

    np_img = np.array(cropped_img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized_img = (np_img - mean) / std

    return normalized_img.transpose((2, 0, 1))
