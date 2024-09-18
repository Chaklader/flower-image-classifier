import os
import numpy as np
from PIL import Image


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
