# Image Classifier Project

This project implements a deep learning image classifier using PyTorch.

## Setup

1. Create and activate a conda environment:

```textmate
conda create -n aipnd python=3.12 --solver=classic
conda activate aipnd
```

2. Install dependencies:

```textmate
conda install pytorch torchvision torchaudio -c pytorch --solver=classic
conda install numpy pillow matplotlib tqdm jupyter
```

## Usage

### Jupyter Notebook

To run the Jupyter notebook version:

1. Copy the cell contents from Part 1 deliverable into `Image Classifier Project.ipynb` in the `aip-project` workspace.
2. Run the notebook cells in order.

### Command Line Interface

The project includes two main scripts: `train.py` for training the model and `predict.py` for making predictions.

#### Training

Run `train.py` in a terminal session in the `aip-project` workspace. GPU mode is recommended for training.

```textmate
python train.py [OPTIONS] [DATA_DIR]
```

Options:
- `--save_dir SAVE_DIR`: Path to checkpoint directory
- `--arch {densenet121,densenet161,resnet18,vgg16}`: Model architecture (default: densenet121)
- `-lr`, `--learning_rate LEARNING_RATE`: Learning rate (default: 0.001)
- `-dout`, `--dropout DROPOUT`: Dropout rate (default: 0.5)
- `-hu`, `--hidden_units HIDDEN_UNITS`: Hidden units (e.g., '500' or '1000, 500')
- `-e`, `--epochs EPOCHS`: Number of epochs to run (default: 3)
- `--gpu`: Use GPU for training

Example:

```textmate
python train.py  --arch densenet161 -hu '1000, 500' -e 10 -lr 0.002 -dout 0.3 --gpu
```

#### Prediction

Run `predict.py` in a terminal session. It can run in either CPU or GPU mode.

```textmate
python predict.py [OPTIONS] [CHECKPOINT]
```

Options:
- `-img`, `--img_pth IMG_PTH`: Path to an image file
- `-cat`, `--category_names CATEGORY_NAMES`: Path to JSON file for category names
- `-k`, `--top_k TOP_K`: Number of top classes to print
- `--gpu`: Use GPU for prediction

Example:

```textmate
python predict.py chksav/20240918_111814_densenet161.pth --img_pth flowers/test/91/image_08061.jpg --category_names cat_to_name.json --top_k 4 --gpu
```


## Notes

- `predict.py` will verify a valid checkpoint is present before starting prediction.
- While it's possible to run `train.py` in CPU mode, GPU mode is more practical for training.
- Refer to the help messages (`python train.py -h` or `python predict.py -h`) for more detailed information on available options.






