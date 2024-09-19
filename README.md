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



# Image Classification Project Results

## Training

The model was trained using a DenseNet121 architecture for 1 epoch. Here are the key training details and results:

- **Architecture**: DenseNet121
- **Epochs**: 1
- **Initial Learning Rate**: 0.001
- **Print Frequency**: 40 batches

### Training Results
- **Train Loss**: 1.7762
- **Train Accuracy**: 57.08%
- **Validation Loss**: 0.8474
- **Validation Accuracy**: 83.50%

The model showed significant improvement from training to validation, with the validation accuracy being notably higher than the training accuracy. This suggests that the model generalizes well to unseen data, even after just one epoch.

### Training Time
- Total training time: 44.26 seconds


<img src="assets/train.png" alt="Training Output" width="2003"/>
<br>

## Testing

After training, the model was evaluated on a separate test set. Here are the test results:

- **Test Loss**: 0.8959
- **Test Accuracy**: 81.20%

### Testing Time
- Total testing time: 6.11 seconds

<img src="assets/test.png" alt="Training Output" width="2003"/>

## Observations
<br>
1. The model achieved a high validation accuracy (83.50%) after just one epoch, indicating quick learning and good generalization.
2. The test accuracy (81.20%) is close to the validation accuracy, suggesting consistent performance on unseen data.
3. The training accuracy is lower than both validation and test accuracies, which could indicate that more epochs might further improve the model's performance on the training set.
4. The quick training and testing times suggest efficient model architecture and data processing.


For a more robust evaluation, we need to train the model for more epochs and monitoring the learning curves to find the optimal number of epochs.
The example is created to showcase how the process is running in the Jupyter notebook. 


# Flower Classification Results

<br>

![Black-eyed Susan flower](/assets/image_05878.jpg)

<br>

This image shows the results of a deep learning classifier predicting the type of flower in the given image.

## Top 5 Predictions

| Rank | Class Name | Probability |
|------|------------|------------|
| 1 | Black-eyed Susan | 20.53% |
| 2 | Gazania | 9.05% |
| 3 | English Marigold | 8.92% |
| 4 | Barbeton Daisy | 8.30% |
| 5 | Californian Poppy | 8.16% |

<br>

<img src="assets/predict.png" alt="Training Output" width="2003"/>

<br>


The classifier correctly identified the flower as a Black-eyed Susan with the highest probability of 20.53%. The image shows a close-up of a yellow flower with a dark center, characteristic of the Black-eyed Susan.

A bar chart visualizes the probabilities for each predicted class, clearly showing the Black-eyed Susan as the top prediction.

## Technical Note

The classifier output includes a deprecation warning:






