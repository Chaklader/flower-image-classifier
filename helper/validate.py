import sys
from pathlib import Path


def validate_path(path, description):
    if not Path(path).exists():
        print(f"*** {description}: '{path}' not found ... exiting\n")
        sys.exit(1)


def validate_positive(value, name, allow_zero=False):
    if (allow_zero and value < 0) or (not allow_zero and value <= 0):
        print(f"*** {name} must be {'non-negative' if allow_zero else 'positive'} ... exiting\n")
        sys.exit(1)


def validate_hidden_units(hidden_units):
    if hidden_units:
        try:
            list(map(int, hidden_units.split(',')))
        except ValueError:
            print(f"Hidden units contain non-numeric value(s): [{hidden_units}] ... exiting\n")
            sys.exit(1)


def validate_train_args(args):
    validate_path(args.data_dir, "Data directory")
    validate_positive(args.learning_rate, "Learning rate")
    validate_positive(args.dropout, "Dropout", allow_zero=True)
    validate_positive(args.epochs, "Epochs")

    if args.arch != 'resnet18':
        validate_hidden_units(args.hidden_units)


def validate_predict_args(args):
    validate_path(args.checkpoint, "Checkpoint")
    validate_path(args.img_pth, "Image path")
    validate_path(args.category_names, "Category names mapper file")
    validate_positive(args.top_k, "Number of top k classes to print")
