import argparse
from helper.constant import test_dir, data_dir, datadir, savedir, model_names
from helper.load import check_point


def create_common_args(parser):
    parser.add_argument('data_dir', type=str, nargs='?', default=data_dir,
                        help='path to datasets')
    parser.add_argument('--gpu', action='store_true',
                        help='use GPU for processing')


def get_predict_input_args(in_arg=None):
    parser = argparse.ArgumentParser()
    create_common_args(parser)

    parser.add_argument('checkpoint', type=str, nargs='?', default=check_point(),
                        help='path to saved checkpoint')
    parser.add_argument('-img', '--img_pth', type=str, default=f'{test_dir}/69/image_05959.jpg',
                        help='path to an image file')
    parser.add_argument('-cat', '--category_names', type=str, default='cat_to_name.json',
                        help='path to JSON file for mapping class values to category names')
    parser.add_argument('-k', '--top_k', type=int, default=1,
                        help='no. of top k classes to print')

    if in_arg is None:
        return parser.parse_args([])
    else:
        return parser.parse_args(in_arg)


# def get_train_input_args():
#     parser = argparse.ArgumentParser()
#     create_common_args(parser)
#
#     parser.add_argument('--save_dir', type=str, default=savedir,
#                         help='path to checkpoint directory')
#     parser.add_argument('--arch', default='densenet121', choices=model_names,
#                         help=f'model architecture: {" | ".join(model_names)} (default: densenet121)')
#     parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
#                         help='learning rate (default: 0.001)')
#     parser.add_argument('-dout', '--dropout', type=float, default=0.5,
#                         help='dropout rate (default: 0.5)')
#     parser.add_argument('-hu', '--hidden_units', type=str,
#                         help="hidden units, one or multiple values (comma separated) enclosed in single quotes. "
#                              "Ex1. one value: '500' Ex2. multiple values: '1000, 500'")
#     parser.add_argument('-e', '--epochs', type=int, default=3,
#                         help='total no. of epochs to run (default: 3)')
#
#     return parser.parse_args()

def get_train_input_args(in_arg=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, nargs='?', default=data_dir,
                        help='path to datasets')
    parser.add_argument('--gpu', action='store_true',
                        help='use GPU for processing')
    parser.add_argument('--save_dir', type=str, default=savedir,
                        help='path to checkpoint directory')
    parser.add_argument('--arch', default='densenet121', choices=model_names,
                        help=f'model architecture: {" | ".join(model_names)} (default: densenet121)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('-dout', '--dropout', type=float, default=0.5,
                        help='dropout rate (default: 0.5)')
    parser.add_argument('-hu', '--hidden_units', type=str,
                        help="hidden units, one or multiple values (comma separated) enclosed in single quotes. "
                             "Ex1. one value: '500' Ex2. multiple values: '1000, 500'")
    parser.add_argument('-e', '--epochs', type=int, default=3,
                        help='total no. of epochs to run (default: 3)')

    # Parse known args only
    if in_arg is None:
        return parser.parse_args([])
    else:
        return parser.parse_args(in_arg)




