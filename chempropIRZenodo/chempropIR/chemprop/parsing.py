from argparse import ArgumentParser, Namespace
import json
import os
from tempfile import TemporaryDirectory
import pickle

import torch

from chemprop.utils import makedirs
from chemprop.features import get_available_features_generators


def add_predict_args(parser: ArgumentParser):
    """
    Adds predict arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    parser.add_argument('--gpu', type=int,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--test_path', type=str,
                        help='Path to CSV file containing testing data for which predictions will be made')
    parser.add_argument('--use_compound_names', action='store_true', default=False,
                        help='Use when test data file contains compound names in addition to SMILES strings')
    parser.add_argument('--preds_path', type=str,
                        help='Path to CSV file where predictions will be saved')
    parser.add_argument('--checkpoint_dir', type=str,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    parser.add_argument('--features_generator', type=str, nargs='*',
                        choices=get_available_features_generators(),
                        help='Method of generating additional features')
    parser.add_argument('--features_path', type=str, nargs='*',
                        help='Path to features to use in FNN (instead of features_generator)')
    parser.add_argument('--max_data_size', type=int,
                        help='Maximum number of data points to load')
    parser.add_argument('--ensemble_variance', action='store_true', default=False,
                        help='Compute ensemble-based epistemic uncertainty estimation')
    parser.add_argument('--ensemble_variance_conv', type=float, default=0.0,
                        help='Standard deviation for convolution before SID calculation, in reference to the'
                            'number of task buckets to span. Default is with no convolution. A convolution value'
                            'of 5.0 will correspond to the SIS metric.')
    parser.add_argument('--qnn', action='store_true', default=False,
                        help='Set hybrid qnn model')



def add_train_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    # General arguments
    parser.add_argument('--gpu', type=int,
                        choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--data_path', type=str,
                        help='Path to data CSV file')
    parser.add_argument('--use_compound_names', action='store_true', default=False,
                        help='Use when test data file contains compound names in addition to SMILES strings')
    parser.add_argument('--max_data_size', type=int,
                        help='Maximum number of data points to load')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Whether to skip training and only test the model')
    parser.add_argument('--features_only', action='store_true', default=False,
                        help='Use only the additional features in an FFN, no graph network')
    parser.add_argument('--features_generator', type=str, nargs='*',
                        choices=get_available_features_generators(),
                        help='Method of generating additional features')
    parser.add_argument('--features_path', type=str, nargs='*',
                        help='Path to features to use in FNN (instead of features_generator)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--save_smiles_splits', action='store_true', default=False,
                        help='Save smiles for each train/val/test splits for prediction convenience later')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory from which to load model checkpoints'
                             '(walks directory and ensembles all models that are found)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--dataset_type', type=str,
                        choices=['classification', 'regression', 'multiclass','spectra'],
                        help='Type of dataset, e.g. classification or regression.'
                             'This determines the loss function used during training.')
    parser.add_argument('--multiclass_num_classes', type=int, default=3,
                        help='Number of classes when running multiclass classification')
    parser.add_argument('--separate_val_path', type=str,
                        help='Path to separate val set, optional')
    parser.add_argument('--separate_val_features_path', type=str, nargs='*',
                        help='Path to file with features for separate val set')
    parser.add_argument('--separate_test_path', type=str,
                        help='Path to separate test set, optional')
    parser.add_argument('--separate_test_features_path', type=str, nargs='*',
                        help='Path to file with features for separate test set')
    parser.add_argument('--split_type', type=str, default='random',
                        choices=['random', 'scaffold_balanced', 'predetermined', 'crossval', 'index_predetermined', 'random_with_repeated_smiles'],
                        help='Method of splitting the data into train/val/test')
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help='Split proportions for train/validation/test sets')
    parser.add_argument('--num_folds', type=int, default=1,
                        help='Number of folds when performing cross validation')
    parser.add_argument('--folds_file', type=str, default=None,
                        help='Optional file of fold labels')
    parser.add_argument('--val_fold_index', type=int, default=None,
                        help='Which fold to use as val for leave-one-out cross val')
    parser.add_argument('--test_fold_index', type=int, default=None,
                        help='Which fold to use as test for leave-one-out cross val')
    parser.add_argument('--crossval_index_dir', type=str,
                        help='Directory in which to find cross validation index files')
    parser.add_argument('--crossval_index_file', type=str,
                        help='Indices of files to use as train/val/test'
                             'Overrides --num_folds and --seed.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed to use when splitting data into train/val/test sets.'
                             'When `num_folds` > 1, the first fold uses this seed and all'
                             'subsequent folds add 1 to the seed.')
    parser.add_argument('--torch_seed', type=int, default=None,
                        help='Random seed to use for PyTorch.')
    parser.add_argument('--metric', type=str, default=None,
                        choices=['auc', 'prc-auc', 'rmse', 'mae', 'mse', 'r2', 'accuracy', 'cross_entropy', 'sid', 'jsd', 'wasserstein','stmse','srmse', 'smse'],
                        help='Metric to use during evaluation.'
                             'Note: Does NOT affect loss function used during training'
                             '(loss is determined by the `dataset_type` argument).'
                             'Note: Defaults to "auc" for classification and "rmse" for regression.')
    parser.add_argument('--normalization_start', type=int, default=None,
                        help='Option for spectral outputs. Normalizes the outputs within the specified range'
                        'to sum to one. To normalize over all targets, specify 0 as the start and no end.'
                        'Indicate the first position (0 indexed) in the targets to include in the normalization',)
    parser.add_argument('--normalization_end', type=int, default=None,
                        help='Option for spectral outputs. Normalizes the outputs within the specified range'
                        'to sum to one. To normalize over all targets, specify 0 as the start and no end.'
                        'Indicate the last position (0 indexed,inclusive) in the targets to include in the normalization',)
    parser.add_argument('--output_activation', type=str, default=None,
                        choices=['None','exp','ReLu'],
                        help='Applies an activation function to the output data after the FFN layers but before optional normalization or scaling.'
                            'Exponential activation is useful for positive-value only outputs.')
    parser.add_argument('--sm_thresh', type=float, default=1.e-8,
                        help='Threshold for positive-value filter in spectral normalization and loss.')
    parser.add_argument('--sm_eps', type=float, default=1.e-8,
                        help='Small scalar for for numerical stability of srmse.')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Skip non-essential print statements')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    parser.add_argument('--show_individual_scores', action='store_true', default=False,
                        help='Show all scores for individual targets, not just average, at the end')
    parser.add_argument('--no_cache', action='store_true', default=False,
                        help='Turn off caching mol2graph computation')
    parser.add_argument('--config_path', type=str,
                        help='Path to a .json file containing arguments. Any arguments present in the config'
                             'file will override arguments specified via the command line or by the defaults.')
    parser.add_argument('--spectral_mask_path', type=str,
                        help='Path to a csv file containing a spectral mask. Spectral mask corresponds to features in rows and targets in columns.'
                            'Include labels for both in first row/column. Fill in rest of the matrix with boolean indicators of what parts of the'
                            'spectra to keep (1 or True) and what parts to replace with nan (0 or False).')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                        help='Maximum number of epochs to run')
    parser.add_argument('--convergence_margin', type=int, default=1e8,
                        help='Considered converged if new optimum not found after this many epochs')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')
    parser.add_argument('--init_lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='Maximum learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4,
                        help='Final learning rate')
    parser.add_argument('--no_features_scaling', action='store_true', default=False,
                        help='Turn off scaling of features')
    parser.add_argument('--no_target_scaling',action='store_true', default=False,
                        help='Turn off scaling of regression targets')

    # Model arguments
    parser.add_argument('--qnn', action='store_true', default=False,
                        help='Set hybrid qnn model')
    parser.add_argument('--qnn_layer', type=int, default=2,
                        help='Numbers of qnn layers')
    parser.add_argument('--ensemble_size', type=int, default=1,
                        help='Number of models in ensemble')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Dimensionality of hidden layers in MPN')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='Whether to add bias to linear layers')
    parser.add_argument('--no_bias_ffn', action='store_true', default=False,
                        help='Whether to add bias to linear layers of the ffn')
    parser.add_argument('--depth', type=int, default=3,
                        help='Number of message passing steps')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout probability')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')
    parser.add_argument('--undirected', action='store_true', default=False,
                        help='Undirected edges (always sum the two relevant bond vectors)')
    parser.add_argument('--ffn_hidden_size', type=int, default=None,
                        help='Hidden dim for higher-capacity FFN (defaults to hidden_size)')
    parser.add_argument('--ffn_num_layers', type=int, default=2,
                        help='Number of layers in FFN after MPN encoding')
    parser.add_argument('--atom_messages', action='store_true', default=False,
                        help='Use messages on atoms instead of messages on bonds')
    parser.add_argument('--spectral_loss_function', type=str, default='sid',
                        choices=['sid','wasserstein','jsd','stmse','srmse','smse'],
                        help='Option to use alternatives to sid as the loss function for dataset_type spectra.')
    parser.add_argument('--frzn_mpn_checkpoint', type=str, default=None,
                        help='Path to checkpoint for model with mpn weights')

def update_checkpoint_args(args: Namespace):
    """
    Walks the checkpoint directory to find all checkpoints, updating args.checkpoint_paths and args.ensemble_size.

    :param args: Arguments.
    """
    if hasattr(args, 'checkpoint_paths') and args.checkpoint_paths is not None:
        return

    if args.checkpoint_dir is not None and args.checkpoint_path is not None:
        raise ValueError('Only one of checkpoint_dir and checkpoint_path can be specified.')

    if args.checkpoint_dir is None:
        args.checkpoint_paths = [args.checkpoint_path] if args.checkpoint_path is not None else None
        return

    args.checkpoint_paths = []

    for root, _, files in os.walk(args.checkpoint_dir):
        for fname in files:
            if fname.endswith('.pt'):
                args.checkpoint_paths.append(os.path.join(root, fname))

    args.ensemble_size = len(args.checkpoint_paths)

    if args.ensemble_size == 0:
        raise ValueError(f'Failed to find any model checkpoints in directory "{args.checkpoint_dir}"')


def modify_predict_args(args: Namespace):
    """
    Modifies and validates predicting args in place.

    :param args: Arguments.
    """
    assert args.test_path
    assert args.preds_path
    assert args.checkpoint_dir is not None or args.checkpoint_path is not None or args.checkpoint_paths is not None

    update_checkpoint_args(args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda
    if args.cuda and args.gpu is not None:
        args.device = f'cuda:{args.gpu}'
    else: args.device = 'cpu'

    # Create directory for preds path
    makedirs(args.preds_path, isfile=True)


def parse_predict_args() -> Namespace:
    parser = ArgumentParser()
    add_predict_args(parser)
    args = parser.parse_args()
    modify_predict_args(args)

    return args


def modify_train_args(args: Namespace):
    """
    Modifies and validates training arguments in place.

    :param args: Arguments.
    """
    global temp_dir  # Prevents the temporary directory from being deleted upon function return

    # Load config file
    if args.config_path is not None:
        with open(args.config_path) as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(args, key, value)

    assert args.data_path is not None
    assert args.dataset_type is not None

    if args.save_dir is not None:
        makedirs(args.save_dir)
    else:
        temp_dir = TemporaryDirectory()
        args.save_dir = temp_dir.name

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    del args.no_cuda
    if args.cuda and args.gpu is not None:
        args.device = f'cuda:{args.gpu}'
    else: args.device = 'cpu'

    args.features_scaling = not args.no_features_scaling
    del args.no_features_scaling
    args.target_scaling = not args.no_target_scaling
    del args.no_target_scaling

    args.bias_ffn = not args.no_bias_ffn
    del args.no_bias_ffn

    if args.metric is None:
        if args.dataset_type == 'classification':
            args.metric = 'auc'
        elif args.dataset_type == 'multiclass':
            args.metric = 'cross_entropy'
        elif args.dataset_type == 'spectra':
            args.metric = 'sid'
        else:
            args.metric = 'rmse'

    if not ((args.dataset_type == 'classification' and args.metric in ['auc', 'prc-auc', 'accuracy']) or
            (args.dataset_type == 'regression' and args.metric in ['rmse', 'mae', 'mse', 'r2']) or
            (args.dataset_type == 'multiclass' and args.metric in ['cross_entropy', 'accuracy']) or
            (args.dataset_type == 'spectra' and args.metric in ['rmse','mae','mse','sid','jsd','wasserstein','stmse','srmse','smse'])):
        raise ValueError(f'Metric "{args.metric}" invalid for dataset type "{args.dataset_type}".')

    args.minimize_score = args.metric in ['rmse', 'mae', 'mse', 'cross_entropy','sid', 'jsd', 'wasserstein', 'stmse', 'srmse', 'smse']

    update_checkpoint_args(args)

    if args.features_only:
        assert args.features_generator or args.features_path

    args.use_input_features = args.features_generator or args.features_path

    if args.features_generator is not None and 'rdkit_2d_normalized' in args.features_generator:
        assert not args.features_scaling

    args.num_lrs = 1

    if args.ffn_hidden_size is None:
        args.ffn_hidden_size = args.hidden_size

    assert (args.split_type == 'predetermined') == (args.folds_file is not None) == (args.test_fold_index is not None)
    assert (args.split_type == 'crossval') == (args.crossval_index_dir is not None)
    assert (args.split_type in ['crossval', 'index_predetermined']) == (args.crossval_index_file is not None)
    if args.split_type in ['crossval', 'index_predetermined']:
        with open(args.crossval_index_file, 'rb') as rf:
            args.crossval_index_sets = pickle.load(rf)
        args.num_folds = len(args.crossval_index_sets)
        args.seed = 0

    if args.test:
        args.epochs = 0

    if args.torch_seed is None:
        torch.manual_seed(args.seed)


def parse_train_args() -> Namespace:
    """
    Parses arguments for training (includes modifying/validating arguments).

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_train_args(parser)
    args = parser.parse_args()
    modify_train_args(args)

    return args


def modify_fingerprint_args(args: Namespace):
    """
    Modifies and validates fingerprint generation args in place, following modifications that would already be done for predictions

    :param args: Arguments.
    """

    assert len(args.checkpoint_paths) == 1, "Fingerprint generation only supports one model, cannot use an ensemble"
    args.mpn_output_only = True


def parse_fingerprint_args() -> Namespace:
    """
    Parses arguments for generation of fingerprints from a pretrained model. Uses predict args set.

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_predict_args(parser)
    args = parser.parse_args()
    modify_predict_args(args)
    modify_fingerprint_args(args)

    return args
