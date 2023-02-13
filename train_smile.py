import argparse
import os
import copy
import sys
import time
import json
import traceback
from types import SimpleNamespace

import numpy
import numpy as np
import torch
from torch.optim import lr_scheduler

from lib_lagc.models.LEModel import build_LEModel_VIB
from utils import datasets, models
from utils.losses import compute_batch_loss, loss_smile, loss_warm
import datetime
from utils.instrumentation import train_logger, get_cosine_schedule_with_warmup
from utils.thelog import initLogger

parser = argparse.ArgumentParser(description='SMILE')
parser.add_argument('-g', '--gpu', default=0, type=int)
parser.add_argument('-d', '--dataset', default='pascal', choices=['pascal', 'coco', 'nuswide', 'cub'], type=str)
parser.add_argument('-l', '--loss', default='smile', type=str)
parser.add_argument('-a', '--alpha', default=1, type=float)
parser.add_argument('-b', '--beta', default=1e-4, type=float)
parser.add_argument('-t', '--theta', default=1e-4, type=float)
parser.add_argument('-z', '--z_dim', default=256, type=int)
parser.add_argument('-lr', default=1e-5, type=float)
parser.add_argument('-wd', default=1e-4, type=float)
parser.add_argument('-bs', default=16, type=int)
parser.add_argument('-T', default=1.0, type=float)

args = parser.parse_args()

# global logger
sys.stdout = open(os.devnull, 'w')
gb_logger, save_dir = initLogger(args, save_dir='param_smile_adam_pred_an_multilr/')


def run_train_phase(model, P, Z, logger, epoch, phase):
    '''
    Run one training phase.
    
    Parameters
    model: Model to train.
    P: Dictionary of parameters, which completely specify the training procedure.
    Z: Dictionary of temporary objects used during training.
    logger: Object used to track various metrics during training. 
    epoch: Integer index of the current epoch.
    phase: String giving the phase name
    '''

    assert phase == 'train'
    model.train()
    for batch in Z['dataloaders'][phase]:
        # move data to GPU: 
        batch['image'] = batch['image'].to(Z['device'], non_blocking=True)
        batch['labels_np'] = batch['label_vec_obs'].clone().numpy()  # copy of labels for use in metrics
        batch['label_vec_obs'] = batch['label_vec_obs'].to(Z['device'], non_blocking=True)
        # forward pass: 
        Z['optimizer'].zero_grad()
        with torch.set_grad_enabled(True):
            # batch['logits'], batch['label_vec_est'] = model(batch)
            batch['logits'], batch['distributions'], \
                batch['mus'], batch['stds'], \
                batch['alphas'], batch['betas'] = model(batch['image'])
            batch['preds'] = torch.sigmoid(batch['logits'] / P['T'])
            if batch['preds'].dim() == 1:
                batch['preds'] = torch.unsqueeze(batch['preds'], 0)
            batch['preds_np'] = batch['preds'].clone().detach().cpu().numpy()  # copy of preds for use in metrics
            if epoch > P['warmup_epoch']:
                batch = loss_smile(batch, P, Z)
                batch['loss_tensor'].backward()
            else:
                batch = loss_warm(batch, P, Z)
                batch['loss_tensor'].backward(retain_graph=True)
                batch['loss_tensor_D'].backward()
        # backward pass:
        Z['optimizer'].step()
        # save current batch data:
        logger.update_phase_data(batch)


def run_eval_phase(model, P, Z, logger, epoch, phase):
    '''
    Run one evaluation phase.
    
    Parameters
    model: Model to train.
    P: Dictionary of parameters, which completely specify the training procedure.
    Z: Dictionary of temporary objects used during training.
    logger: Object used to track various metrics during training. 
    epoch: Integer index of the current epoch.
    phase: String giving the phase name
    '''

    assert phase in ['val', 'test']
    model.eval()
    for batch in Z['dataloaders'][phase]:
        # move data to GPU: 
        batch['image'] = batch['image'].to(Z['device'], non_blocking=True)
        batch['labels_np'] = batch['label_vec_obs'].clone().numpy()  # copy of labels for use in metrics
        batch['label_vec_obs'] = batch['label_vec_obs'].to(Z['device'], non_blocking=True)
        # forward pass:
        with torch.set_grad_enabled(False):
            batch['logits'] = model(batch['image'])[0]
            batch['preds'] = torch.sigmoid(batch['logits'])
            if batch['preds'].dim() == 1:
                batch['preds'] = torch.unsqueeze(batch['preds'], 0)
            batch['preds_np'] = batch['preds'].clone().detach().cpu().numpy()  # copy of preds for use in metrics
            batch['loss_np'] = -1
            batch['reg_loss_np'] = -1
        # save current batch data:
        logger.update_phase_data(batch)


def train(model, P, Z):
    """
    Train the model.

    Parameters
    P: Dictionary of parameters, which completely specify the training procedure.
    Z: Dictionary of temporary objects used during training.
    """

    # best_weights_f = copy.deepcopy(model.feature_extractor.state_dict())
    logger = train_logger(P)  # initialize logger

    for epoch in range(P['num_epochs']):
        gb_logger.info('Epoch {}/{}'.format(epoch, P['num_epochs'] - 1))
        P['epoch'] = epoch + 1
        for phase in ['train', 'val', 'test']:
            # reset phase metrics:
            logger.reset_phase_data()

            # run one phase:
            t_init = time.time()
            if phase == 'train':
                run_train_phase(model, P, Z, logger, epoch, phase)
            else:
                run_eval_phase(model, P, Z, logger, epoch, phase)

            # save end-of-phase metrics:
            logger.compute_phase_metrics(phase, epoch)

            # print epoch status:
            logger.report(t_init, time.time(), phase, epoch, gb_logger)

            # update best epoch, if applicable:
            new_best = logger.update_best_results(phase, epoch, P['val_set_variant'])
            if new_best:
                gb_logger.info('*** new best weights ***')
                # best_weights_feature_extractor = copy.deepcopy(model.feature_extractor.state_dict())
                # best_weights_encoder_z = copy.deepcopy(model.encoder_z.state_dict())
                # best_weights_linear_classifier = copy.deepcopy(model.linear_classifier.state_dict())
        # Z['scheduler'].step()
    gb_logger.info('')
    gb_logger.info('*** TRAINING COMPLETE ***')
    gb_logger.info('Best epoch: {}'.format(logger.best_epoch))
    gb_logger.info('Best epoch validation score: {:.2f}'.format(
        logger.get_stop_metric('val', logger.best_epoch, P['val_set_variant'])))
    gb_logger.info(
        'Best epoch test score:       {:.2f}'.format(logger.get_stop_metric('test', logger.best_epoch, 'clean')))
    if 'best_weights_feature_extractor' not in locals():
        best_weights_feature_extractor = None
        best_weights_encoder_z = None
        best_weights_linear_classifier = None

    # return P, model, logger, best_weights_feature_extractor, best_weights_encoder_z, best_weights_linear_classifier
    return P, model, logger, None


def initialize_training_run(P, feature_extractor, linear_classifier):
    '''
    Set up for model training.
    
    Parameters
    P: Dictionary of parameters, which completely specify the training procedure.
    feature_extractor: Feature extractor model to start from.
    linear_classifier: Linear classifier model to start from.
    estimated_labels: NumPy array containing estimated training set labels to start from (for ROLE).
    '''

    # os.makedirs(P['save_path'], exist_ok=True)
    np.random.seed(P['seed'])

    Z = {}

    # accelerator:
    Z['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data:
    Z['datasets'] = datasets.get_data(P)

    # observed label matrix:
    label_matrix = Z['datasets']['train'].label_matrix
    num_examples = int(np.shape(label_matrix)[0])
    mtx = np.array(label_matrix).astype(np.int8)
    total_pos = np.sum(mtx == 1)
    total_neg = np.sum(mtx == 0)
    gb_logger.info('training samples: {} total'.format(num_examples))
    gb_logger.info(
        'true positives: {} total, {:.2f} per example on average.'.format(total_pos, total_pos / num_examples))
    gb_logger.info(
        'true negatives: {} total, {:.2f} per example on average.'.format(total_neg, total_neg / num_examples))
    observed_label_matrix = Z['datasets']['train'].label_matrix_obs
    num_examples = int(np.shape(observed_label_matrix)[0])
    obs_mtx = np.array(observed_label_matrix).astype(np.int8)
    obs_total_pos = np.sum(obs_mtx == 1)
    obs_total_neg = np.sum(obs_mtx == 0)
    gb_logger.info('observed positives: {} total, {:.2f} per example on average.'.format(obs_total_pos,
                                                                                         obs_total_pos / num_examples))
    gb_logger.info('observed negatives: {} total, {:.2f} per example on average.'.format(obs_total_neg,
                                                                                         obs_total_neg / num_examples))

    # save dataset-specific parameters:
    P['num_classes'] = Z['datasets']['train'].num_classes

    # dataloaders:
    Z['dataloaders'] = {}
    for phase in ['train', 'val', 'test']:
        Z['dataloaders'][phase] = torch.utils.data.DataLoader(
            Z['datasets'][phase],
            batch_size=P['bsize'],
            shuffle=phase == 'train',
            sampler=None,
            num_workers=P['num_workers'],
            drop_last=True
        )

    # model:
    model = models.MultilabelModel_smile(P, feature_extractor)

    # optimization objects:
    f_params = [param for param in list(model.feature_extractor.parameters()) if param.requires_grad]
    g_params = [param for param in list(model.decoder_D.parameters()) +
                                   list(model.linear_classifier.parameters()) +
                                   list(model.proj.parameters()) if param.requires_grad]
    # print(f_params)
    opt_params = [
        {'params': f_params, 'lr': P['lr']},
        {'params': g_params, 'lr': 10 * P['lr']}
    ]
    Z['optimizer'] = torch.optim.Adam(
        opt_params,
        lr=P['lr']
    )
    # Z['optimizer'] = torch.optim.SGD(
    #     f_params,
    #     lr=P['lr'],
    #     weight_decay=P['wd'],
    #     momentum=0.9
    # )

    return P, Z, model


def execute_training_run(P, feature_extractor, linear_classifier):
    '''
    Initialize, run the training process, and save the results.
    
    Parameters
    P: Dictionary of parameters, which completely specify the training procedure.
    feature_extractor: Feature extractor model to start from.
    linear_classifier: Linear classifier model to start from.
    estimated_labels: NumPy array containing estimated training set labels to start from (for ROLE).
    '''

    P, Z, model = initialize_training_run(P, feature_extractor, linear_classifier)
    model.to(Z['device'])

    P, model, logger, best_weights_f = train(model, P, Z)

    final_logs = logger.get_logs()

    # return model.feature_extractor, model.encoder_z, model.linear_classifier, final_logs
    return None, None, final_logs


if __name__ == '__main__':

    lookup = {
        'feat_dim': {
            'resnet50': 2048
        },
        'expected_num_pos': {
            'pascal': 1.5,
            'coco': 2.9,
            'nuswide': 1.9,
            'cub': 31.4
        },
    }

    P = {}

    # Top-level parameters:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    P['GPU'] = str(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = P['GPU']
    P['dataset'] = args.dataset  # pascal, coco, nuswide, cub
    P['loss'] = args.loss  # bce, bce_ls, iun, iu, pr, an, an_ls, wan, epr, role
    P['train_mode'] = 'end_to_end'  # linear_fixed_features, end_to_end, linear_init
    P['val_set_variant'] = 'clean'  # clean, observed

    # Optimization parameters:
    P['lr_mult'] = 10.0  # learning rate multiplier for the parameters of g
    P['stop_metric'] = 'map'  # metric used to select the best epoch

    # Loss-specific parameters:
    P['ls_coef'] = 0.1  # label smoothing coefficient

    # Additional parameters:
    P['seed'] = 1200  # overall numpy seed
    P['use_pretrained'] = True  # True, False
    P['num_workers'] = 4

    # Dataset parameters:
    P['split_seed'] = 1200  # seed for train / val splitting
    P['val_frac'] = 0.2  # fraction of train set to split off for val
    P['ss_seed'] = 999  # seed for subsampling
    P['ss_frac_train'] = 1.0  # fraction of training set to subsample
    P['ss_frac_val'] = 1.0  # fraction of val set to subsample

    # Optimization parameters:
    P['sample_times'] = 5
    P['warmup_epoch'] = 5
    # P['z_dim'] = 256
    if P['dataset'] == 'pascal':
        P['bsize'] = 8
        P['lr'] = 1e-5
        P['wd'] = 1e-2
        P['beta'] = 1e-4
        P['theta'] = 0.8
        P['z_dim'] = 512
        P['warmup_epoch'] = 2
    elif P['dataset'] == 'cub':
        P['bsize'] = 8
        P['lr'] = 1e-5
        P['wd'] = 1e-5
        P['warmup_epoch'] = 15
        P['beta'] = 1e-4
        P['theta'] = 0.8
        P['z_dim'] = 512
    elif P['dataset'] == 'nuswide':
        P['bsize'] = 16
        P['lr'] = 1e-5
        P['wd'] = 1e-4
        P['beta'] = 1e-4
        P['theta'] = 0.8
        P['z_dim'] = 512
    elif P['dataset'] == 'coco':
        P['bsize'] = 16
        P['lr'] = 1e-5
        P['wd'] = 1e-4
        P['beta'] = 1e-5
        P['theta'] = 0.8
        P['z_dim'] = 512
    # P['lr'] = args.lr
    # P['wd'] = args.wd
    P['alpha'] = args.alpha
    P['beta'] = args.beta
    P['theta'] = args.theta
    # P['z_dim'] = args.z_dim
    # P['bsize'] = args.bs
    P['T'] = args.T

    # Dependent parameters:
    if P['loss'] in ['bce', 'bce_ls']:
        P['train_set_variant'] = 'clean'
    else:
        P['train_set_variant'] = 'observed'

    P['num_epochs'] = 40
    P['freeze_feature_extractor'] = False
    P['use_feats'] = False
    P['arch'] = 'resnet50'

    P['feature_extractor_arch'] = 'resnet50'
    P['feat_dim'] = lookup['feat_dim'][P['feature_extractor_arch']]
    P['expected_num_pos'] = lookup['expected_num_pos'][P['dataset']]
    P['train_feats_file'] = './data/{}/train_features_imagenet_{}.npy'.format(P['dataset'], P['feature_extractor_arch'])
    P['val_feats_file'] = './data/{}/val_features_imagenet_{}.npy'.format(P['dataset'], P['feature_extractor_arch'])

    # (feature_extractor, encoder_z, linear_classifier, logs) = execute_training_run(P,
    #                                                                                feature_extractor=None,
    #                                                                                encoder_z_init=None,
    #                                                                                linear_classifier=None
    #                                                                                )
    try:
        (feature_extractor, linear_classifier, logs) = execute_training_run(P, feature_extractor=None, linear_classifier=None)
    except Exception as e:
        gb_logger.error("Error : " + str(e))
        gb_logger.error('traceback.format_exc():\n%s' % traceback.format_exc())
