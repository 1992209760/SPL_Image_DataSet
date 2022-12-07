import os
import copy
import time
from types import SimpleNamespace

import numpy as np
import torch

from lib_lagc.dataset.get_dataset import get_datasets
from lib_lagc.utils.helper import ModelEma
from lib_lagc.utils.lacloss import LACLoss
from utils import datasets, models
import argparse
from utils.losses import compute_batch_loss
from utils.instrumentation import train_logger
from utils.thelog import initLogger
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='AckTheUnknown-ECCV2022')
parser.add_argument('-g', '--gpu', default=1, type=int)
parser.add_argument('-d', '--dataset', default='pascal', choices=['pascal', 'coco', 'nuswide', 'cub'], type=str)
parser.add_argument('-l', '--loss', default='lagc', choices=['lagc'], type=str)
args = parser.parse_args()

# global logger
gb_logger, save_dir = initLogger(args, save_dir='results_LEM/')


def run_train_phase(model, ema_m, P, Z, logger, epoch, phase):

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

    if P['stage'] == 1:
        for i, ((inputs_w, inputs_s), targets, idx) in enumerate(Z['dataloaders'][phase]):
            batch_size = inputs_w.size(0)

            inputs = torch.cat([inputs_w, inputs_s], dim=0).cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True).float()
            with torch.cuda.amp.autocast(enabled=True):
                logits = model.f(inputs)
            logits_w, logits_s = torch.split(logits[:], batch_size)

            L_an = F.binary_cross_entropy_with_logits(logits_w, targets, reduction='mean')

            pseudo_label = torch.sigmoid(logits_w.detach()) + targets
            pseudo_label_mask = ((pseudo_label >= P['threshold']) | (pseudo_label < (1 - P['threshold']))).float()
            pseudo_label_hard = (pseudo_label >= P['threshold']).float()

            L_plc = (F.binary_cross_entropy_with_logits(logits_s, pseudo_label_hard,
                                                        reduction='none') * pseudo_label_mask).sum() / pseudo_label_mask.sum()

            loss = L_an + P['lambda_plc'] * L_plc
            Z['optimizer'].zero_grad()
            loss.backward()
            Z['optimizer'].step()
            ema_m.update(model)
    else:
        for i, ((inputs_w, inputs_s), targets, idx) in enumerate(Z['dataloaders'][phase]):
            batch_size = inputs_w.size(0)
            inputs = torch.cat([inputs_w, inputs_s], dim=0).cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True).float()
            with torch.cuda.amp.autocast(enabled=True):
                logits, features = model.f(inputs)
            logits_w, logits_s = torch.split(logits[:], batch_size)
            feats_w, feats_s = torch.split(features[:], batch_size)

            L_an = F.binary_cross_entropy_with_logits(logits_w, targets, reduction='mean')

            pseudo_label = torch.sigmoid(logits_w.detach()) + targets
            pseudo_label_mask = ((pseudo_label >= P['threshold']) | (pseudo_label < (1 - P['threshold']))).float()
            pseudo_label_hard = (pseudo_label >= P['threshold']).float()

            L_plc = (F.binary_cross_entropy_with_logits(logits_s, pseudo_label_hard,
                                                        reduction='none') * pseudo_label_mask).sum() / pseudo_label_mask.sum()

            feats_w = torch.cat(torch.unbind(feats_w, dim=0), dim=0)
            feats_s = torch.cat(torch.unbind(feats_s, dim=0), dim=0)
            feats = torch.stack([feats_w, feats_s], dim=1)

            pseudo_label_hard = (pseudo_label >= P['threshold']).float()
            sequence_code = torch.arange(start=1, end=(P['num_classes'] + 1), step=1).repeat(targets.shape[0], 1).cuda()
            labels = sequence_code * (pseudo_label_hard.bool() | targets.bool())
            labels = torch.cat(torch.unbind(labels, dim=0), dim=0)

            # filter positive samples whose labels are not 0 (now, labels are numbered from 1 to num_class)
            positive_pos = (labels != 0).nonzero().squeeze()
            labels = labels[positive_pos]
            feats = feats[positive_pos]

            # *************************compute Lc and update MQ*************************
            P['queue_feats'].detach_()
            P['queue_labels'].detach_()

            ptr_increase = feats.shape[0]
            P['queue_ptr'] = ptr_increase if P['queue_ptr'] + ptr_increase >= 512 else P['queue_ptr'] + ptr_increase
            P['queue_feats'][P['queue_ptr'] - ptr_increase: P['queue_ptr']] = feats
            P['queue_labels'][P['queue_ptr'] - ptr_increase: P['queue_ptr']] = labels

            L_lac = P['criterion'](P['queue_feats'], P['queue_labels'])

            loss = L_an + P['lambda_plc'] * L_plc + P['lambda_lac'] * L_lac

            Z['optimizer'].zero_grad()
            loss.backward()
            Z['optimizer'].step()
            ema_m.update(model)

    # for batch in Z['dataloaders'][phase]:
    #     # move data to GPU:
    #     batch['image'] = batch['image'].to(Z['device'], non_blocking=True)
    #     batch['labels_np'] = batch['label_vec_obs'].clone().numpy()  # copy of labels for use in metrics
    #     batch['label_vec_obs'] = batch['label_vec_obs'].to(Z['device'], non_blocking=True)
    #     # forward pass:
    #     Z['optimizer'].zero_grad()
    #     with torch.set_grad_enabled(True):
    #         batch['logits'] = model.f(batch['image'])
    #         batch['preds'] = torch.sigmoid(batch['logits'])
    #         if batch['preds'].dim() == 1:
    #             batch['preds'] = torch.unsqueeze(batch['preds'], 0)
    #         batch['preds_np'] = batch['preds'].clone().detach().cpu().numpy()  # copy of preds for use in metrics
    #         batch = compute_batch_loss(batch, P, Z)
    #     # backward pass:
    #     batch['loss_tensor'].backward()
    #     Z['optimizer'].step()
    #     # save current batch data:
    #     logger.update_phase_data(batch)


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
    for i, (inputs, targets, idx) in enumerate(Z['dataloaders'][phase]):
        batch = {}
        batch['idx'] = idx
        batch['image'] = inputs.to(Z['device'], non_blocking=True)
        batch['labels_np'] = targets.clone().numpy()  # copy of labels for use in metrics
        batch['label_vec_obs'] = targets.to(Z['device'], non_blocking=True)
        batch['label_vec_true'] = targets.to(Z['device'], non_blocking=True)
        with torch.set_grad_enabled(False):
            if P['stage'] == 1:
                batch['logits'] = model.f(batch['image'])
            else:
                batch['logits'] = model.f(batch['image'])[0]
            batch['preds'] = torch.sigmoid(batch['logits'])
            if batch['preds'].dim() == 1:
                batch['preds'] = torch.unsqueeze(batch['preds'], 0)
            batch['preds_np'] = batch['preds'].clone().detach().cpu().numpy()  # copy of preds for use in metrics
            batch['loss_np'] = -1
            batch['reg_loss_np'] = -1
        # save current batch data:
        logger.update_phase_data(batch)


def train(model, ema_m, P, Z):

    '''
    Train the model.

    Parameters
    P: Dictionary of parameters, which completely specify the training procedure.
    Z: Dictionary of temporary objects used during training.
    '''

    best_weights = copy.deepcopy(model.f.state_dict())
    logger = train_logger(P) # initialize logger
    if_early_stop = False

    for epoch_idx in range(0, P['num_epochs']):
        gb_logger.info('start epoch [{}/{}] ...'.format(epoch_idx + 1, P['num_epochs']))
        P['epoch'] = epoch_idx + 1
        for phase in ['train', 'val', 'test']:
            # reset phase metrics:
            logger.reset_phase_data()

            # run one phase:
            t_init = time.time()
            if phase == 'train':
                run_train_phase(model, ema_m, P, Z, logger, P['epoch'], phase)
            else:
                run_eval_phase(model, P, Z, logger, P['epoch'], phase)

            # save end-of-phase metrics:
            if phase in ['val', 'test']:
                logger.compute_phase_metrics(phase, P['epoch'])
                # print epoch status:
                logger.report(t_init, time.time(), phase, P['epoch'], gb_logger)

            # update best epoch, if applicable:
            new_best = logger.update_best_results(phase, P['epoch'], P['val_set_variant'])
            if new_best:
                gb_logger.info('*** new best weights ***')
                best_weights = copy.deepcopy(model.f.state_dict())
            elif (not new_best) and (phase == 'val'):
                gb_logger.info('*** early stop ***')
                if_early_stop = True
                break

        if if_early_stop:
            break

    gb_logger.info('')
    gb_logger.info('*** TRAINING COMPLETE ***')
    gb_logger.info('Best epoch: {}'.format(logger.best_epoch))
    gb_logger.info('Best epoch validation score: {:.2f}'.format(logger.get_stop_metric('val', logger.best_epoch, P['val_set_variant'])))
    gb_logger.info('Best epoch test score:       {:.2f}'.format(logger.get_stop_metric('test', logger.best_epoch, 'clean')))

    return P, model, ema_m, logger, best_weights


def initialize_training_run(P, Z=None):

    '''
    Set up for model training.

    Parameters
    P: Dictionary of parameters, which completely specify the training procedure.
    feature_extractor: Feature extractor model to start from.
    linear_classifier: Linear classifier model to start from.
    estimated_labels: NumPy array containing estimated training set labels to start from (for ROLE).
    '''

    if P['stage'] == 1:
        np.random.seed(P['seed'])

        Z = {}

        # accelerator:
        Z['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # data:
        # Z['datasets'] = datasets.get_data(P)
        args = SimpleNamespace()
        args.img_size = 448
        args.dataset_name = P['dataset']
        args.dataset_dir = './data/' + args.dataset_name
        train_dataset, val_dataset, test_dataset = get_datasets(args)
        Z['datasets'] = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}

        # observed label matrix:
        label_matrix = Z['datasets']['train'].Y
        num_examples = int(np.shape(label_matrix)[0])
        mtx = np.array(label_matrix).astype(np.int8)
        total_pos = np.sum(mtx == 1)
        total_neg = np.sum(mtx == 0)
        gb_logger.info('training samples: {} total'.format(num_examples))
        gb_logger.info('true positives: {} total, {:.2f} per example on average.'.format(total_pos, total_pos / num_examples))
        gb_logger.info('true negatives: {} total, {:.2f} per example on average.'.format(total_neg, total_neg / num_examples))
        observed_label_matrix = Z['datasets']['train'].Y
        num_examples = int(np.shape(observed_label_matrix)[0])
        obs_mtx = np.array(observed_label_matrix).astype(np.int8)
        obs_total_pos = np.sum(obs_mtx == 1)
        obs_total_neg = np.sum(obs_mtx == 0)
        gb_logger.info('observed positives: {} total, {:.2f} per example on average.'.format(obs_total_pos, obs_total_pos / num_examples))
        gb_logger.info('observed negatives: {} total, {:.2f} per example on average.'.format(obs_total_neg, obs_total_neg / num_examples))

        # save dataset-specific parameters:
        P['num_classes'] = Z['datasets']['train'].num_classes

        # dataloaders:
        Z['dataloaders'] = {}
        for phase in ['train', 'val', 'test']:
            Z['dataloaders'][phase] = torch.utils.data.DataLoader(
                Z['datasets'][phase],
                batch_size = P['bsize'],
                shuffle = phase == 'train',
                sampler = None,
                num_workers = P['num_workers'],
                drop_last = False  # FIXME
            )

        # pseudo-labeling data:
        P['unlabel_num'] = []
        for i in range(observed_label_matrix.shape[1]):
            P['unlabel_num'].append(np.sum(observed_label_matrix[:, i] == 0))

        # model:
        model = models.MultilabelModel_LAGC(P)
        ema_m = ModelEma(model, 0.9997)  # 0.9997

        param_dicts = [
            {"params": [p for n, p in model.f.named_parameters() if p.requires_grad]},
        ]
        lr_mult = P['bsize'] / 256
        Z['optimizer'] = getattr(torch.optim, 'AdamW')(
            param_dicts,
            lr_mult * P['lr'],
            betas=(0.9, 0.999), eps=1e-08, weight_decay=P['wd']
        )
    else:
        model = models.MultilabelModel_LAGC(P, is_proj=True)
        ema_m = ModelEma(model, 0.9997)  # 0.9997
        model.to(Z['device'])
        ema_m.to(Z['device'])
        gb_logger.info('=> loading checkpoint')
        model.f.load_state_dict(P['state']['state_dict'], strict=False)
        gb_logger.info("=> loaded checkpoint '{}' (epoch {})"
                       .format(True, P['state']['epoch']))

        param_dicts = [
            {"params": [p for n, p in model.f.named_parameters() if p.requires_grad]},
        ]
        lr_mult = P['bsize'] / 256
        Z['optimizer'] = getattr(torch.optim, 'AdamW')(
            param_dicts,
            lr_mult * P['lr'],
            betas=(0.9, 0.999), eps=1e-08, weight_decay=P['wd']
        )

        # memory queue
        P['queue_feats'] = torch.zeros(512, 2, 128).cuda()
        P['queue_labels'] = torch.zeros(512).cuda()
        P['queue_ptr'] = 0

        P['criterion'] = LACLoss(temperature=P['temperature']).cuda()

    return P, Z, model, ema_m


def execute_training_run(P, Z=None):

    '''
    Initialize, run the training process, and save the results.

    Parameters
    P: Dictionary of parameters, which completely specify the training procedure.
    feature_extractor: Feature extractor model to start from.
    linear_classifier: Linear classifier model to start from.
    estimated_labels: NumPy array containing estimated training set labels to start from (for ROLE).
    '''
    P, Z, model, ema_m = initialize_training_run(P, Z)
    model.to(Z['device'])
    ema_m.to(Z['device'])

    P, model, ema_m, logger, best_weights_f = train(model, ema_m, P, Z)

    P['state'] = {
        'epoch': logger.best_epoch,
        'state_dict': best_weights_f,
        'best_mAP': logger.get_stop_metric('test', logger.best_epoch, 'clean'),
        'optimizer': Z['optimizer'].state_dict(),
    }

    final_logs = logger.get_logs()
    # model.f.load_state_dict(best_weights_f)

    # return model.f.feature_extractor, model.f.linear_classifier, final_logs
    return P, Z, final_logs


if __name__ == '__main__':

    P = {}

    # System parameters:
    args.pytorch_seed = int(time.time()) % (2 ** 16)
    P['pytorch_seed'] = args.pytorch_seed
    torch.manual_seed(P['pytorch_seed'])
    torch.cuda.manual_seed(P['pytorch_seed'])

    # Top-level parameters:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    P['GPU'] = str(args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = P['GPU']
    P['dataset'] = args.dataset
    P['loss'] = args.loss
    P['val_set_variant'] = 'clean'  # clean, observed

    # Optimization parameters:
    P['lambda_plc'] = 1
    P['lambda_lac'] = 1
    P['threshold'] = 0.6
    P['temperature'] = 0.5
    if P['dataset'] == 'pascal':
        P['bsize'] = 8
        P['lr'] = 1e-4
        P['wd'] = 1e-2

    elif P['dataset'] == 'cub':
        P['bsize'] = 8
        P['lr'] = 1e-4
        P['wd'] = 1e-2

    elif P['dataset'] == 'nuswide':
        P['bsize'] = 8
        P['lr'] = 1e-4
        P['wd'] = 1e-2

    elif P['dataset'] == 'coco':
        P['bsize'] = 8
        P['lr'] = 1e-4
        P['wd'] = 1e-2

    # Additional parameters:
    P['seed'] = 1200  # overall numpy seed
    P['use_pretrained'] = True  # True, False
    P['num_workers'] = 8
    P['stop_metric'] = 'map'  # metric used to select the best epoch

    # Dataset parameters:
    P['split_seed'] = 1200  # seed for train/val splitting
    P['val_frac'] = 0.2  # fraction of train set to split off for val
    P['ss_seed'] = 999  # seed for subsampling
    P['ss_frac_train'] = 1.0  # fraction of training set to subsample
    P['ss_frac_val'] = 1.0  # fraction of val set to subsample

    # Dependent parameters:
    if P['loss'] == 'bce':
        P['train_set_variant'] = 'clean'
    else:
        P['train_set_variant'] = 'observed'

    # training parameters:
    P['num_epochs'] = 1
    P['freeze_feature_extractor'] = False
    P['use_feats'] = False
    P['arch'] = 'resnet50'
    P['feature_extractor_arch'] = 'resnet50'
    P['feat_dim'] = 2048
    P['z_dim'] = 128

    # run training process:
    gb_logger.info('[{} + {}] start exp ...'.format(P['dataset'], P['loss']))
    P['stage'] = 1
    (P, Z, logs) = execute_training_run(P)
    P['stage'] = 2
    P['threshold'] = 0.9
    execute_training_run(P, Z)
