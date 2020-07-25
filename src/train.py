import argparse
import gc
import json
import os
import random
import sys

import neptune
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from tqdm import tqdm

sys.path.insert(0, "src")
sys.path.insert(0, "timm_effdet")
sys.path.insert(0, "weightedboxesfusion")

from src.data import DatasetRetriever
from src.utils import calculate_image_precision

from timm_effdet.effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from timm_effdet.effdet.efficientdet import HeadNet
from weightedboxesfusion.ensemble_boxes import *

mixed_precision = False
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed
print(f'Apex is {mixed_precision}')


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)

            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def seed_everything(seed: int):
    """ Seeds and fixes every possible random state """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TrainProcess(object):
    def __init__(self):
        # model
        self.model = None
        self.optimized_rounder = None
        self.optimizer = None
        self.lr_scheduler = None
        self.criterion = None

        # all data
        self.train_loader = None
        self.val_loader = None
        self.train_database = None  # main train data WITH augmentations
        self.val_database = None  # main train data WITHOUT augmentations (for CV)
        self.test_database = None  # delayed test data
        self.init_settings()  # init data and model variables

    def load_data(self, marking, df_folds, fold_number):
        self.train_database = DatasetRetriever(
            image_ids=df_folds[df_folds['fold'] != fold_number].index.values,
            marking=marking,
            mode='train',
            test=False,
        )

        self.val_database = DatasetRetriever(
            image_ids=df_folds[df_folds['fold'] == fold_number].index.values,
            marking=marking,
            mode='val',
            test=True,
        )

    def init_settings(self):
        """ initialization of data and model variables """
        self.criterion = FocalLoss(alpha=hyp['alpha'], gamma=hyp['gamma'], logits=True).to(device)

        # load nn model
        config = get_efficientdet_config('tf_efficientdet_d5')
        net = EfficientDet(config, pretrained_backbone=False)
        checkpoint = torch.load('input/efficientdet/efficientdet_d5-ef44aea8.pth')
        net.load_state_dict(checkpoint)
        config.num_classes = 1
        config.image_size = 512
        net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
        self.model = DetBenchTrain(net, config)

        if args.weights:  # load prev weight and continue train process
            checkpoint = torch.load(os.path.join(wdir, args.weights))
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError:
                state_dict = torch.load(os.path.join(wdir, args.weights))
                # create new OrderedDict that does not contain `module.`
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict['model_state_dict'].items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                # load params
                self.model.load_state_dict(new_state_dict)

    def make_predictions(self, images, score_threshold=0.22):
        images = torch.stack(images).cuda().float()
        predictions = []
        with torch.no_grad():
            det = self.model(images, torch.tensor([1] * images.shape[0]).float().cuda())
            for i in range(images.shape[0]):
                boxes = det[i].detach().cpu().numpy()[:, :4]
                scores = det[i].detach().cpu().numpy()[:, 4]
                indexes = np.where(scores > score_threshold)[0]
                boxes = boxes[indexes]
                boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
                predictions.append({
                    'boxes': boxes[indexes],
                    'scores': scores[indexes],
                })
        return [predictions]

    def run_wbf(self, predictions, image_index, image_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None):
        boxes = [(prediction[image_index]['boxes'] / (image_size - 1)).tolist() for prediction in predictions]
        scores = [prediction[image_index]['scores'].tolist() for prediction in predictions]
        labels = [np.ones(prediction[image_index]['scores'].shape[0]).tolist() for prediction in predictions]
        boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr,
                                                      skip_box_thr=skip_box_thr)
        boxes = boxes * (image_size - 1)
        return boxes, scores, labels

    def val_step(self, epoch):
        """ Validation step """
        # Service variables
        validation_image_precisions = []
        iou_thresholds = [x for x in np.arange(0.4, 0.76, 0.05)]

        self.model.eval()
        pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader), ascii=True, desc='validation')
        mloss = torch.zeros(1).to(device)  # mean losses
        print(('\n' + '%10s' * 3) % ('Epoch', 'gpu_mem', 'total'))
        for step, (images, targets, image_ids) in pbar:
            images = torch.stack(images)
            images = images.to(device).float()
            batch_size = images.shape[0]
            boxes = [target['boxes'].to(device).float() for target in targets]
            labels = [target['labels'].to(device).float() for target in targets]

            with torch.set_grad_enabled(False):
                loss, _, _ = self.model(images, boxes, labels)

                # Scale loss by nominal batch_size
                loss /= hyp['accumulate']

                predictions = self.make_predictions(images)
                for i, image in enumerate(images):
                    boxes, scores, labels = self.run_wbf(predictions, image_index=i)
                    boxes = boxes.astype(np.int32).clip(min=0, max=1023)

                    preds = boxes
                    preds_sorted_idx = np.argsort(scores)[::-1]
                    preds_sorted = preds[preds_sorted_idx]
                    gt_boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
                    image_precision = calculate_image_precision(preds_sorted,
                                                                gt_boxes,
                                                                thresholds=iou_thresholds,
                                                                form='coco')

                    validation_image_precisions.append(image_precision)

            # Print batch results
            mloss = (mloss * step + loss.item() * hyp['accumulate']) / (step + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 1) % ('%g/%g' % (epoch, hyp['epochs'] - 1), mem, mloss)
            pbar.set_description(s)

        val_iou = np.mean(validation_image_precisions)
        return mloss, val_iou

    def train_step(self, epoch):
        """ Training step """

        # reset gradients
        self.model.train()
        self.optimizer.zero_grad()
        self.model.zero_grad()

        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), ascii=True, desc='train')
        mloss = torch.zeros(1).to(device)  # mean losses
        print(('\n' + '%10s' * 3) % ('Epoch', 'gpu_mem', 'total'))
        for step, (images, targets, image_ids) in pbar:
            images = torch.stack(images)
            images = images.to(device).float()
            batch_size = images.shape[0]
            boxes = [target['boxes'].to(device).float() for target in targets]
            labels = [target['labels'].to(device).float() for target in targets]

            with torch.set_grad_enabled(True):
                loss, _, _ = self.model(images, boxes, labels)

                # Scale loss by nominal batch_size
                loss /= hyp['accumulate']

                # Compute gradient
                if mixed_precision:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # Accumulate gradient optimization
                if (step + 1) % hyp['accumulate'] or hyp['accumulate'] == 1:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.model.zero_grad()

            mloss = (mloss * step + loss.item() * hyp['accumulate']) / (step + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 1) % ('%g/%g' % (epoch, hyp['epochs'] - 1), mem, mloss)
            pbar.set_description(s)

        gc.collect()
        return mloss

    def run(self):
        """ Main training loop """
        scheduler_step = hyp['epochs'] // hyp['snapshots']

        # create CV folds for split
        marking = pd.read_csv('input/global-wheat-detection/train.csv')

        bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
        for i, column in enumerate(['x', 'y', 'w', 'h']):
            marking[column] = bboxs[:, i]
        marking.drop(columns=['bbox'], inplace=True)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        df_folds = marking[['image_id']].copy()
        df_folds.loc[:, 'bbox_count'] = 1
        df_folds = df_folds.groupby('image_id').count()
        df_folds.loc[:, 'source'] = marking[['image_id', 'source']].groupby('image_id').min()['source']
        df_folds.loc[:, 'stratify_group'] = np.char.add(
            df_folds['source'].values.astype(str),
            df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
        )
        df_folds.loc[:, 'fold'] = 0

        for fold_number, (train_index, val_index) in enumerate(
                skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
            df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

        df_folds.to_csv(f'output/df_split_{nep_id}.csv', index=False)

        if args.data_id:
            df_folds = pd.read_csv(f'output/df_split_{args.data_id}.csv')

        for fold in range(hyp['num_folds']):
            print(f'************************'
                  f'**** [FOLD: {fold}] ****'
                  f'************************')
            # create train dataloader with augmentations
            self.load_data(marking, df_folds, fold)

            def collate_fn(batch):
                return tuple(zip(*batch))

            self.train_loader = torch.utils.data.DataLoader(
                self.train_database,
                batch_size=hyp['batch_size'],
                sampler=RandomSampler(self.train_database),
                pin_memory=False,
                drop_last=True,
                num_workers=hyp['num_workers'],
                collate_fn=collate_fn,
            )

            self.val_loader = torch.utils.data.DataLoader(
                self.val_database,
                batch_size=hyp['batch_size'],
                num_workers=hyp['num_workers'],
                shuffle=False,
                sampler=SequentialSampler(self.val_database),
                pin_memory=False,
                collate_fn=collate_fn,
            )

            num_snapshot = 0  # current shot
            best_acc = 0  # current best accuracy metric

            # init optimizer for current fold
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=hyp['max_lr'],
                                             momentum=hyp['momentum'],
                                             weight_decay=hyp['weight_decay'])

            # self.optimizer = torch.optim.Adam([{'params': self.model.model.parameters(),},
            #                                    {'params': self.model.head.parameters(),
            #                                     'lr': 6e-4}],
            #                                   lr=hyp['max_lr'],
            #                                   weight_decay=hyp['weight_decay'])

            # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
            #                                                               T_max=hyp['epochs'] - hyp['warmup_epochs'],
            #                                                               eta_min=hyp["min_lr"],
            #                                                               last_epoch=-1)
            #
            # self.lr_scheduler = GradualWarmupScheduler(self.optimizer, multiplier=hyp['warmup_factor'],
            #                                            total_epoch=hyp['warmup_epochs'],
            #                                            after_scheduler=scheduler_cosine)

            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                           T_max=scheduler_step,
                                                                           eta_min=hyp["min_lr"],
                                                                           last_epoch=-1)

            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                self.model = nn.DataParallel(self.model)
            self.model.to(device)  # set device

            if mixed_precision:
                self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1', verbosity=0)

            # run main tran loop on fold
            for epoch in range(hyp['epochs']):
                loss_train = self.train_step(epoch)

                loss_val, acc_val = self.val_step(epoch)
                acc, qwk = acc_val

                self.lr_scheduler.step(epoch)  # for CosineAnnealingLR

                # Write epoch results
                # epoch_results = loss_val, acc_val
                # with open(results_file, 'a') as f:
                #     f.write('%10.3g' * 2 % epoch_results + '\n')  # bce_loss, jacard, mloss, dice

                if args.use_neptune_log:
                    neptune.log_metric('lr', epoch, self.optimizer.param_groups[0]["lr"])
                    neptune.log_metric('train mean_loss', epoch, loss_train)
                    neptune.log_metric('val mean_loss', epoch, loss_val)
                    neptune.log_metric('val acc', epoch, acc)
                    neptune.log_metric('val cohen score', epoch, qwk)

                # scheduler checkpoint
                fine_tune = 'scratch'
                if args.data_id:
                    fine_tune = f'tune{args.data_id}'
                wbest = wdir + f'{args.model}CL_best_{nep_id}_{fine_tune}.pth'
                if qwk >= best_acc:
                    torch.save({
                        'name': args.model,
                        'epoch': epoch,
                        'fold': fold,
                        'snapshot': num_snapshot,
                        'best_fitness': best_acc,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()
                    }, wbest)
                    best_acc = qwk

                # save current snapshot and restart params
                if (epoch + 1) == scheduler_step:
                    torch.save({
                        'name': args.model,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()
                    }, wdir + args.model + f'CL_{nep_id}' + '_last_f' + str(fold) + '_s' + str(num_snapshot) + '.pth')

                    self.optimizer = torch.optim.SGD(self.model.parameters(),
                                                     lr=2e-5,
                                                     momentum=hyp['momentum'],
                                                     weight_decay=hyp['weight_decay'])

                    # init lr scheduler state
                    self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                                   T_max=3,
                                                                                   eta_min=hyp["min_lr"],
                                                                                   last_epoch=-1)

                    num_snapshot += 1
                    # best_acc = 0

            torch.save({
                'name': args.model,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()

            }, wdir + args.model + f'CL_{nep_id}' + '_last' + '.pth')
        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', default='efficientnet-b0', type=str, help='Model version')
    parser.add_argument('--cuda', action='store_true', help='Use cuda to train model')
    parser.add_argument('--hyp_params', type=str, default='cfg/default_train.json', help='hyper params file path')
    parser.add_argument('--use_neptune_log', action='store_true', help='Use neptune.ai as a logger')
    parser.add_argument('--neptune_params', type=str, default='cfg/default_neptune.json',
                        help='neptune params file path')
    parser.add_argument('--results_name', type=str, default='results.txt', help='text logging file name in output/')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--data_id', type=str, default='', help='initial data for resume training (NEP-ID)')

    args = parser.parse_args()

    wdir = 'output/weights/'  # weights dir

    if not os.path.isdir(wdir):
        os.mkdir('output')
        os.mkdir(wdir)

    results_file = f"output/{args.results_name}"
    results = open(results_file, "w")
    results.close()

    device = torch.device('cuda' if args.cuda else 'cpu')

    with open(args.hyp_params) as json_file:
        hyp = json.load(json_file)

    PARAMS = dict(hyp, **vars(args))

    # set neptune logger
    if args.use_neptune_log:
        with open(args.neptune_params) as json_file:
            np_params = json.load(json_file)

        api_token = np_params["API_token"] if np_params["API_token"] is not 0 else "ANONYMOUS"
        full_proj_name = f'{np_params["user"]}/{np_params["proj_name"]}' if np_params["user"] is not 0 else None

        assert api_token != "ANONYMOUS" or full_proj_name != None, "set corrected neptune's user and project name"

        np_params["tags"] = np_params["tags"] + [args.model, 'CL']
        neptune.init(full_proj_name, api_token=api_token)
        experiment = neptune.create_experiment(name=np_params["experiment_name"],
                                               params=PARAMS,
                                               upload_source_files=np_params["source"],
                                               tags=np_params["tags"]
                                               )

        nep_id = experiment.id  # uniq ID token from neptune

    seed_everything(62522)

    train_model = TrainProcess()
    train_model.run()

    if args.use_neptune_log:
        neptune.stop()
