# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from pathlib import Path
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict

import os
import re
import mmcv
import numpy as np
import torch
import time
import pickle
import inspect
from sklearn.cluster import KMeans
from mmcv.utils import print_log
from torch.utils.data import Dataset

from slurm.gcd4da.commons.kmeans_pre import train_wrapper as train_sskmeans

from ..core import (mean_average_precision, mean_class_accuracy,
                    mmit_mean_average_precision, top_k_accuracy,
                    confusion_matrix)
from .pipelines import Compose
from .custom_metrics import split_cluster_acc_v2, split_cluster_acc_v2_balanced

# from slurm.gcd4da.commons.kmeans import train

END = '\033[0m'
BOLD = '\033[1m'
GREEN = '\033[92m'
PURPLE = '\033[95m'

class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for datasets.

    All datasets to process video should subclass it.
    All subclasses should overwrite:

    - Methods:`load_annotations`, supporting to load information from an
    annotation file.
    - Methods:`prepare_train_frames`, providing train data.
    - Methods:`prepare_test_frames`, providing test data.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        data_prefix (str | None): Path to a directory where videos are held.
            Default: None.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        multi_class (bool): Determines whether the dataset is a multi-class
            dataset. Default: False.
        num_classes (int | None): Number of classes of the dataset, used in
            multi-class datasets. Default: None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Default: 1.
        modality (str): Modality of data. Support 'RGB', 'Flow', 'Audio'.
            Default: 'RGB'.
        sample_by_class (bool): Sampling by class, should be set `True` when
            performing inter-class data balancing. Only compatible with
            `multi_class == False`. Only applies for training. Default: False.
        power (float): We support sampling data with the probability
            proportional to the power of its label frequency (freq ^ power)
            when sampling data. `power == 1` indicates uniformly sampling all
            data; `power == 0` indicates uniformly sampling all classes.
            Default: 0.
        dynamic_length (bool): If the dataset length is dynamic (used by
            ClassSpecificDistributedSampler). Default: False.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_prefix=None,
                 test_mode=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 power=0,
                 dynamic_length=False):
        super().__init__()

        self.ann_file = ann_file
        self.data_prefix = osp.realpath(
            data_prefix) if data_prefix is not None and osp.isdir(
                data_prefix) else data_prefix
        self.test_mode = test_mode
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        self.sample_by_class = sample_by_class
        self.power = power
        self.dynamic_length = dynamic_length

        assert not (self.multi_class and self.sample_by_class)

        self.pipeline = Compose(pipeline)
        self.video_infos = self.load_annotations()
        if self.sample_by_class:
            self.video_infos_by_class = self.parse_by_class()

            class_prob = []
            for _, samples in self.video_infos_by_class.items():
                class_prob.append(len(samples) / len(self.video_infos))
            class_prob = [x**self.power for x in class_prob]

            summ = sum(class_prob)
            class_prob = [x / summ for x in class_prob]

            self.class_prob = dict(zip(self.video_infos_by_class, class_prob))

    @abstractmethod
    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""

    # json annotations already looks like video_infos, so for each dataset,
    # this func should be the same
    def load_json_annotations(self):
        """Load json annotation file to get video information."""
        video_infos = mmcv.load(self.ann_file)
        num_videos = len(video_infos)
        path_key = 'frame_dir' if 'frame_dir' in video_infos[0] else 'filename'
        for i in range(num_videos):
            path_value = video_infos[i][path_key]
            if self.data_prefix is not None:
                path_value = osp.join(self.data_prefix, path_value)
            video_infos[i][path_key] = path_value
            if self.multi_class:
                assert self.num_classes is not None
            else:
                assert len(video_infos[i]['label']) == 1
                video_infos[i]['label'] = video_infos[i]['label'][0]
        return video_infos

    def parse_by_class(self):
        video_infos_by_class = defaultdict(list)
        for item in self.video_infos:
            label = item['label']
            video_infos_by_class[label].append(item)
        return video_infos_by_class

    @staticmethod
    def label2array(num, label):
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr

    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 metric_options=dict(top_k_accuracy=dict(topk=(1, 5))),
                 logger=None,
                 **deprecated_kwargs):
        """Perform evaluation for common datasets.

        Args:
            results (list): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            metric_options (dict): Dict for metric options. Options are
                ``topk`` for ``top_k_accuracy``.
                Default: ``dict(top_k_accuracy=dict(topk=(1, 5)))``.
            logger (logging.Logger | None): Logger for recording.
                Default: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.
                See 'https://github.com/open-mmlab/mmaction2/pull/286'.

        Returns:
            dict: Evaluation results dict.
        """
        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        if deprecated_kwargs != {}:
            warnings.warn(
                'Option arguments for metrics has been changed to '
                "`metric_options`, See 'https://github.com/open-mmlab/mmaction2/pull/286' "  # noqa: E501
                'for more details')
            # metric_options['top_k_accuracy'] = dict(
            #     metric_options['top_k_accuracy'], **deprecated_kwargs)

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = [
            'top_k_accuracy', 'mean_class_accuracy', 'H_mean_class_accuracy', 'mean_average_precision',
            'mmit_mean_average_precision', 'recall_unknown', 'confusion_matrix', 'kmeans', 'sskmeans', 'logits',
            'gcd_v2', 'gcd_v2_cheated'
        ]

        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = OrderedDict()
        if metric_options.get('use_predefined_labels', False):
            gt_labels:list = self.predefined_labels
        else:
            gt_labels:list = [ann['label'] for ann in self.video_infos]

        results = np.array(results)

        for metric in metrics:
            msg = f'Evaluating {metric} ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy',
                                                 {}).setdefault(
                                                     'topk', (1, 5))
                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')
                if isinstance(topk, int):
                    topk = (topk, )

                top_k_acc = top_k_accuracy(results, gt_labels, topk)
                log_msg = []
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}_acc'] = acc
                    log_msg.append(f'\ntop{k}_acc\t{acc:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric == 'mean_class_accuracy':
                mean_acc = mean_class_accuracy(results, gt_labels)
                eval_results['mean_class_accuracy'] = mean_acc
                log_msg = f'\nmean_acc\t{mean_acc:.4f}'
                print_log(log_msg, logger=logger)
                continue

            if metric == 'H_mean_class_accuracy':  # only valid for open-set
                pred = np.argmax(results, axis=1)
                cf_mat = confusion_matrix(pred, gt_labels)
                cls_cnt = cf_mat.sum(axis=1)
                cls_hit = np.diag(cf_mat)
                cls_acc = np.array([hit / cnt if cnt else 0.0 for cnt, hit in zip(cls_cnt, cls_hit)])
                os_star, unk = cls_acc[:-1].mean(), cls_acc[-1]
                H_mean_acc = 2 * os_star * unk / (os_star + unk)
                eval_results['H_mean_class_accuracy'] = H_mean_acc
                eval_results['os*'] = os_star
                eval_results['recall_unknown'] = unk
                log_msg = f'\nH\t\t{H_mean_acc:.4f}'
                log_msg += f'\nOS*\t\t{os_star:.4f}'
                log_msg += f'\nUNK\t\t{unk:.4f}'
                print_log(log_msg, logger=logger)
                continue

            if metric in [
                    'mean_average_precision', 'mmit_mean_average_precision'
            ]:
                gt_labels_arrays = [
                    self.label2array(self.num_classes, label)
                    for label in gt_labels
                ]
                if metric == 'mean_average_precision':
                    mAP = mean_average_precision(results, gt_labels_arrays)
                    eval_results['mean_average_precision'] = mAP
                    log_msg = f'\nmean_average_precision\t{mAP:.4f}'
                elif metric == 'mmit_mean_average_precision':
                    mAP = mmit_mean_average_precision(results,
                                                      gt_labels_arrays)
                    eval_results['mmit_mean_average_precision'] = mAP
                    log_msg = f'\nmmit_mean_average_precision\t{mAP:.4f}'
                print_log(log_msg, logger=logger)
                continue

            if metric == 'recall_unknown':
                pred = np.argmax(results, axis=1)
                conf = confusion_matrix(pred, gt_labels)
                recall = conf[-1,-1] / conf[-1,:].sum()
                eval_results['recall_unknown'] = recall
                log_msg = f'\nrecall_unknown\t{recall:.4f}'
                print_log(log_msg, logger=logger)
                continue

            if metric == 'confusion_matrix':
                pred = np.argmax(results, axis=1)
                conf = confusion_matrix(pred, gt_labels)
                conf_normed = conf / (conf.sum(axis=1, keepdims=True)+1e-6)
                h, w = conf.shape
                s = ''
                with np.printoptions(threshold=np.inf, linewidth=np.inf):  # thres: # elems, width: # chars
                    s += 'Confusion matrix\n'
                    s += str(conf) + '\n\n'
                with np.printoptions(threshold=np.inf, linewidth=np.inf, suppress=True):
                    s += 'Normalized Confusion matrix\n'
                    s += str((100*conf_normed).astype(int)) + '\n'
                if 'SRUN_DEBUG' in os.environ:  # if in srun(debuging) session
                    for start, end in reversed([(m.start(0), m.end(0)) for i, m in enumerate(re.finditer(r'\d+', str(s))) if i%h == i//w]):
                        s = s[:start] + '\033[1m' + s[start:end] + '\033[0m' + s[end:]  # diag vals to be bold
                log_msg = '\n' + s
                print_log(log_msg, logger=logger)
                continue

            if metric == 'kmeans':
                # make sure model.test_cfg.feature_extraction=True
                # to resolve circular import
                from mmaction.datasets.dataset_wrappers import ConcatDataset
                only_target = type(self) == ConcatDataset  # evaluate only target if test else evaluate source if valid
                gt_labels = np.array(gt_labels)
                num_old_classes = metric_options.setdefault('num_old_classes', gt_labels.max()+1)
                is_closed_set = gt_labels.max()+1 <= num_old_classes
                num_all_classes = num_old_classes if is_closed_set else metric_options.setdefault('num_all_classes', gt_labels.max()+1)
                kmeans = KMeans(init='k-means++', n_clusters=num_all_classes)
                kmeans.fit(results)
                pred = kmeans.labels_
                old_mask = (gt_labels < num_old_classes)
                if only_target:
                    # gt_labels = gt_labels[self.cumsum[0]:]  # this line is in the for loop, gt_labels should not be changed
                    pred = pred[self.cumsum[0]:]
                    old_mask = old_mask[self.cumsum[0]:]
                    total_acc, old_acc, new_acc, conf = split_cluster_acc_v2(gt_labels[self.cumsum[0]:], pred, old_mask, return_conf=True)
                    total_acc_balanced, old_acc_balanced, new_acc_balanced = split_cluster_acc_v2_balanced(gt_labels[self.cumsum[0]:], pred, old_mask)
                else:
                    total_acc, old_acc, new_acc, conf = split_cluster_acc_v2(gt_labels, pred, old_mask, return_conf=True)
                    total_acc_balanced, old_acc_balanced, new_acc_balanced = split_cluster_acc_v2_balanced(gt_labels, pred, old_mask)
                log_msg = '\n' + inspect.cleandoc(f'''
                    K-Means:
                        ALL: {total_acc:.4f}
                        Old: {old_acc:.4f}
                        New: {new_acc:.4f}
                    K-Means (Balanced):
                        ALL: {total_acc_balanced:.4f}
                        Old: {old_acc_balanced:.4f}
                        New: {new_acc_balanced:.4f}
                ''')
                eval_results['kmeans'] = total_acc
                eval_results['kmeans_old'] = old_acc
                eval_results['kmeans_new'] = new_acc
                eval_results['kmeans_balanced'] = total_acc_balanced
                eval_results['kmeans_balanced_old'] = old_acc_balanced
                eval_results['kmeans_balanced_new'] = new_acc_balanced

                # confmat
                h, w = conf.shape
                with np.printoptions(threshold=np.inf, linewidth=np.inf):  # thres: # elems, width: # chars
                    s = str(conf)
                if 'SRUN_DEBUG' in os.environ:  # if in srun(debuging) session
                    for (ii, jj), (start, end) in reversed([((i//w, i%h), (m.start(0), m.end(0))) for i, m in enumerate(re.finditer(r'\d+', str(s))) if i%h == i//w]):
                        s = s[:start] + BOLD + (GREEN if max(ii, jj) < num_old_classes else PURPLE) + s[start:end] + END + s[end:]  # diag vals to be bold
                log_msg += '\nConfmat (gt/pred)\n' + s + '\n'
                print_log(log_msg, logger=logger)
                continue

            if metric == 'sskmeans':
                # to resolve circular import
                from mmaction.datasets.dataset_wrappers import ConcatDataset
                assert type(self) == ConcatDataset
                gt_labels = np.array(gt_labels)
                Xs = {
                    'train_source': results[:self.cumsum[0]],
                    'train_target': results[self.cumsum[0]:self.cumsum[1]],
                    'valid': results[self.cumsum[1]:self.cumsum[2]],
                    'test': results[self.cumsum[2]:],
                }
                anns = {
                    'train_source': gt_labels[:self.cumsum[0]],
                    'train_target': None,
                    'valid': gt_labels[self.cumsum[1]:self.cumsum[2]],  # cheat
                    'test': gt_labels[self.cumsum[2]:],
                }
                metric_option = metric_options.get('sskmeans', {})
                n_tries = metric_option.get('n_tries', 1)  # default = {'sskmeans': {'n_tries': 1}}
                fixed_k = metric_option.get('fixed_k', None)
                _, ks, rows = train_sskmeans(
                    Xs, anns, num_known_classes=self.num_classes,
                    fixed_k=fixed_k, n_tries=n_tries, verbose=False
                )
                if fixed_k:
                    # ks = [fixed_k]
                    global_best_mean_test_score, os_star, unk = rows[0][-4], rows[0][-2], rows[0][-1]
                    log_msg = f'\nSS k-means H\t{global_best_mean_test_score:.4f}\n{"OS*":>14s}\t{os_star:.4f}\n{"UNK":>14s}\t{unk:.4f} (fixed_k: {fixed_k})'
                else:
                    global_best_mean_test_score, global_best_mean_test_k = max(zip([means[-4] for means in rows], ks), key=lambda zipped_row: zipped_row[0])
                    log_msg = f'\nSS k-means (H)\t{global_best_mean_test_score:.4f} (best_k: {global_best_mean_test_k})'
                eval_results[metric] = global_best_mean_test_score
                eval_results['os*'] = os_star
                eval_results['recall_unknown'] = unk
                print_log(log_msg, logger=logger)
                continue

            if metric == 'logits':
                p_out_dir = metric_options.get('logits', {}).get('p_out_dir', None)
                assert p_out_dir is not None, "Specify the out dir in metric_options['logits']['p_out_dir']"
                p_out = Path(p_out_dir) / 'logits' / f'{int(time.time())}.pkl'
                p_out.parent.mkdir(exist_ok=True)
                y_ = np.array(results)
                with p_out.open('wb') as f:
                    pickle.dump(y_, f)
                log_msg = f'\nSaving logits at {str(p_out)}'
                continue

            if metric == 'gcd_v2':
                # to resolve circular import
                from mmaction.datasets.dataset_wrappers import ConcatDataset
                from slurm.gcd4da.commons.kmeans import SSKMeansTrainer as SSKMeans
                if type(self) != ConcatDataset:
                    print_log('\tpassed because `self` is not an instance of `ConcatDataset`\n', logger=logger)
                    continue

                gt_labels = np.array(gt_labels)
                num_old_classes = metric_options.setdefault('num_old_classes', gt_labels.max()+1)
                num_all_classes = metric_options.setdefault('num_all_classes', gt_labels.max()+1)

                feat_source = results[:self.cumsum[0]]
                feat_target = results[self.cumsum[0]:]
                gt_source = gt_labels[:self.cumsum[0]]
                gt_target = gt_labels[self.cumsum[0]:]
                sskmeans = SSKMeans(ks=num_all_classes, autoinit=False, num_known_classes=num_old_classes, verbose=False)
                sskmeans.Xs = {
                    'train_source': feat_source[gt_source<num_old_classes],
                    'train_target': feat_target,
                    'valid': feat_target,
                    'test': feat_target,
                }
                sskmeans.anns = {
                    'train_source': gt_source[gt_source<num_old_classes],
                    'train_target': gt_target,
                    'valid': gt_target,
                    'test': gt_target,
                }
                sskmeans.train()

                pred_target = sskmeans.predict(sskmeans.model_best)['test']
                old_mask = (gt_target < num_old_classes)
                total_acc, old_acc, new_acc, conf = split_cluster_acc_v2(gt_target, pred_target, old_mask, True)
                log_msg = '\n' + inspect.cleandoc(f'''
                    SS k-means (GCD V2; ss k-means -ed on source+target train sets and evaluated on target train set):
                        ALL: {total_acc:.4f}
                        Old: {old_acc:.4f}
                        New: {new_acc:.4f}
                ''')
                eval_results['gcd_v2'] = total_acc
                eval_results['gcd_v2_old'] = old_acc
                eval_results['gcd_v2_new'] = new_acc
                total_acc, old_acc, new_acc = split_cluster_acc_v2_balanced(gt_target, pred_target, old_mask)
                log_msg += '\n' + inspect.cleandoc(f'''
                    SS k-means (GCD V2 Balanced):
                        ALL: {total_acc:.4f}
                        Old: {old_acc:.4f}
                        New: {new_acc:.4f}
                ''')
                eval_results['gcd_v2_balanced'] = total_acc
                eval_results['gcd_v2_balanced_old'] = old_acc
                eval_results['gcd_v2_balanced_new'] = new_acc

                # confmat
                h, w = conf.shape
                with np.printoptions(threshold=np.inf, linewidth=np.inf):  # thres: # elems, width: # chars
                    s = str(conf)
                if 'SRUN_DEBUG' in os.environ:  # if in srun(debugging) session
                    for (ii, jj), (start, end) in reversed([((i//w, i%h), (m.start(0), m.end(0))) for i, m in enumerate(re.finditer(r'\d+', str(s))) if i%h == i//w]):
                        s = s[:start] + BOLD + (GREEN if max(ii, jj) < num_old_classes else PURPLE) + s[start:end] + END + s[end:]  # diag vals to be bold
                log_msg += '\nConfmat (gt/pred)\n' + s + '\n'

                print_log(log_msg, logger=logger)
                continue

            if metric == 'gcd_v2_cheated':
                gt_labels = np.array(gt_labels)
                num_old_classes = metric_options.setdefault('num_old_classes', gt_labels.max()+1)
                num_all_classes = metric_options.setdefault('num_all_classes', gt_labels.max()+1)
                pred = np.argmax(results, axis=1)
                conf = confusion_matrix(pred, gt_labels)
                corrects = conf.diagonal()
                topk_all = corrects.sum() / gt_labels.shape[0]
                topk_old = corrects[:num_old_classes].sum() / conf[:num_old_classes].sum()
                topk_new = corrects[num_old_classes:].sum() / conf[num_old_classes:].sum()
                recalls = corrects / conf.sum(axis=1)
                mca_all = recalls.mean()
                mca_old = recalls[:num_old_classes].mean()
                mca_new = recalls[num_old_classes:].mean()
                log_msg = '\n' + inspect.cleandoc(f'''
                    Cheated Top-1 scores with old/new separated
                        ALL: {topk_all:.4f}
                        Old: {topk_old:.4f}
                        New: {topk_new:.4f}

                    Cheated MCA scores with old/new separated
                        ALL: {mca_all:.4f}
                        Old: {mca_old:.4f}
                        New: {mca_new:.4f}
                ''')
                eval_results['gcd_v2_cheated'] = topk_all
                eval_results['gcd_v2_cheated_old'] = topk_old
                eval_results['gcd_v2_cheated_new'] = topk_new
                eval_results['gcd_v2_cheated_balanced'] = mca_all
                eval_results['gcd_v2_cheated_balanced_old'] = mca_old
                eval_results['gcd_v2_cheated_balanced_new'] = mca_new
                print_log(log_msg, logger=logger)
                continue

        return eval_results

    @staticmethod
    def dump_results(results, out):
        """Dump data to json/yaml/pickle strings or files."""
        return mmcv.dump(results, out)

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            return self.prepare_test_frames(idx)

        return self.prepare_train_frames(idx)
