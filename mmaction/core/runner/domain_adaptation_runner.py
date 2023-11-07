import time
import warnings

import mmcv
from mmcv.runner import EpochBasedRunner, Hook
from mmcv.runner.utils import get_host_info
from .omnisource_runner import OmniSourceDistSamplerSeedHook


DomainAdaptationDistSamplerSeedHook = OmniSourceDistSamplerSeedHook


def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


class DomainAdaptationRunner(EpochBasedRunner):

    def run_iter(self, data_batches, domains, train_mode, **kwargs):
        if self.batch_processor is not None:
            # not actually used
            outputs = self.batch_processor(
                self.model, data_batches, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batches, domains, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batches, domains, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        # No need to differentiate the log_vars because, different from OmniSource
        # this run_iter method is to be called only once at a step
        if 'log_vars' in outputs:
            log_vars = outputs['log_vars']
            # log_vars = {k + source: v for k, v in log_vars.items()}
            self.log_buffer.update(log_vars, outputs['num_samples'])

        self.outputs = outputs

    def train(self, data_loaders, **kwargs):
        """
        Different from OmniSourceRunner's
        - No train_ratio
        - All domains' data are fed forward together in a single step
        - Injecting domain names to run_iter
        """
        self.model.train()
        self.mode = 'train'
        self.data_loaders = data_loaders
        self.data_loader = data_loaders[0]  # for Hook
        self._max_iters = self._max_epochs * len(self.data_loaders[0])
        self.domains = ['source'] + [f'target{i}' for i in range(1, len(data_loaders))]
        self.aux_iters = [cycle(loader) for loader in self.data_loaders[1:]]
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        # the number of iterations is fit to the main loader
        for i, data_batches in enumerate(zip(*(self.data_loaders[:1]+self.aux_iters))):
            main_batch_length = min(data_batches[0]['imgs'].shape[0], data_batches[1]['imgs'].shape[0])
            for idx_data_batch in range(len(data_batches)):  # when drop_last=False, main_loader's batch size may differ
                data_batches[idx_data_batch]['imgs']  = data_batches[idx_data_batch]['imgs'][:main_batch_length]
                data_batches[idx_data_batch]['label'] = data_batches[idx_data_batch]['label'][:main_batch_length]
            self._inner_iter = i
            self.call_hook('before_train_iter')
            kwargs['iter'] = self._iter
            kwargs['cur_epoch'] = self._epoch  # for edl loss
            kwargs['total_epoch'] = self._max_epochs  # for edl loss
            self.run_iter(data_batches, self.domains, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training.
                `data_loaders[0]` is the main data_loader, which contains
                target datasets and determines the epoch length.
                `data_loaders[1:]` are auxiliary data loaders, which contain
                auxiliary web datasets.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2)] means running 2
                epochs for training iteratively. Note that val epoch is not
                supported for this runner for simplicity.
            max_epochs (int | None): The max epochs that training lasts,
                deprecated now. Default: None.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(workflow) == 1 and workflow[0][0] == 'train'
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        mode, epochs = workflow[0]
        self._max_iters = self._max_epochs * len(data_loaders[0])

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            if isinstance(mode, str):  # self.train()
                if not hasattr(self, mode):
                    raise ValueError(
                        f'runner has no method named "{mode}" to run an '
                        'epoch')
                epoch_runner = getattr(self, mode)
            else:
                raise TypeError(
                    f'mode in workflow must be a str, but got {mode}')

            for _ in range(epochs):
                if mode == 'train' and self.epoch >= self._max_epochs:
                    break
                epoch_runner(data_loaders, **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')
