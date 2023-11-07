import time
from mmcv.runner import EpochBasedRunner, RUNNERS


@RUNNERS.register_module()
class EpochBasedRunnerWrapper(EpochBasedRunner):  # passes epoch and iter infos
    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(
                data_batch,
                train_mode=True,
                iter=self.iter,
                cur_epoch=self.epoch,
                total_epoch=self.max_epochs,
                **kwargs
            )
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1
