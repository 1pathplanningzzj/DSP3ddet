from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class GaussianPruningEpochHook(Hook):
    def _set_current_epoch(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        head = getattr(model, 'head', None)
        if head is not None and hasattr(head, 'current_epoch'):
            head.current_epoch = runner.epoch

    def before_train_epoch(self, runner):
        self._set_current_epoch(runner)

    def before_val_epoch(self, runner):
        self._set_current_epoch(runner)
