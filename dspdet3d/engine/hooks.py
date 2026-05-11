from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class GaussianPruningEpochHook(Hook):
    def before_train_epoch(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        head = getattr(model, 'head', None)
        if head is not None and hasattr(head, 'current_epoch'):
            head.current_epoch = runner.epoch
