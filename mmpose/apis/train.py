import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, EpochBasedRunner, OptimizerHook

from mmpose.core import (DistEvalHook, EvalHook, Fp16OptimizerHook,
                         build_optimizers)
from mmpose.core.distributed_wrapper import DistributedDataParallelWrapper
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.utils import get_root_logger


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """Train model entry function.

    Args:
        model (nn.Module): The model to be trained.
        dataset (Dataset): Train dataset.
        cfg (dict): The config dict for training.
        distributed (bool): Whether to use distributed training.
            Default: False.
        validate (bool): Whether to do evaluation. Default: False.
        timestamp (str | None): Local time for runner. Default: None.
        meta (dict | None): Meta dict to record some important information.
            Default: None
    """
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    dataloader_setting = dict(
        samples_per_gpu=cfg.data.get('samples_per_gpu', {}), # 获取cfg.data中键samples_per_gpu对应的值，如果没有该键，则返回{}
        workers_per_gpu=cfg.data.get('workers_per_gpu', {}),
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed)
    # print(dataloader_setting) # {'samples_per_gpu': 64, 'workers_per_gpu': 2, 'num_gpus': 1, 'dist': True, 'seed': None}
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('train_dataloader', {}))
    # print(dataloader_setting) # {'samples_per_gpu': 64, 'workers_per_gpu': 2, 'num_gpus': 1, 'dist': True, 'seed': None}

    data_loaders = [
        build_dataloader(ds, **dataloader_setting) for ds in dataset
    ]

    # determine wether use adversarial training precess or not
    use_adverserial_train = cfg.get('use_adversarial_train', False)
    # print(use_adverserial_train) # False

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel

        if use_adverserial_train:
            # Use DistributedDataParallelWrapper for adversarial training
            model = DistributedDataParallelWrapper(
                model,
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            model = MMDistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizers(model, cfg.optimizer)
    # print(cfg.work_dir)
    # print(isinstance(optimizer, dict)) # False

    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp
    # print(runner._max_epochs) # None
    # print(runner._max_iters) # None
    if use_adverserial_train:
        # The optimizer step process is included in the train_step function
        # of the model, so the runner should NOT include optimizer hook.
        optimizer_config = None
    else:
        # fp16 setting
        fp16_cfg = cfg.get('fp16', None)
        # print(fp16_cfg) # None
        if fp16_cfg is not None:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
        elif distributed and 'type' not in cfg.optimizer_config:
            optimizer_config = OptimizerHook(**cfg.optimizer_config)
        else:
            optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks，加载验证数据集
    if validate:
        eval_cfg = cfg.get('evaluation', {})
        # print(eval_cfg) # {'interval': 200, 'metric': 'mAP', 'key_indicator': 'AP'}
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        dataloader_setting = dict(
            # samples_per_gpu=cfg.data.get('samples_per_gpu', {}),
            samples_per_gpu=1, # 训练时测试的每个gpu上的batch为1
            workers_per_gpu=cfg.data.get('workers_per_gpu', {}),
            # cfg.gpus will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('val_dataloader', {}))
        val_dataloader = build_dataloader(val_dataset, **dataloader_setting)
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from) # resume从某一个epoch处（如epoch_2）继续训练，继承所有的状态（包括epoch，lr）
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from) # load_checkpoint用某个模型进行预训练，并不继承状态，而是从头开始训
    # print(len(data_loaders)) # 1，训练集
    # print(runner._hooks) # [<mmcv.runner.hooks.lr_updater.StepLrUpdaterHook object at 0x7f4822d7bf50>, 
                           #  <mmcv.runner.hooks.optimizer.OptimizerHook object at 0x7f4824354750>, 
                           #  <mmcv.runner.hooks.checkpoint.CheckpointHook object at 0x7f4822d7b1d0>, 
                           #  <mmcv.runner.hooks.iter_timer.IterTimerHook object at 0x7f4822d7bd90>, 
                           #  <mmcv.runner.hooks.sampler_seed.DistSamplerSeedHook object at 0x7f4822d7b350>, 
                           #  <mmpose.core.evaluation.eval_hooks.DistEvalHook object at 0x7f47a1050650>, 
                           #  <mmcv.runner.hooks.logger.text.TextLoggerHook object at 0x7f4822d7b8d0>]
    # print(len(data_loaders[0])) # 149813/(cfg.data.get('samples_per_gpu')*gpus)
    # print(len(val_dataloader)) # 6352/gpus，因为val_dataloader的samples_per_gpu设置为1，并没有采用cfg中的samples_per_gpu
    # print(runner._max_epochs) # None
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
    