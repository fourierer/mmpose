import argparse
import os
import os.path as osp

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmpose.apis import multi_gpu_test, single_gpu_test
from mmpose.core import wrap_fp16_model
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet
# for test-dev
import json
import numpy as np
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def parse_args():
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        default='mAP',
        nargs='+',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "mAP" for MSCOCO')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    args.work_dir = osp.join('./work_dirs',
                             osp.splitext(osp.basename(args.config))[0])
    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_posenet(cfg.model)
    fp16_cfg = cfg.get('fp16', None) # None
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    _ = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # for backward compatibility
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        # print(args.gpu_collect) # False
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)
    
    rank, _ = get_dist_info()
    eval_config = cfg.get('eval_config', {})
    eval_config = merge_configs(eval_config, dict(metric=args.eval))
    
    args.out = args.work_dir + '/test_dev2017_results_kps.json'

    if rank == 0:
        if args.out:
            # output_file_path = args.work_dir + '/outputs.json'
            # print(f'\nwirting oututs to {output_file_path}')
            # print(type(outputs)) # list,需要在rank=0下进行outputs的相关操作，因为outputs只有第一张卡有输出
            # with open(output_file_path,'w') as f:
                # json.dump(outputs, f, cls=NpEncoder) # 需要加个NpEncoder类，因为outputs中有np.array，而json无法将array写入json文件
            print(f'\nwriting results to {args.out}')
            results = []
            for img in outputs:
                for i, person in enumerate(img[0]):
                    kps = person[:, :3]
                    kps = kps.reshape((-1)).round(3).tolist()
                    kps = [round(k, 3) for k in kps] # 保留3位小数
                    # score = round(float(img[1][i]), 3) # 第5位是bbox_score，/mmpose/mmpose/models/detectors/top_down.py 272行
                    score = round(float(img[1][i][5]),3)
                    id = ''
                    for key in img[2][19:31]:
                        id = id + key
                    results.append({
                        'category_id': int(1),
                        'image_id': int(id),
                        'keypoints': kps,
                        'score': score
                    })
            with open(args.out,'w') as fid:
                json.dump(results, fid)
            # mmcv.dump(outputs, args.out)
        print(dataset.evaluate(outputs, args.work_dir, **eval_config))


if __name__ == '__main__':
    main()
