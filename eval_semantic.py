import os
import argparse

import custom_datasets
from model_semantic import *

from mmengine.config import Config
from mmengine.runner import Runner



def parse_args():
    parser = argparse.ArgumentParser(description='TextRegion evaluation with MMSeg')
    parser.add_argument('--config', default='./configs/cfg_ds/cfg_voc20.py')
    parser.add_argument("--clip_pretrained", type=str, choices=['openai', 'laion2b_s32b_b79k', 'siglip2', 'meta'], default='openai')
    parser.add_argument("--clip_architecture", type=str, default='ViT-B/16',
                        choices=['ViT-B/16', 'ViT-H/14', 'ViT-B-16-SigLIP2', 'ViT-SO400M-16-SigLIP2-384', 'PE-Core-B16-224', 'PE-Core-L14-336'])

    parser.add_argument('--work-dir', default='./eval_results/debug')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg.model.clip_pretrained = args.clip_pretrained
    cfg.model.clip_architecture = args.clip_architecture
    cfg.model.config = args.config

    cfg.launcher = args.launcher
    cfg.work_dir = args.work_dir


    runner = Runner.from_cfg(cfg)
    results = runner.test()

    if runner.rank == 0:
        with open(os.path.join(cfg.work_dir, 'results.txt'), 'a') as f:
            f.write(os.path.basename(args.config).split('.')[0] + '\n')
            f.write(f'mIoU: {results["mIoU"]} aAcc: {results["aAcc"]} mAcc: {results["mAcc"]}' + '\n')

    print(f'config: {args.config}, clip_pretrained: {args.clip_pretrained}, clip_architecture: {args.clip_architecture}')



if __name__ == '__main__':
    main()