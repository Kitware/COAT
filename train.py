# This file is part of COAT, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE file or
# https://github.com/Kitware/COAT/blob/master/LICENSE for details.

import argparse
import datetime
import os.path as osp
import time

import torch
import torch.utils.data

from datasets import build_test_loader, build_train_loader
from defaults import get_default_cfg
from engine import evaluate_performance, train_one_epoch
from models.coat import COAT
from utils.utils import mkdir, resume_from_ckpt, save_on_master, set_random_seed

from loss.softmax_loss import SoftmaxLoss


def main(args):
    cfg = get_default_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(cfg.DEVICE)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    print("Creating model...")
    model = COAT(cfg)
    model.to(device)

    print("Loading data...")
    train_loader = build_train_loader(cfg)
    gallery_loader, query_loader = build_test_loader(cfg)

    softmax_criterion_s2 = None
    softmax_criterion_s3 = None
    if cfg.MODEL.LOSS.USE_SOFTMAX:
        softmax_criterion_s2 = SoftmaxLoss(cfg)
        softmax_criterion_s3 = SoftmaxLoss(cfg)
        softmax_criterion_s2.to(device)
        softmax_criterion_s3.to(device)

    if args.eval:
        assert args.ckpt, "--ckpt must be specified when --eval enabled"
        resume_from_ckpt(args.ckpt, model)
        evaluate_performance(
            model,
            gallery_loader,
            query_loader,
            device,
            use_gt=cfg.EVAL_USE_GT,
            use_cache=cfg.EVAL_USE_CACHE,
            use_cbgm=cfg.EVAL_USE_CBGM,
            gallery_size=cfg.EVAL_GALLERY_SIZE,
        )
        exit(0)

    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.MODEL.LOSS.USE_SOFTMAX:
        params_softmax_s2 = [p for p in softmax_criterion_s2.parameters() if p.requires_grad]
        params_softmax_s3 = [p for p in softmax_criterion_s3.parameters() if p.requires_grad]
        params.extend(params_softmax_s2)
        params.extend(params_softmax_s3)

    optimizer = torch.optim.SGD(
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.SGD_MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.SOLVER.LR_DECAY_MILESTONES, gamma=cfg.SOLVER.GAMMA
    )

    start_epoch = 0
    if args.resume:
        assert args.ckpt, "--ckpt must be specified when --resume enabled"
        start_epoch = resume_from_ckpt(args.ckpt, model, optimizer, lr_scheduler) + 1

    print("Creating output folder...")
    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)
    path = osp.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        f.write(cfg.dump())
    print(f"Full config is saved to {path}")
    tfboard = None
    if cfg.TF_BOARD:
        from torch.utils.tensorboard import SummaryWriter

        tf_log_path = osp.join(output_dir, "tf_log")
        mkdir(tf_log_path)
        tfboard = SummaryWriter(log_dir=tf_log_path)
        print(f"TensorBoard files are saved to {tf_log_path}")

    print("Start training...")
    start_time = time.time()
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        train_one_epoch(cfg, model, optimizer, train_loader, device, epoch, tfboard, softmax_criterion_s2, softmax_criterion_s3)
        lr_scheduler.step()

        # only save the last three checkpoints
        if epoch >= cfg.SOLVER.MAX_EPOCHS - 3:
            save_on_master(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                },
                osp.join(output_dir, f"epoch_{epoch}.pth"),
            )

        # evaluate the current checkpoint
            evaluate_performance(
                model,
                gallery_loader,
                query_loader,
                device,
                use_gt=cfg.EVAL_USE_GT,
                use_cache=cfg.EVAL_USE_CACHE,
                use_cbgm=cfg.EVAL_USE_CBGM,
                gallery_size=cfg.EVAL_GALLERY_SIZE,
            )

    if tfboard:
        tfboard.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time {total_time_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument("--cfg", dest="cfg_file", help="Path to configuration file.")
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate the performance of a given checkpoint."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the specified checkpoint."
    )
    parser.add_argument("--ckpt", help="Path to checkpoint to resume or evaluate.")
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line"
    )
    args = parser.parse_args()
    main(args)
