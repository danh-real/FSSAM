import os
import sys
import datetime
import random
import time
import cv2
import numpy as np
import logging
import argparse
import math
from visdom import Visdom
from tabulate import tabulate
import os.path as osp

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch import autocast, GradScaler
from torchvision import transforms

from tensorboardX import SummaryWriter

from model import FSSAM, FSSAM5s

from util import dataset, dataset_sarcoma, dataset_btcv, dataset_msd
from util import transform_new as transform, transform_tri, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, get_model_para_number, setup_seed, \
    get_logger, get_save_path, \
    is_same_model, fix_bn, sum_list, check_makedirs
import matplotlib.pyplot as plt

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
val_manual_seed = 123
setup_seed(val_manual_seed, True)
seed_array = [321]
val_num = len(seed_array)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Few-Shot Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='FSSAM')
    parser.add_argument('--viz', action='store_true', default=False)
    parser.add_argument('--config', type=str, default='config/coco/coco_split3_resnet50.yaml',
                        help='config file')  # coco/coco_split0_resnet50.yaml
    parser.add_argument('--num_refine', type=int, default=3,
                        help='number of memory refinement')
    parser.add_argument('--ver_refine', type=str, default="v1",
                        help='version of memory refinement')
    parser.add_argument('--ver_dino', type=str, default="dinov2_vitb14", choices=["dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
                        help="version of dino")
    parser.add_argument('--episode', help='number of test episodes', type=int, default=1000)
    parser.add_argument('--opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg = config.merge_cfg_from_args(cfg, args)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def score_cal(seg_map, prd_map):
    '''
    seg_map B * H * W
    prd_map B * H * W
    '''
    assert seg_map.ndim == 2 or seg_map.ndim == 3
    assert prd_map.ndim == 2 or prd_map.ndim == 3
    if seg_map.ndim == 2:
        seg_map = seg_map.unsqueeze(0)
        prd_map = prd_map.unsqueeze(0)
        
    total_num = seg_map.shape[0]
    
    seg_map = seg_map.reshape(total_num, -1)
    prd_map = prd_map.reshape(total_num, -1)
    dot_product = (seg_map * prd_map)
    b_seg_map = 1 - seg_map
    b_prd_map = 1 - prd_map
    b_dot_product = (b_seg_map * b_prd_map)

    sum_dot = torch.sum(dot_product, dim=-1)
    sum_seg = torch.sum(seg_map, dim=-1)
    sum_prd = torch.sum(prd_map, dim=-1)
    b_sum_dot = torch.sum(b_dot_product, dim=-1)
    b_sum_seg = torch.sum(b_seg_map, dim=-1)
    b_sum_prd = torch.sum(b_prd_map, dim=-1)

    iou_score = sum_dot/((sum_seg + sum_prd)-sum_dot)
    dice_score = 2.*sum_dot / (sum_seg+sum_prd)
    
    b_iou_score = b_sum_dot/((b_sum_seg + b_sum_prd)-b_sum_dot)
    fb_iou_score = (iou_score + b_iou_score) / 2

    return iou_score, dice_score, fb_iou_score

def eval_seg(pred, mask):
    """
    Args:
        pred: [D, H, W]
        mask: [D, H, W]
    """
    pred = (torch.sigmoid(pred) > 0.5).float()
    iou, dice, fb_iou = score_cal(mask, pred)
    
    iou[iou.isnan()] = 0. 
    dice[dice.isnan()] = 0.
    fb_iou[fb_iou.isnan()] = 0.
    
    return iou, dice, fb_iou

def get_model(args):
    model = eval(args.arch).OneModel(args)
    optimizer = model.get_optim(model, args, LR=args.base_lr)

    model = model.cuda()

    # Resume
    get_save_path(args)
    check_makedirs(args.snapshot_path)
    check_makedirs(args.result_path)
    
    print(args.weight)

    if args.weight:
        weight_path = osp.join(args.snapshot_path, args.weight)
        if os.path.isfile(weight_path):
            logger.info("=> loading checkpoint '{}'".format(weight_path))
            checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try:
                model.load_state_dict(new_param)
            except RuntimeError:                   # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(weight_path, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(weight_path))

    # ========================================
    # use bfloat16 for the entire program
    # ========================================
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    # Get model para.
    total_number, learnable_number = get_model_para_number(model)
    print('Number of Parameters: %d' % (total_number))
    print('Number of Learnable Parameters: %d' % (learnable_number))

    time.sleep(5)
    return model, optimizer


def main():
    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    args.distributed = True if torch.cuda.device_count() > 1 else False
    print(args)

    if args.manual_seed is not None:
        setup_seed(args.manual_seed, args.seed_deterministic)

    logger.info("=> creating model ...")
    model, optimizer = get_model(args)
    # logger.info(model)

    # ----------------------  DATASET  ----------------------
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    # Val
    if args.evaluate:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(args.train_h, args.train_w), interpolation=transforms.InterpolationMode.BILINEAR),
        ])
        if args.data_set == 'pascal' or args.data_set == 'coco':
            val_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root,
                                       data_list=args.val_list, transform=val_transform, mode='val',
                                       ann_type=args.ann_type, data_set=args.data_set, use_split_coco=args.use_split_coco)
        elif args.data_set ==  'sarcoma':
            val_data = dataset_sarcoma.Sarcoma(split=args.split, shot=args.shot, data_root=args.data_root,
                                       data_list=args.val_list, transform=val_transform, mode='val',
                                       ann_type=args.ann_type, data_set=args.data_set, use_split_coco=args.use_split_coco)
        elif args.data_set == 'btcv':
            val_data = dataset_btcv.BTCV(split=args.split, shot=args.shot, data_root=args.data_root,
                                        data_list=args.train_list, transform=val_transform, mode='val',
                                        ann_type=args.ann_type, data_set=args.data_set, use_split_coco=args.use_split_coco,
                                        image_size=(args.train_h, args.train_w))
        elif args.data_set.startswith('msd'):
            val_data = dataset_msd.MSD(split=args.split, shot=args.shot, data_root=args.data_root,
                                        data_list=args.train_list, transform=val_transform, mode='val',
                                        ann_type=args.ann_type, data_set=args.data_set, use_split_coco=args.use_split_coco,
                                        image_size=(args.train_h, args.train_w))
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                                 num_workers=args.workers, pin_memory=False, sampler=None)

    # ========================================
    # Test one batch first to warmup
    # Global autocast needs to cache the conversion of fp32->bfp16
    # ========================================
    validate(val_loader, model, 321, args.episode, warmup=True)

    # ----------------------  VAL  ----------------------
    start_time = time.time()
    FBIoU_array = np.zeros(val_num)
    mIoU_array = np.zeros(val_num)
    mDice_array = np.zeros(val_num)
    for val_id in range(val_num):
        val_seed = seed_array[val_id]
        print('Val: [{}/{}] \t Seed: {}'.format(val_id + 1, val_num, val_seed))
        miou, dice, fb_iou = validate(val_loader, model, val_seed, args.episode)
        FBIoU_array[val_id], mIoU_array[val_id], mDice_array[val_id] = \
            fb_iou, miou, dice

    total_time = time.time() - start_time
    t_m, t_s = divmod(total_time, 60)
    t_h, t_m = divmod(t_m, 60)
    total_time = '{:02d}h {:02d}m {:02d}s'.format(int(t_h), int(t_m), int(t_s))

    print('\nTotal running time: {}'.format(total_time))
    print('Seed0: {}'.format(val_manual_seed))
    print('Seed:  {}'.format(seed_array))
    print('mIoU:  {}'.format(np.round(mIoU_array, 4)))
    print('FBIoU: {}'.format(np.round(FBIoU_array, 4)))
    print('mDice:  {}'.format(np.round(mDice_array, 4)))
    print('-' * 43)
    print('Best_Seed_m: {} \t Best_Seed_F: {} \t Best_Seed_p: {}'.format(seed_array[mIoU_array.argmax()],
                                                                         seed_array[FBIoU_array.argmax()],
                                                                         seed_array[mDice_array.argmax()]))
    # print('Best_mIoU: {:.4f} \t Best_FBIoU: {:.4f} \t Best_pIoU: {:.4f}'.format(
    #     mIoU_array.max(), FBIoU_array.max(), mDice_array.max()))
    print('Mean_mIoU: {:.4f} \t Mean_FBIoU: {:.4f} \t Mean_mDice: {:.4f}'.format(
        mIoU_array.mean(), FBIoU_array.mean(), mDice_array.mean()))


def validate(val_loader, model, val_seed, episode, warmup=False):
    if not warmup:
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    intersection_meter = AverageMeter()  # final
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    if args.data_set == 'pascal':
        test_num = 1000
        split_gap = 5
    elif args.data_set == 'coco':
        test_num = episode if not warmup else 1000
        split_gap = 20
    else:
        test_num = len(val_loader)
        split_gap = 0

    class_intersection_meter = [0] * split_gap
    class_union_meter = [0] * split_gap

    score_per_class = {}
    setup_seed(val_seed, args.seed_deterministic)

    pos_weight = torch.ones([1]).cuda() * 2
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.eval()
    end = time.time()
    val_start = end

    assert test_num % args.batch_size_val == 0
    db_epoch = math.ceil(test_num / (len(val_loader) - args.batch_size_val))
    iter_num = 0

    for e in range(db_epoch):
        for i, (input, target, s_input, s_mask, subcls, ori_label) in enumerate(val_loader):
            if iter_num == 1 and warmup: break

            if iter_num * args.batch_size_val >= test_num:
                break
            iter_num += 1
            data_time.update(time.time() - end)

            s_input = s_input.cuda(non_blocking=True)
            s_mask = s_mask.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            ori_label = ori_label.cuda(non_blocking=True)

            priors = None

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    start_time = time.time()
                    output, priors = model(s_x=s_input, s_y=s_mask, x=input, y_m=target, cat_idx=subcls, priors=priors)
                    model_time.update(time.time() - start_time)

                    if args.ori_resize:
                        output = F.interpolate(output.unsqueeze(0), size=ori_label.size()[-2:], mode='bilinear', align_corners=True)
                        output = output.squeeze(0)
                        target = ori_label.long()

                    output = F.interpolate(output.unsqueeze(0), size=target.size()[1:], mode='bilinear', align_corners=True)
                    output = output.squeeze(0)
                    # label = target.clone()
                    # label[label == 255] = 0
                    # loss = criterion(output, label.float())
        
            output = torch.sigmoid(output)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            subcls = subcls[0].cpu().numpy()[0]
            if subcls not in score_per_class.keys():
                score_per_class[subcls] = {
                    "iou": torch.FloatTensor([]).cuda(non_blocking=True),
                    "dice": torch.FloatTensor([]).cuda(non_blocking=True),
                    "fb_iou": torch.FloatTensor([]).cuda(non_blocking=True),
                }
            (
                iou,
                dice,
                fb_iou,
            ) = eval_seg(output, target)
            
            score_dict = score_per_class[subcls]
            score_dict["iou"] = torch.cat([score_dict["iou"], iou.detach()])
            score_dict["dice"] = torch.cat([score_dict["dice"], dice.detach()])
            score_dict["fb_iou"] = torch.cat([score_dict["fb_iou"], fb_iou.detach()])

            # intersection, union, new_target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
            # intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
            # intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)
            # class_intersection_meter[subcls] += intersection[1]
            # class_union_meter[subcls] += union[1]

            # accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            # loss_meter.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            remain_iter = test_num / args.batch_size_val - iter_num
            remain_time = remain_iter * batch_time.avg
            t_m, t_s = divmod(remain_time, 60)
            t_h, t_m = divmod(t_m, 60)
            remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

            if ((i + 1) % round((test_num / 20)) == 0):
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Remain {remain_time} '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}).'.format(iter_num * args.batch_size_val, test_num,
                                                              data_time=data_time,
                                                              batch_time=batch_time,
                                                              remain_time=remain_time,
                                                              loss_meter=loss_meter))
    if not warmup:
        avg = {
            "iou": torch.FloatTensor([]).cuda(non_blocking=True),
            "dice": torch.FloatTensor([]).cuda(non_blocking=True),
            "fb_iou": torch.FloatTensor([]).cuda(non_blocking=True),
        }
        
        table_data = []
        
        for name, metrics_dict in score_per_class.items():
            miou = metrics_dict["iou"].mean(dim=0, keepdim=True)
            mdice = metrics_dict["dice"].mean(dim=0, keepdim=True)
            mfb_iou = metrics_dict["fb_iou"].mean(dim=0, keepdim=True)
            
            table_data.append((
                name, 
                miou.item(), 
                mdice.item(), 
                mfb_iou.item(),
            ))
            
            avg["iou"] = torch.cat([avg["iou"], miou])
            avg["dice"] = torch.cat([avg["dice"], mdice])
            avg["fb_iou"] = torch.cat([avg["fb_iou"], mfb_iou])
            
        avg["iou"] = avg["iou"].mean()
        avg["dice"] = avg["dice"].mean()
        avg["fb_iou"] = avg["fb_iou"].mean()
                
        table_data.append((
            "Average",
            avg["iou"].item(),
            avg["dice"].item(),
            avg["fb_iou"].item(),
        ))

        print(tabulate(table_data, headers=["name", "iou", "dice", "fb_iou"], floatfmt=".4f", tablefmt="grid"))

        return avg["iou"], avg["dice"], avg["fb_iou"]


if __name__ == '__main__':
    main()
