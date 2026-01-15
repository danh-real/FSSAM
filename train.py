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

from util import dataset, dataset_sarcoma, dataset_msd, dataset_btcv
from util import transform_new as transform, transform_tri, config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, get_model_para_number, setup_seed, \
    get_logger, get_save_path, \
    is_same_model, fix_bn, sum_list, check_makedirs
    
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Few-Shot Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='FSSAM')  #
    parser.add_argument('--viz', action='store_true', default=False)
    parser.add_argument('--config', type=str, default='config/pascal/pascal_split0_vgg.yaml',
                        help='config file')  # coco/coco_split0_resnet50.yaml
    parser.add_argument('--local-rank', type=int, default=-1,
                        help='number of cpu threads to use during batch generation')
    parser.add_argument('--num_refine', type=int, default=3,
                        help='number of memory refinement')
    parser.add_argument('--ver_refine', type=str, default="v1",
                        help='version of memory refinement')
    parser.add_argument('--ver_dino', type=str, default="dinov2_vitb14", choices=["dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
                        help="version of dino")
    parser.add_argument('--distributed', help='training distributed', action="store_true")
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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12348'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_model(args, rank=0, world_size=0):
    # Create model and optimizer
    model = eval(args.arch).OneModel(args)
    optimizer = model.get_optim(model, args, LR=args.base_lr)

    # Freeze backbone
    if hasattr(model, 'freeze_modules'):
        model.freeze_modules(model)

    # Initialize process for distributed training
    if args.distributed:
        setup(rank, world_size)
        args.local_rank = rank
        torch.cuda.set_device(rank)
        device = torch.device('cuda', rank)
        model.to(device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    else:
        device = torch.device('cuda')
        model = model.to(device)

    # Resume
    get_save_path(args)
    check_makedirs(args.snapshot_path)
    check_makedirs(args.result_path)

    if args.resume:
        resume_path = osp.join(args.snapshot_path, args.resume)
        if os.path.isfile(resume_path):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try:
                model.load_state_dict(new_param)
            except RuntimeError:  # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(resume_path))

    # ========================================
    # use bfloat16 for the entire program
    # ========================================
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    # Get model para.
    total_number, learnable_number = get_model_para_number(model)
    if main_process():
        print('Number of Parameters: %d' % (total_number))
        print('Number of Learnable Parameters: %d' % (learnable_number))

    time.sleep(5)
    return model, optimizer


def main_process():
    return not args.distributed or (args.distributed and (args.local_rank == 0))


def main(rank=0, world_size=0):
    global args, logger, writer
    args = get_parser()
    logger = get_logger()
    args.distributed = True if torch.cuda.device_count() > 1 else False
    if main_process():
        print(args)

    # if args.manual_seed is not None:
    #     setup_seed(args.manual_seed, args.seed_deterministic)

    # Create model and optimizer
    if main_process():
        logger.info("=> creating model ...")
    model, optimizer = get_model(args, rank, world_size)
    # if main_process():
        # logger.info(model)
    if main_process() and args.viz:
        writer = SummaryWriter(args.result_path)

    # ----------------------  DATASET  ----------------------
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    # mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    # std = [item * value_scale for item in std]
    # Train
    train_transform = transforms.Compose([
        # transform.RandScale([args.scale_min, args.scale_max]),
        # transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
        # transform.RandomGaussianBlur(),
        # transform.RandomHorizontalFlip(),
        # transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.padding_label),
        transforms.ToTensor(),
        transforms.Resize(size=(args.train_h, args.train_w), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if args.data_set == 'pascal' or args.data_set == 'coco':
        train_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root,
                                     data_list=args.train_list, transform=train_transform, mode='train',
                                     ann_type=args.ann_type, data_set=args.data_set, use_split_coco=args.use_split_coco)
    elif args.data_set == 'sarcoma':
        train_data = dataset_sarcoma.Sarcoma(split=args.split, shot=args.shot, data_root=args.data_root,
                                     data_list=args.train_list, transform=train_transform, mode='train',
                                     ann_type=args.ann_type, data_set=args.data_set, use_split_coco=args.use_split_coco,
                                     image_size=(args.train_h, args.train_w))
    elif args.data_set == 'btcv':
        train_data = dataset_btcv.BTCV(split=args.split, shot=args.shot, data_root=args.data_root,
                                     data_list=args.train_list, transform=train_transform, mode='train',
                                     ann_type=args.ann_type, data_set=args.data_set, use_split_coco=args.use_split_coco,
                                     image_size=(args.train_h, args.train_w))
    elif args.data_set.startswith('msd'):
        train_data = dataset_msd.MSD(split=args.split, shot=args.shot, data_root=args.data_root,
                                     data_list=args.train_list, transform=train_transform, mode='train',
                                     ann_type=args.ann_type, data_set=args.data_set, use_split_coco=args.use_split_coco,
                                     image_size=(args.train_h, args.train_w))
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers,
                                               pin_memory=True, sampler=train_sampler, drop_last=True,
                                               shuffle=False if args.distributed else True)
    # Val
    if args.evaluate:
        val_transform = transform.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(args.train_h, args.train_w), interpolation=transforms.InterpolationMode.BILINEAR),
        ])
        if args.data_set == 'pascal' or args.data_set == 'coco':
            val_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root,
                                       data_list=args.val_list, transform=val_transform, mode='val',
                                       ann_type=args.ann_type, data_set=args.data_set, use_split_coco=args.use_split_coco)
        elif args.data_set == 'sarcoma':
            val_data = dataset_sarcoma.Sarcoma(split=args.split, shot=args.shot, data_root=args.data_root,
                                        data_list=args.train_list, transform=train_transform, mode='val',
                                        ann_type=args.ann_type, data_set=args.data_set, use_split_coco=args.use_split_coco,
                                        image_size=(args.train_h, args.train_w))
        elif args.data_set == 'btcv':
            val_data = dataset_btcv.BTCV(split=args.split, shot=args.shot, data_root=args.data_root,
                                        data_list=args.train_list, transform=train_transform, mode='val',
                                        ann_type=args.ann_type, data_set=args.data_set, use_split_coco=args.use_split_coco,
                                        image_size=(args.train_h, args.train_w))
        elif args.data_set.startswith('msd'):
            val_data = dataset_msd.MSD(split=args.split, shot=args.shot, data_root=args.data_root,
                                        data_list=args.train_list, transform=train_transform, mode='val',
                                        ann_type=args.ann_type, data_set=args.data_set, use_split_coco=args.use_split_coco,
                                        image_size=(args.train_h, args.train_w))
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                                 num_workers=args.workers, pin_memory=False, sampler=None)

    # ----------------------  TRAINVAL  ----------------------
    global best_miou, best_FBiou, best_dice, best_epoch, keep_epoch, val_num
    global best_miou_m, best_miou_b, best_FBiou_m
    best_miou = 0.
    best_FBiou = 0.
    best_dice = 0.
    best_epoch = 0
    keep_epoch = 0
    val_num = 0
    best_miou_m = 0.
    best_miou_b = 0.
    best_FBiou_m = 0.

    start_time = time.time()
    
    # ========================================
    # Test one batch first to warmup
    # Global autocast needs to cache the conversion of fp32->bfp16
    # ========================================
    validate(val_loader, model, warmup=True)

    # amp
    scaler = GradScaler()
    for epoch in range(args.start_epoch, args.epochs):
        # early stop - 75 epochs
        if keep_epoch == args.stop_interval:
            break
        if args.fix_random_seed_val:
            setup_seed(args.manual_seed + epoch, args.seed_deterministic)

        epoch_log = epoch + 1
        keep_epoch += 1
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # ----------------------  TRAIN  ----------------------
        loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, val_loader, model, optimizer, epoch, scaler)

        if main_process() and args.viz:
            writer.add_scalar('FBIoU_train', mIoU_train, epoch_log)

        # -----------------------  VAL  -----------------------
        if args.evaluate and epoch % 1 == 0:
            mIoU, mDice, mFBIoU = validate(val_loader, model)
            val_num += 1
            if main_process() and args.viz:
                writer.add_scalar('mDice_val', mDice, epoch_log)
                writer.add_scalar('FBIoU_val', mFBIoU, epoch_log)
                writer.add_scalar('mIoU_val', mIoU, epoch_log)

            # save model for <testing>
            if mDice > best_dice:
                best_miou, best_dice, best_FBiou, best_epoch = mIoU, mDice, mFBIoU, epoch
                keep_epoch = 0
                if args.shot == 1:
                    filename = args.snapshot_path + '/train_epoch_' + str(epoch) + '_{:.4f}'.format(best_miou) + '.pth'
                else:
                    filename = args.snapshot_path + '/train{}_epoch_'.format(args.shot) + str(epoch) + '_{:.4f}'.format(
                        best_miou) + '.pth'
                if main_process():
                    logger.info('Saving checkpoint to: ' + filename)
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                               filename)

    total_time = time.time() - start_time
    t_m, t_s = divmod(total_time, 60)
    t_h, t_m = divmod(t_m, 60)
    total_time = '{:02d}h {:02d}m {:02d}s'.format(int(t_h), int(t_m), int(t_s))

    if main_process():
        print('\nEpoch: {}/{} \t Total running time: {}'.format(epoch_log, args.epochs, total_time))
        print('The number of models validated: {}'.format(val_num))
        print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  Final Best Result   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print(args.arch + '\t Group:{} \t Best_step:{}'.format(args.split, best_epoch))
        print('mIoU:{:.4f}'.format(best_miou))
        print('FBIoU:{:.4f} \t pIoU:{:.4f}'.format(best_FBiou, best_dice))
        print('>' * 80)
        print('%s' % datetime.datetime.now())
        
    if args.distributed:
        cleanup()


def train(train_loader, val_loader, model, optimizer, epoch, scaler):
    global best_miou, best_FBiou, best_dice, best_epoch, keep_epoch, val_num
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter1 = AverageMeter()
    aux_loss_meter2 = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    if args.fix_bn:
        model.apply(fix_bn)  # fix batchnorm

    end = time.time()
    val_time = 0.
    max_iter = args.epochs * len(train_loader)
    if main_process():
        print('Warmup: {}'.format(args.warmup))

    for i, (input, target, s_input, s_mask, subcls) in enumerate(train_loader):
        data_time.update(time.time() - end)
        current_iter = epoch * len(train_loader) + i + 1

        poly_learning_rate(optimizer, args.base_lr, current_iter, max_iter, power=args.power,
                           index_split=args.index_split, warmup=args.warmup, warmup_step=len(train_loader) // 2)

        s_input = s_input.cuda(non_blocking=True)
        s_mask = s_mask.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with autocast(device_type="cuda"):
            output, main_loss, aux_loss1, aux_loss2 = model(s_x=s_input, s_y=s_mask, x=input, y_m=target, cat_idx=subcls)
            # loss = main_loss + args.aux_weight1 * aux_loss1 + args.aux_weight2 * aux_loss2
            loss = main_loss
        optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        n = input.size(0)  # batch_size

        # output = torch.sigmoid(output)
        # output[output >= 0.5] = 1
        # output[output < 0.5] = 0

        # intersection, union, target = intersectionAndUnionGPU(output, target, args.classes, args.ignore_label)
        # intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        # intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        # accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)  # allAcc

        main_loss_meter.update(main_loss.item(), n)
        # aux_loss_meter1.update(aux_loss1.item(), n)
        # aux_loss_meter2.update(aux_loss2.item(), n)
        loss_meter.update(loss.item(), n)

        batch_time.update(time.time() - end - val_time)
        end = time.time()

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,))
            if args.viz:
                writer.add_scalar('loss_train', loss_meter.val, current_iter)
                writer.add_scalar('loss_train_main', main_loss_meter.val, current_iter)
                writer.add_scalar('loss_train_aux1', aux_loss_meter1.val, current_iter)
                writer.add_scalar('loss_train_aux2', aux_loss_meter2.val, current_iter)

        # -----------------------  SubEpoch VAL  -----------------------
        if args.evaluate and args.SubEpoch_val and ((args.epochs <= 100 and epoch % 1 == 0) or (epoch > 100)) and (i == round(len(train_loader) / 2)):  # <if> max_epoch<=100 <do> half_epoch Val
            mIoU, mDice, mFBIoU = validate(val_loader, model)
            val_num += 1
            # save model for <testing>
            if mDice > best_dice:
                best_miou, best_dice, best_FBiou, best_epoch = mIoU, mDice, mFBIoU, (epoch - 0.5)
                keep_epoch = 0
                if args.shot == 1:
                    filename = args.snapshot_path + '/train_epoch_' + str(epoch - 0.5) + '_{:.4f}'.format(
                        best_miou) + '.pth'
                else:
                    filename = args.snapshot_path + '/train{}_epoch_'.format(args.shot) + str(
                        epoch - 0.5) + '_{:.4f}'.format(best_miou) + '.pth'
                if main_process():
                    logger.info('Saving checkpoint to: ' + filename)
                    torch.save(
                        {'epoch': epoch - 0.5, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                        filename)

            model.train()
            if args.fix_bn:
                model.apply(fix_bn)  # fix batchnorm

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    # allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    allAcc = 0

    # if main_process():
    #     logger.info(
    #         'Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch, args.epochs, mIoU,
    #                                                                                        mAcc, allAcc))
    #     for i in range(args.classes):
    #         logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))

    return main_loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, warmup=False):
    if main_process() and not warmup:
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
        test_num = 1000
        split_gap = 20
    else:
        test_num = len(val_loader)
        split_gap = 2

    class_intersection_meter = [0] * split_gap
    class_union_meter = [0] * split_gap
    
    score_per_class = {}

    if args.manual_seed is not None and args.fix_random_seed_val:
        setup_seed(args.manual_seed, args.seed_deterministic)

    pos_weight = torch.ones([1]).cuda() * 2
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.eval()
    end = time.time()
    val_start = end

    assert test_num % args.batch_size_val == 0
    db_epoch = math.ceil(test_num / (len(val_loader) - args.batch_size_val))
    iter_num = 0

    # ========================================
    # Saving priors
    # ========================================
    prior_dir = "priors/"

    dataset = args.data_set
    prior_dir = "{}{}/".format(prior_dir, dataset)  # priors/pascal/

    sam2_type = args.sam2_type
    prior_dir = "{}{}/".format(prior_dir, sam2_type)  # priors/pascal/small/
    
    ver_dino = args.ver_dino
    prior_dir = "{}{}/".format(prior_dir, ver_dino)  # priors/pascal/small/dinov2_vitb14/

    split = args.split  # fold
    prior_dir = "{}split{}/".format(prior_dir, split)  # priors/pascal/small/dinov2_vitb14/split0/

    shot = args.shot  # shot
    prior_dir = "{}{}shot/".format(prior_dir, shot)  # priors/pascal/small/dinov2_vitb14/split0/1shot

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

            # load prior masks
            prior_path = "{}{}.npy".format(prior_dir, iter_num)
            if os.path.exists(prior_path) and not warmup:
                priors = np.load(prior_path)
                priors = torch.from_numpy(priors)
                priors = priors.cuda(non_blocking=True)
            else:
                priors = None

            start_time = time.time()
            with torch.no_grad():
                with autocast(device_type="cuda"):
                    output, priors = model(s_x=s_input, s_y=s_mask, x=input, y_m=target, cat_idx=subcls, priors=priors)
                    model_time.update(time.time() - start_time)

                    if args.ori_resize:
                        output = F.interpolate(output.unsqueeze(0), size=ori_label.size()[-2:], mode='bilinear', align_corners=True)
                        output = output.squeeze(0)
                        target = ori_label.long()

                    output = F.interpolate(output.unsqueeze(0), size=target.size()[1:], mode='bilinear', align_corners=True)
                    output = output.squeeze(0)
                    label = target.clone()
                    label[label == 255] = 0
                    loss = criterion(output, label.float())

            # save prior masks
            if main_process():
                if not os.path.exists(prior_path):
                    priors = priors.detach().cpu().numpy()
                    np.save(prior_path, priors)

            # output = torch.sigmoid(output)
            # output[output >= 0.5] = 1
            # output[output < 0.5] = 0
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
            loss_meter.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if ((i + 1) % round((test_num / 20)) == 0) and main_process():
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}).'.format(iter_num * args.batch_size_val, test_num,
                                                              data_time=data_time,
                                                              batch_time=batch_time,
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

        if main_process():
            print(tabulate(table_data, headers=["name", "iou", "dice", "fb_iou"], floatfmt=".4f", tablefmt="grid"))

        return avg["iou"], avg["dice"], avg["fb_iou"]


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args = get_parser()

    if args.distributed:
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
    else:
        main()
