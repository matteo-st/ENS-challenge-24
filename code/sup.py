import copy
import os
import random
from datetime import datetime

import torch.nn.functional as F
from loss.dice_loss import DiceLoss
# from dataset.acdc_graph import ACDC
from dataset.chd import  RAIDIUM, get_split_raidium, raidius_sup_collate
# from dataset.raidium import RAIDIUM, raidius_sup_collate, get_split_raidium
# from dataset.hvsmr import HVSMR
# from dataset.mmwhs import MMWHS
from experiment_log import PytorchExperimentLogger
from lr_scheduler import LR_Scheduler
from metrics import SegmentationMetric
from myconfig import get_config
# from network.unet2d import UNet2D
from network.dynamic_graph_unet2d import GraphUnetV5
from torch.utils.tensorboard import SummaryWriter
from utils import *
import segmentation_models_pytorch as smp


import sys
# project_path = "/mnt/c/Users/matte/iCloudDrive/Documents/Studies/IASD/College-De-France/Challenge"
# sys.path.append(project_path + '/code/network')
# # sys.path.append('/workspace/src/graph_chd_sup_1/')
# sys.path.append(project_path + '/code/network/gcn_lib')


def get_kwargs_model(args):
    model_kwargs = vars(copy.deepcopy(args))
    model_kwargs.pop('initial_filter_size')
    model_kwargs.pop('classes')
    return model_kwargs


def run(fold, writer, args):
    maybe_mkdir_p(os.path.join(args.save_path, 'cross_val_'+str(fold)))
    logger = PytorchExperimentLogger(os.path.join(
        args.save_path, 'cross_val_'+str(fold)), "elog", ShowTerminal=True)
    # setup cuda
    args.device = torch.device(
        args.device if torch.cuda.is_available() else "cpu")
    logger.print(f"the model will run on device:{args.device}")
    torch.manual_seed(args.seed)
    if 'cuda' in str(args.device):
        torch.cuda.manual_seed_all(args.seed)
    logger.print(f"starting training for cross validation fold {fold} ...")
    model_result_dir = join(args.save_path, 'cross_val_'+str(fold), 'model')
    maybe_mkdir_p(model_result_dir)
    args.model_result_dir = model_result_dir
    # create model
    logger.print("creating model ...")
    # model = UNet2D(in_channels=1, initial_filter_size=args.initial_filter_size, kernel_size=3, classes=args.classes, do_instancenorm=True)
    model_kwargs = get_kwargs_model(args)
    model = GraphUnetV5(in_channels=1,
                        initial_filter_size=args.initial_filter_size,
                        kernel_size=3, classes=args.classes,
                        do_instancenorm=True, **model_kwargs)
    if args.restart:
        logger.print('loading from saved model ' + args.pretrained_model_path)
        dict = torch.load(args.pretrained_model_path,
                          map_location=lambda storage, loc: storage)
        save_model = dict["net"]
        model_dict = model.state_dict()
        # we only need to load the parameters of the encoder
        state_dict = {
            k: v
            for k, v in save_model.items()
            if k.startswith('encoder')
        }
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

        # load superglue
        # dict = torch.load('/workspace/src/gcl_1/results/contrast_chd_pcl_2023-04-22_21-16-16/model/latest.pth',
        #                   map_location=lambda storage, loc: storage)
        # save_model = dict["net"]
        # model_dict = model.state_dict()
        # # we only need to load the parameters of the encoder
        # state_dict = {
        #     k: v
        #     for k, v in save_model.items()
        #     if 'superglue' in k
        # }
        # model_dict.update(state_dict)
        # model.load_state_dict(model_dict)

    model.to(args.device)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logger.print(f"number of parameters: {num_parameters}")

    if args.dataset == 'chd':
        train_keys, val_keys = get_split_chd(os.path.join(
            args.data_dir, 'train'), fold, args.cross_vali_num)
        # now random sample train_keys
        if args.enable_few_data:
            random.seed(args.seed)
            train_keys = random.sample(list(train_keys), k=args.sampling_k)
        logger.print(f'train_keys:{train_keys}')
        logger.print(f'val_keys:{val_keys}')
        # train_dataset = CHD(keys=train_keys, purpose='train', args=args)
        # validate_dataset = CHD(keys=val_keys, purpose='val', args=args)
    elif args.dataset == 'mmwhs':
        train_keys, val_keys = get_split_mmwhs(fold, args.cross_vali_num)
        if args.enable_few_data:
            random.seed(args.seed)
            train_keys = random.sample(list(train_keys), k=args.sampling_k)
        logger.print(f'train_keys:{train_keys}')
        # train_dataset = MMWHS(keys=train_keys, purpose='val', args=args)
        # logger.print('training data dir '+train_dataset.data_dir)
        # validate_dataset = MMWHS(keys=val_keys, purpose='val', args=args)
    elif args.dataset == 'acdc':
        train_keys, val_keys = get_split_acdc(fold, args.cross_vali_num)
        if args.enable_few_data:
            random.seed(args.seed)
            train_keys = random.sample(list(train_keys), k=args.sampling_k)
        logger.print(f'train_keys:{train_keys}')
        logger.print(f'val_keys:{val_keys}')
        # train_dataset = ACDC(keys=train_keys, purpose='train', args=args)
        # validate_dataset = ACDC(keys=val_keys, purpose='val', args=args)
    elif args.dataset == 'hvsmr':
        train_keys, val_keys = get_split_hvsmr(fold, args.cross_vali_num)
        if args.enable_few_data:
            random.seed(args.seed)
            train_keys = random.sample(list(train_keys), k=args.sampling_k)
        logger.print(f'train_keys:{train_keys}')
        logger.print(f'val_keys:{val_keys}')
        # train_dataset = HVSMR(keys=train_keys, purpose='train', args=args)
        # validate_dataset = HVSMR(keys=val_keys, purpose='val', args=args)
    elif args.dataset == "raidium":
        train_files, val_files = get_split_raidium(args.data_dir, fold, args.cross_vali_num)
        if args.enable_few_data:
            random.seed(args.seed)
            train_files = random.sample(list(train_files), k=args.sampling_k)
        logger.print(f'Number train files :{len(train_files)}')
        logger.print(f'Number val files :{len(val_files)}')
        train_dataset = RAIDIUM(files=train_files, purpose='train', args=args)
        validate_dataset = RAIDIUM(files=val_files, purpose='val', args=args)
        

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        # num_workers=args.num_works,
          drop_last=False,
        collate_fn=raidius_sup_collate
    )
    validate_loader = torch.utils.data.DataLoader(
        validate_dataset, batch_size=args.batch_size, shuffle=False,
        # num_workers=args.num_works, 
        drop_last=False,
        collate_fn=raidius_sup_collate
    )

    # criterion = torch.nn.CrossEntropyLoss()
    criterion = DiceLoss(activation="softmax", ignore_background=False)
    
    optimizer = torch.optim.Adam(filter(
        lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5
    )
    '''finetuen option'''
    # params_to_optimize = list(model.superglue.parameters()) + \
    #     list(model.feature_mlp.parameters()) + \
    #     list(model.diffusion_cnn.parameters()) + \
    #     list(model.decoder.expand_1_1.parameters()) + \
    #     list(model.decoder.expand_1_2.parameters()) + \
    #     list(model.decoder.head.parameters()) + \
    #     list(model.head.parameters())
    # optimizer = torch.optim.Adam(
    #     params_to_optimize,
    #     lr=args.lr, weight_decay=1e-5
    # )
    scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                             args.epochs, len(train_loader), min_lr=args.min_lr)
    best_dice = 0
    for epoch in range(args.epochs):
        # val_dice = validate(validate_loader, model, epoch, logger, args)
        # logger.print('Epoch: {0}\t'
        #              'Validation Dice {val_dice:.4f} \t'
        #              .format(epoch, val_dice=val_dice))
        # train for one epoch
        train_loss, train_dice = train(
            train_loader, model, criterion, epoch, optimizer, scheduler,
            logger, args)
        writer.add_scalar('training_loss_fold'+str(fold), train_loss, epoch)
        writer.add_scalar('training_dice_fold'+str(fold), train_dice, epoch)
        writer.add_scalar('learning_rate_fold'+str(fold),
                          optimizer.param_groups[0]['lr'], epoch)

        # val_dice = validate(validate_loader, model, epoch, logger, args)

        if (epoch % 2 == 0):
            # evaluate for one epoch
            val_dice = validate(validate_loader, model, epoch, logger, args)

            logger.print('Epoch: {0}\t'
                         'Training Loss {train_loss:.4f} \t'
                         'Validation Dice {val_dice:.4f} \t'
                         .format(epoch, train_loss=train_loss, val_dice=val_dice))

            if best_dice < val_dice:
                best_dice = val_dice
                save_dict = {"net": model.state_dict()}
                torch.save(save_dict, os.path.join(
                    args.model_result_dir, "best.pth"))
            writer.add_scalar('validate_dice_fold'+str(fold), val_dice, epoch)
            writer.add_scalar('best_dice_fold'+str(fold), best_dice, epoch)
            # save model
            save_dict = {"net": model.state_dict()}
            torch.save(save_dict, os.path.join(
                args.model_result_dir, "latest.pth"))

        # if args.debug_mode:
        #     break

        logger.print(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))


def train(data_loader, model, criterion, epoch, optimizer, scheduler, logger, args, step=5):
    model.train()
    metric_val = SegmentationMetric(args.classes)
    metric_val.reset()
    graph_metric_val = SegmentationMetric(args.classes)
    graph_metric_val.reset()
    losses = AverageMeter()
    for batch_idx, tup in enumerate(data_loader):
        img, label, keypoints = tup
        # img, label, keypoints, keypoints_label, nimg, nkp = tup
        image_var = img.float().to(args.device)
        label = label.float().to(args.device)
        keypoints = keypoints.float().to(args.device)
        # keypoints_label = keypoints_label.long().to(args.device)
        # nimg = nimg.float().to(args.device)
        # nkp = nkp.float().to(args.device)
        scheduler(optimizer, batch_idx, epoch)
        print('image_var', image_var.shape)
        print("label ", label.shape)
        print("keypoints ", keypoints.shape)
        x_out, graph_out = model(image_var, keypoints)
        print('label', label.shape)
        print("model output", x_out.shape)
        loss_1 = criterion(x_out, label)
        # loss_2 = criterion(graph_out, keypoints_label.squeeze(dim=1))
        loss = loss_1
        losses.update(loss_1.item(), image_var.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Do softmax
        x_out = F.softmax(x_out, dim=1)
        # graph_out = F.softmax(graph_out, dim=1)
        metric_val.update(label.long().squeeze(dim=1), x_out)
        # graph_metric_val.update(
        #     keypoints_label.long().squeeze(dim=1), graph_out
        # )
        _, _, Dice = metric_val.get()
        _, _, graph_Dice = graph_metric_val.get()
        print('loss' , loss_1.item())
        logger.print(
            f"Training epoch:{epoch:4d}, batch:{batch_idx:4d}/{len(data_loader)}, lr:{optimizer.param_groups[0]['lr']:.6f}, loss:{losses.avg:.4f}, pixel / graph Dice:{Dice:.4f} / {graph_Dice:.4f}")

        if args.debug_mode:
            break
        # if batch_idx > step:
        #     break

    pixAcc, mIoU, mDice = metric_val.get()

    return losses.avg, mDice


def validate(data_loader, model, epoch, logger, args):
    model.eval()
    metric_val = SegmentationMetric(args.classes)
    metric_val.reset()
    graph_metric_val = SegmentationMetric(args.classes)
    graph_metric_val.reset()
    with torch.no_grad():
        for batch_idx, tup in enumerate(data_loader):
            img, label, keypoints = tup
            image_var = img.float().to(args.device)
            label = label.long().to(args.device)
            keypoints = keypoints.float().to(args.device)
            # keypoints_label = keypoints_label.long().to(args.device)
            # nimg = nimg.float().to(args.device)
            # nkp = nkp.float().to(args.device)

            # x_out, graph_out = model(image_var, keypoints, nimg, nkp)
            x_out, graph_out = model(image_var, keypoints)

            # x_out = F.softmax(x_out, dim=1)
            # graph_out = F.softmax(graph_out, dim=1)
            metric_val.update(label.long().squeeze(dim=1), x_out)
            # graph_metric_val.update(
            #     keypoints_label.long().squeeze(dim=1), graph_out)
            pixAcc, mIoU, Dice = metric_val.get()
            _, _, graph_Dice = graph_metric_val.get()
            logger.print(
                f"Validation epoch:{epoch:4d}, batch:{batch_idx:4d}/{len(data_loader)}, pixel / graph Dice:{Dice:.04f} / {graph_Dice:.04f}")

            if args.debug_mode:
                break

    pixAcc, mIoU, Dice = metric_val.get()
    return Dice


if __name__ == '__main__':
    # initialize config
    args = get_config()
    if args.save == '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.save_path = os.path.join(
        args.results_dir, args.experiment_name + args.save)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    writer = SummaryWriter(os.path.join(
        args.runs_dir, args.experiment_name + args.save))
    for i in range(0, args.cross_vali_num):
        if i == args.fold:
            run(i, writer, args)
