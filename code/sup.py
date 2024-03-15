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
from loss.loss_perm_inv import GlobalLoss
from train import Trainer
from meters import RandScore


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
            logger.print(f"Number of parameters: {num_parameters}")

            if args.dataset == "raidium":
                train_files, val_files = get_split_raidium(args.data_dir, args.fold, args.cross_vali_num)
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
            val_loader = torch.utils.data.DataLoader(
                validate_dataset, batch_size=args.batch_size, shuffle=False,
                # num_workers=args.num_works, 
                drop_last=False,
                collate_fn=raidius_sup_collate
            )

            criterion = GlobalLoss().to(args.device)
            metric = RandScore()

            trainer = Trainer(
                args.fold,
                writer,
                train_loader,
                val_loader,
                model,
                criterion,
                metric,
                args,
                logger)
            