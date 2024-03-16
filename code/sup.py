import copy
import os
import random
from datetime import datetime

import torch.nn.functional as F

from dataset.chd import  RaidiumLabeled, get_split_raidium, raidius_sup_collate


from myconfig import get_config

from network.dynamic_graph_unet2d import GraphUnetV5
from torch.utils.tensorboard import SummaryWriter
from utils import *
import segmentation_models_pytorch as smp
from loss.loss_perm_inv import GlobalLoss
from train import Trainer
from models import UNet
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
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

def create_directories(args):
    # initialize config
    
    args.experiment_name = args.experiment_name + '_' + args.model_name + f"bs-{args.batch_size}_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.checkpoints_dir = os.path.join(args.checkpoints_dir, args.experiment_name)
    args.logs_dir = os.path.join(args.logs_dir, args.experiment_name)
    args.tensorboard_dir = os.path.join(args.logs_dir, "tensorboard")
    args.save_img_dir = os.path.join(args.logs_dir, "image")
    
    for dir in [args.checkpoints_dir, args.tensorboard_dir, args.save_img_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

if __name__ == '__main__':
    args = get_config()
    create_directories(args)
    writer = SummaryWriter(args.tensorboard_dir)

    for i in range(0, args.cross_vali_num):
        if i == args.fold:
    
            # setup cuda
            args.device = torch.device(
                args.device if torch.cuda.is_available() else "cpu")
            print(f"The model will run on device:{args.device}")
            torch.manual_seed(args.seed)
            if 'cuda' in str(args.device):
                torch.cuda.manual_seed_all(args.seed)

            # Create model
            print("Importing model ...")
            # model = UNet2D(in_channels=1, initial_filter_size=args.initial_filter_size, kernel_size=3, classes=args.classes, do_instancenorm=True)
            model_kwargs = get_kwargs_model(args)
            # model = GraphUnetV5(in_channels=1,
            #                     initial_filter_size=args.initial_filter_size,
            #                     kernel_size=3, classes=args.classes,
            #                     do_instancenorm=True, **model_kwargs)
            
            if args.model_name == "UNet":
                model = UNet(1, args.classes)
            elif args.model_name == "TransUNet":
                config_vit = CONFIGS_ViT_seg[args.vit_name]
                config_vit.n_classes = args.num_classes
                config_vit.n_skip = args.n_skip
                if args.vit_name.find('R50') != -1:
                    config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
                model = ViT_seg(config_vit, img_size=args.img_size, num_classes=args.classes).cuda()
                model.load_from(weights=np.load(config_vit.pretraianed_path))
            print("Model {model_name} imported".format(model_name=args.model_name))

            if args.restart:
                print('Loading from saved model ' + args.pretrained_model_path)
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

            model.to(args.device)
            num_parameters = sum([l.nelement() for l in model.parameters()])
            print(f"Number of parameters: {num_parameters}")

            if args.dataset == "raidium":
                train_files, val_files = get_split_raidium(args.data_dir, args.fold, args.cross_vali_num)
                if args.enable_few_data:
                    random.seed(args.seed)
                    train_files = random.sample(list(train_files), k=args.sampling_k)
                print(f'Number train files :{len(train_files)}')
                print(f'Number val files :{len(val_files)}')
                train_dataset = RaidiumLabeled(files=train_files, purpose='train', args=args)
                validate_dataset = RaidiumLabeled(files=val_files, purpose='val', args=args)
                

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
            
            print("Starting training ...")
            trainer = Trainer(
                args.fold,
                writer,
                train_loader,
                val_loader,
                model,
                criterion,
                metric,
                args
                )
            
