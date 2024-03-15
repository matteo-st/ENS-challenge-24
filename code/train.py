import json 
import os.path
from itertools import chain
import random 
import torch
from torch import optim
from einops import rearrange
import pandas as pd
import numpy as np
from lr_scheduler import LR_Scheduler
from meanshift import MeanShiftCluster
from tqdm import tqdm 
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self, 
                 fold, 
                 writer,
                 train_loader,
                 val_loader,
                 model,
                 criterion,
                 metric,
                 args,
                 logger
                 ):
        self.args = args
        self.model = model
        self.metric = metric
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = writer
        self.fold = fold
        self.logger = logger
        self.dict_record = {'epoch' : {}}
        
        self.meanshift = MeanShiftCluster()
        self.init_optim()
        self.training()


    def init_optim(self):
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, 
                   self.model.parameters()), 
                   lr=self.args.lr, weight_decay=1e-5)
        self.scheduler = LR_Scheduler(self.args.lr_scheduler, self.args.lr,
                                      self.args.epochs, len(self.train_loader), 
                                      min_lr=self.args.min_lr)
        
    def training(self):

        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.val_epoch()

            self.writer.add_scalar('training_loss_fold'+str(self.fold), 
                                   train_loss, epoch)
            # self.writer.add_scalar('training_dice_fold'+str(self.fold), train_metric, epoch)
            self.writer.add_scalar('learning_rate_fold'+str(self.fold),
                                   self.optimizer.param_groups[0]['lr'], epoch)
            # self.writer.add_scalar('validate_d_fold'+str(self.fold), val_metric, epoch)
            self.writer.add_scalar('val_loss_fold'+str(self.fold), val_loss, epoch)
            
            self.logger.print('Epoch: {0}\t'
                         'Training Loss {train_loss:.4f} \t'
                         'Val Loss {val_loss:.4f} \t'
                         'Validation Metric  \t'
                         .format(epoch, train_loss=train_loss, 
                                 val_loss=val_loss
                                #  val_dice=val_metric
                                 ))
            
            self.logger.print(self.metric.summary())
            self.dict_record['epoch'][epoch] = {
                    'train_loss': train_loss, 
                    'val_loss ': val_loss,
                    'Metric': self.metric.summary()
                }
            if epoch % self.args.num_epoch_record == 0:
                torch.save({
                    'epoch' : epoch,
                    'model_state_dic' : self.model.state_dict(),
                    "optimizer_state_dict" : self.optimizer.state_dict(),
                    "stats" : self.dict_record["epoch"][epoch]
                    },
                    os.path.join(self.args.model_result_dir, "epoch-{}.pth".format(epoch)))
                # save_dict = {"net": self.model.state_dict()}
                # torch.save(save_dict, os.path.join(
                #     self.args.model_result_dir, "best.pth"))


            # if best_metric < val_metric:
            #     best_metric = val_metric
                

            # # save model
            # save_dict = {"net": self.model.state_dict()}
            # torch.save(save_dict, os.path.join(
            #     self.args.model_result_dir, "latest.pth"))

    def val_epoch(self):
        self.model.eval()
        self.metric.reset_val()
        with torch.no_grad():
            for batch_idx, tup in enumerate(self.val_loader):
                img, label, keypoints = tup
                image_var = img.float().to(self.args.device)
                label = label.float().to(self.args.device).unsqueeze(1)
                keypoints = keypoints.float().to(self.args.device)
                label_logits, _ = self.model(image_var, keypoints)
                loss = self.criterion(label_logits, label)
                seg = self.meanshift(label_logits)

                seg = rearrange(seg, 'b h w -> b (h w)')
                #seg = (seg - seg.min(dim=-1)) / (seg.max(dim=-1) - seg.min(dim=-1)) * 255
                label = rearrange(label.squeeze(1), 'b h w -> b (h w)')
                self.metric.add_val(seg.detach().cpu(), label.detach().cpu())

        self.visualize_seg(type="val")    
        return loss 


    def visualize_seg(self, epoch=0, type="train"):
        if type == "train":
            data_loader = self.train_loader
        elif type == "val":
            data_loader = self.val_loader

        with torch.no_grad():
            img, label, keypoints = next(iter(data_loader))  # Example: get first batch
            img = img.float().to(self.args.device)
            label = label.float().to(self.args.device)
            label_logits, _ = self.model(img, keypoints)
            keypoints = keypoints.float().to(self.args.device)
            # Assume meanshift or similar method is used to get seg from label_logits
            seg = self.meanshift(label_logits).cpu().numpy()
            
            # 
            fig, axes = plt.subplots(1, 3)
            axes[0].imshow(img, cmap="gray")
            axes[1].imshow(img, cmap="gray")
            axes[2].imshow(img, cmap="gray")
            seg_masked = np.ma.masked_where(seg.reshape((512,512)) == 0, (seg.reshape((512,512))))
            label_masked = np.ma.masked_where(label.reshape((512,512)) == 0, (label.reshape((512,512))))
            axes[1].imshow(seg_masked, cmap="tab20")
            axes[2].imshow(seg_masked, cmap="tab20")
            plt.axis("off")

              # Make a grid with 3 images in a row
            # # Visualize the first sample of the batch
            # self.plot_seg(img[0].cpu().numpy(), label[0].cpu().numpy(), seg[0], epoch, "train")

            # os.makedirs(save_dir, exist_ok=True)
    
            # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            # axes[0].imshow(image.squeeze(), cmap="gray")
            # axes[0].set_title("Image")
            # axes[1].imshow(label.squeeze(), cmap="jet", alpha=0.5)
            # axes[1].set_title("Ground Truth")
            # axes[2].imshow(seg.squeeze(), cmap="jet", alpha=0.5)
            # axes[2].set_title("Segmentation Prediction")
            # for ax in axes:
            #     ax.axis("off")
            # plt.tight_layout()
            
            # Save the figure
            # fig.savefig(os.path.join(save_dir, f"{prefix}_epoch_{epoch}.png"))
            # plt.close(fig)




    def train_epoch(self, epoch):
        self.model.train()
        self.metric.reset_train()

        for batch_idx, tup in tqdm(enumerate(self.train_loader)):
            img, label, keypoints = tup
            image_var = img.float().to(self.args.device)
            label = label.float().to(self.args.device).unsqueeze(1)
            keypoints = keypoints.float().to(self.args.device)
            self.scheduler(self.optimizer, batch_idx, epoch)
            label_logits, _ = self.model(image_var, keypoints)
            loss = self.criterion(label_logits, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                seg = self.meanshift(label_logits)
                seg = rearrange(seg, 'b h w -> b (h w)')
                #seg = (seg - seg.min(dim=-1)) / (seg.max(dim=-1) - seg.min(dim=-1)) * 255
                label = rearrange(label.squeeze(1), 'b h w -> b (h w)')
                self.metric.add_train(seg.detach().cpu(), label.detach().cpu())

        self.visualize_seg(type="train")    
        return loss




        
