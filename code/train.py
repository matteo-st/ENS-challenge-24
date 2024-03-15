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
            train_loss = self.train_epoch()
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
            self.dict_record['epoch'][epoch] = {
                    'train_loss': train_loss, 
                    'val_loss ': val_loss,
                    'Metric': self.metric.summary()
                }
            if epoch % self.num_epoch_record == 0:
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
                label = rearrange(label, 'b h w -> b (h w)')
                self.metric.add_val(seg.detach().cpu(), label.detach().cpu())

                seg = self.meanshift(label_logits)
                seg = rearrange(seg, 'b h w -> b (h w)')
                #seg = (seg - seg.min(dim=-1)) / (seg.max(dim=-1) - seg.min(dim=-1)) * 255
                label = rearrange(label, 'b h w -> b (h w)')
                self.metric.add_val(seg.detach().cpu(), label.detach().cpu())

            return loss 



    def train_epoch(self, epoch):
        self.model.train()
        self.metric.reset_train()

        for batch_idx, tup in enumerate(self.train_loader):
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
                label = rearrange(label, 'b h w -> b (h w)')
                self.metric.add_train(seg.detach().cpu(), label.detach().cpu())


        return loss




        