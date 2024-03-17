#!/bin/bash

cd ~/ENS-challenge-24
#sample_size=${1:-100}
fold=${2:-1}
export num_layer=8

CUDA_VISIBLE_DEVICES=0                                                      \
python                                                                      \
code/sup.py                                                                 \
--batch_size             4                                                  \
--classes                1                                                  \
--num_epoch_record       2                                                  \
--data_dir               ./data/supervised                                  \
--dataset                raidium                                            \
--device                 cuda:0                                             \
--epochs                 100                                                 \
--fold                   $fold                                              \
--initial_filter_size    32                                                 \
--lr                     5e-5                                               \
--min_lr                 5e-6                                               \
--num_works              1                                                  \
--patch_size             512 512                                            \
--img_size               512                                                \
--checkpoints_dir        ./checkpoints/sup                                  \
--logs_dir               ./logs/sup                                         \
--n_skip                 3                                                  \
--vit_name               ViT-B_16                                           \
--vit_patches_size       16                                                 \
--n_skip                 0                                                  \
--model_name             TransUNet                                          \
#--sampling_k             $sample_size                                       \
