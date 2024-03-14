#!/bin/bash

#cd /mnt/c/Users/matte/iCloudDrive/Documents/Studies/IASD/College-De-France/Challenge
cd ~/ENS-challenge-24
sample_size=${1:-5}
fold=${2:-1}
export num_layer=8

CUDA_VISIBLE_DEVICES=0                                                      \
python                                                                      \
code/sup.py                                                                      \
--batch_size             1                                                 \
--classes                105                                                  \
--data_dir               ./data/supervised                                  \
--dataset                raidium                                                \
--device                 cuda:0                                             \
--enable_few_data                                                           \
--epochs                 101                                                \
--experiment_name        sup_raidium_sample_$sample_size\_fold_$fold\_      \
--fold                   $fold                                              \
--initial_filter_size    32                                                 \
--lr                     0.05                                               \
--lr                     5e-5                                               \
--min_lr                 5e-6                                               \
--num_works              1                                                 \
--patch_size             512 512                                            \
--results_dir            ./results/sup                                      \
--runs_dir               ./runs/sup                                         \
--sampling_k             $sample_size                                       \
--save                   all_graph_features
