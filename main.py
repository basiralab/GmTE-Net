#!/usr/bin/python
# -*- coding: utf8 -*-
"""
Main function of GmTE-Net framework
for jointly predicting multiple trajectories from a single input graph at baseline.
Details can be found in:
(1) the original paper https://link.springer.com/
    Alaa Bessadok, Mohamed Ali Mahjoub, and Islem Rekik. "A Few-shot Learning Graph Multi-Trajectory Evolution Network
     for Forecasting Multimodal Baby Connectivity Development from a Baseline Timepoint", MICCAI 2021, Strasbourg, France.
(2) the youtube channel of BASIRA Lab:
---------------------------------------------------------------------
This file contains the implementation of the main step of our GmTE-Net framework:
  (1) brain graph multi-trajectory evolution prediction => step1: train the teacher network and,
     step2: test the teacher network on the augmented graphs and start training the student network using the knowledge of the teacher.

  GmTE-Net(F_4_t0_T, A_100_t0_S, teacher_M_tn_loaders, teacher_F_tn_loaders, student_M_tn_loaders, student_F_tn_loaders,
                 nb_timepoints, opts, all_times_loaders_M, all_times_loaders_F)
          Inputs:
                  F_4_T:   represents the real few-shot data acquired at t0 which are the input to the teacher network
                            ==> it is a PyTorch dataloader returning elements from source dataset batch by batch
                  A_100_S:  represents the augmented data representing the timepoint t0 which are the input to the student network
                            ==> it is a PyTorch dataloader returning elements from target dataset batch by batch
                  teacher_M_tn_loaders, student_M_tn_loaders:  each is a PyTorch dataloader representing the modality 'M' (i.e., low-resolution)
                            acquired at multiple timepoints and given as input to the teacher and the student models, respectively.
                  teacher_F_tn_loaders, student_F_tn_loaders:  two PyTorch dataloaders representing the modality 'F' (i.e., super-resolution)
                            acquired at multiple timepoints and given as input to the teacher and the student models, respectively.
                  nb_timepoints: is the number of timepoints for each trajectory.
                  opts:         a python object (parser) storing all arguments needed to run the code such as hyper-parameters
                  all_times_loaders_LR, all_times_loaders_SR: these parameters are only given as input to the model in the testing stage,
                                                            during the training we initialize them to zero.
          Output:
                  model:        our GmTE-Net model

Sample use for training:
  model = GmTE-Net(F_4_t0_T, A_100_t0_S, teacher_M_tn_loaders, teacher_F_tn_loaders, student_M_tn_loaders, student_F_tn_loaders,
                 nb_timepoints, opts, all_times_loaders_LR, all_times_loaders_SR)
  model.train()

Sample use for testing:
  model = GmTE-Net(LR_11_t0_T, 0, 0, 0, 0, 0, nb_timepoints, opts, all_times_loaders_LR, all_times_loaders_SR)
  predicted_Trajectory_LR_from_teacher, predicted_Trajectory_SR_from_teacher, predicted_Trajectory_LR_from_student, predicted_Trajectory_SR_from_student = model.test()
          Inputs:
                  LR_11_t0_T:   represents the testing graphs acquired at t0 which are the input to  both teacher and student network
                            ==> it is a PyTorch dataloader returning elements from source dataset batch by batch
          Output:
                  predicted_Trajectory_LR_from_teacher, predicted_Trajectory_LR_from_student: each is a list of size nb_timepoints-1 where nb_timepoints is the number of timepoints of the LR trajectory.
                                           Each element is an (n_s × n_d) matrix stacking the feature matrix V denoting the low-resolution predicted from the teacher and the student, respectively.
                                           n_s is the number of testing subjects and n_d is the dimension of each feature vector stacked in the V matrix.
                  predicted_Trajectory_SR_from_teacher, predicted_Trajectory_SR_from_student: each is a list of size nb_timepoints where nb_timepoints is the number of timepoints of the LR trajectory.
                                           Each element is an (n_s × n_d) matrix stacking the feature matrix V denoting the super-resolution predicted from the teacher and the student, respectively.

---------------------------------------------------------------------
Copyright 2021 Alaa Bessadok, Istanbul Technical University.
Please cite the above paper if you use this code.
All rights reserved.
"""
import argparse
import random
import yaml
import numpy as np
from torch.backends import cudnn
from prediction1b import GmTE_Net
from data_loader import *

parser = argparse.ArgumentParser()
# initialisation
# Basic opts.
parser.add_argument('--nb_timepoints', type=int, default=4, help='how many timepoint we have in a trajectory')
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--log_dir', type=str, default='logs/')
parser.add_argument('--checkpoint_dir', type=str, default='models/')
parser.add_argument('--result_dir', type=str, default='results/')
parser.add_argument('--result_root', type=str, default='result')

# GCN model opts
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--hidden1', type=int, default=100)
parser.add_argument('--hidden2', type=int, default=50)
parser.add_argument('--hidden3', type=int, default=16)
parser.add_argument('--LRout', type=int, default=595)
parser.add_argument('--SRout', type=int, default=6670)

# Training opts.
parser.add_argument('--t_lr', type=float, default=0.0001, help='learning rate for teacher')
parser.add_argument('--s_lr', type=float, default=0.0001, help='learning rate for student')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
parser.add_argument('--num_workers', type=int, default=0, help='num_workers to load data.')
parser.add_argument('--num_iters', type=int, default=30, help='number of total iterations for training')
parser.add_argument('--log_step', type=int, default=30)
parser.add_argument('--model_save_step', type=int, default=30)
# Test opts.
parser.add_argument('--test_iters', type=int, default=30, help='test model from this step')

opts = parser.parse_args()
opts.log_dir = os.path.join(opts.result_root, opts.log_dir)
opts.checkpoint_dir = os.path.join(opts.result_root, opts.checkpoint_dir)
opts.result_dir = os.path.join(opts.result_root, opts.result_dir)

if __name__ == '__main__':

    # For fast training.
    cudnn.benchmark = True

    if opts.mode == 'train':
        """
        Training TS
        """
        # Create directories if not exist.
        create_dirs_if_not_exist([opts.log_dir, opts.checkpoint_dir, opts.result_dir])

        # log opts.
        with open(os.path.join(opts.result_root, 'opts.yaml'), 'w') as f:
            f.write(yaml.dump(vars(opts)))

            input_timepoint = 0
            #----READ MODALITY 1 AT T0 => REAL and AUGMENTED
            real_t0_encoder = np.random.normal(random.random(), random.random(), (4,595))
            augmented_t0_encoder = np.random.normal(random.random(), random.random(), (100,595))
            F_4_t0_T = get_loader(real_t0_encoder, real_t0_encoder.shape[0], "train", opts.num_workers)
            A_100_t0_S = get_loader(augmented_t0_encoder, augmented_t0_encoder.shape[0], "train", opts.num_workers)

            #----READ MULTI-TRAJECTORY DATA FROM T1 to TN => REAL and AUGMENTED
            teacher_M_tn_loaders = []
            student_M_tn_loaders = []
            teacher_F_tn_loaders = []
            student_F_tn_loaders = []
            for timepoint in range(0, opts.nb_timepoints):
                # here you need to upload the real data at each timepoint
                real_input_M_tn_encoder = np.random.normal(random.random(), random.random(), (4,595))
                real_input_F_tn_encoder = np.random.normal(random.random(), random.random(), (4,6670))
                # here you need to upload the augmented data at each timepoint
                augmented_input_M_tn_encoder = np.random.normal(random.random(), random.random(), (100,595))
                augmented_input_F_tn_encoder = np.random.normal(random.random(), random.random(), (100,6670))

                input_M_tn_real_data_teacher_loader = get_loader(real_input_M_tn_encoder, real_input_M_tn_encoder.shape[0], "train", opts.num_workers)
                input_F_tn_real_data_teacher_loader = get_loader(real_input_F_tn_encoder, real_input_F_tn_encoder.shape[0], "train", opts.num_workers)
                input_M_tn_augmented_data_student_loader = get_loader(augmented_input_M_tn_encoder, augmented_input_M_tn_encoder.shape[0], "train", opts.num_workers)
                input_F_tn_augmented_data_student_loader = get_loader(augmented_input_F_tn_encoder, augmented_input_F_tn_encoder.shape[0], "train", opts.num_workers)

                teacher_M_tn_loaders.append(input_M_tn_real_data_teacher_loader)
                teacher_F_tn_loaders.append(input_F_tn_real_data_teacher_loader)
                student_M_tn_loaders.append(input_M_tn_augmented_data_student_loader)
                student_F_tn_loaders.append(input_F_tn_augmented_data_student_loader)

            model = GmTE_Net(F_4_t0_T, A_100_t0_S, teacher_M_tn_loaders, teacher_F_tn_loaders,
                                     student_M_tn_loaders, student_F_tn_loaders, opts.nb_timepoints,
                                     opts, 0, 0)
            model.train()
    elif opts.mode == 'test':
        """
        Testing GmTE-Net
        """
        # Create directories if not exist.
        create_dirs_if_not_exist([opts.result_dir])

        input_timepoint = 0
        #--- READ MULTI-TRAJECTORY DATA FROM T0 to TN => REAL and AUGMENTED
        real_11_morpho_t0 = np.random.normal(random.random(), random.random(), (11,595))
        input_t0_loader = get_loader(real_11_morpho_t0, real_11_morpho_t0.shape[0], "test", opts.num_workers)

        all_times_loaders_LR = []
        for timepoint in range(0, opts.nb_timepoints):
            real_11_morpho_tn = np.random.normal(random.random(), random.random(), (11,595))
            tn_input = get_loader(real_11_morpho_tn, real_11_morpho_tn.shape[0], "test", opts.num_workers)
            all_times_loaders_LR.append(tn_input)

        all_times_loaders_SR = []
        for timepoint in range(0, opts.nb_timepoints):
            real_11_func_tn = np.random.normal(random.random(), random.random(), (11,595))
            tn_input = get_loader(real_11_func_tn, real_11_func_tn.shape[0], "test", opts.num_workers)
            all_times_loaders_SR.append(tn_input)

        model = GmTE_Net(input_t0_loader, 0, 0, 0, 0, 0, opts.nb_timepoints, opts,
                                 all_times_loaders_LR, all_times_loaders_SR)

        predicted_Trajectory_LR_from_teacher, predicted_Trajectory_SR_from_teacher, \
        predicted_Trajectory_LR_from_student, predicted_Trajectory_SR_from_student = model.test()
