import os
import time
import datetime
import itertools
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
from torch.autograd import grad
from model1b import *
from data_loader import *
from centrality import *
import numpy as np



class GmTE_Net(object):
    """
    Build GmTE-Net model for training and testing
    """

    def __init__(self, F_4_t0_T, A_100_t0_S,
                 teacher_M_tn_loaders, teacher_F_tn_loaders, student_M_tn_loaders, student_F_tn_loaders,
                 nb_timepoints, opts, all_times_loaders_LR, all_times_loaders_SR):

        self.F_4_t0_T = F_4_t0_T
        self.A_100_t0_S = A_100_t0_S

        self.teacher_M_tn_loaders = teacher_M_tn_loaders
        self.teacher_F_tn_loaders = teacher_F_tn_loaders

        self.student_M_tn_loaders = student_M_tn_loaders
        self.student_F_tn_loaders = student_F_tn_loaders

        self.nb_timepoints = nb_timepoints
        self.opts = opts

        self.all_times_loaders_morphological = all_times_loaders_LR
        self.all_times_loaders_functional = all_times_loaders_SR

        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print(self.device)

        # build models
        self.build_model()

    def build_model(self):
        """
        Build teachers and students networks and initialize optimizers.
        """
        self.EncoderT = Encoder(self.opts.LRout, self.opts.hidden1, self.opts.hidden2,
                                self.opts.dropout).to(self.device)
        self.EncoderS = Encoder(self.opts.LRout, self.opts.hidden1, self.opts.hidden2,
                                self.opts.dropout).to(self.device)

        self.Teacher_LR = Teacher_LR(self.opts.hidden2, self.opts.hidden1, self.opts.LRout,
                                     self.opts.dropout, self.nb_timepoints).to(self.device)
        self.Student_LR = Student_LR(self.opts.hidden2, self.opts.hidden1, self.opts.LRout,
                                     self.opts.dropout, self.nb_timepoints).to(self.device)

        self.Teacher_SR = Teacher_SR(self.opts.hidden2, self.opts.hidden1,
                                     self.opts.SRout, self.opts.dropout, self.nb_timepoints).to(self.device)
        self.Student_SR = Student_SR(self.opts.hidden2, self.opts.hidden1, self.opts.LRout,
                                     self.opts.dropout, self.nb_timepoints).to(self.device)

        # build optimizer for teachers
        param_list = [self.EncoderT.parameters()] + [self.Teacher_LR.parameters()] + [self.Teacher_SR.parameters()]
        self.teacher_optimizer = torch.optim.Adam(itertools.chain(*param_list),
                                                  self.opts.t_lr, [self.opts.beta1, self.opts.beta2])

        # build optimizer for students
        param_list = [self.EncoderS.parameters()] + [self.Student_LR.parameters()] + [self.Student_SR.parameters()]
        self.student_optimizer = torch.optim.Adam(itertools.chain(*param_list),
                                                  self.opts.s_lr, [self.opts.beta1, self.opts.beta2])

    def restore_model(self, resume_iters, model_name="teacher"):
        """
        Restore the trained students and encoder.
        """
        print('Loading the trained models from step {}...'.format(resume_iters))

        if model_name == "teacher":

            EncoderT_path = os.path.join(self.opts.checkpoint_dir, '{}-Encoder_T.ckpt'.format(resume_iters))
            self.EncoderT.load_state_dict(torch.load(EncoderT_path, map_location=lambda storage, loc: storage))

            Teacher_LR_path = os.path.join(self.opts.checkpoint_dir, '{}-Teacher_LR.ckpt'.format(resume_iters))
            self.Teacher_LR.load_state_dict(torch.load(Teacher_LR_path, map_location=lambda storage, loc: storage))

            Teacher_SR_path = os.path.join(self.opts.checkpoint_dir, '{}-Teacher_SR.ckpt'.format(resume_iters))
            self.Teacher_SR.load_state_dict(torch.load(Teacher_SR_path, map_location=lambda storage, loc: storage))
        else:

            EncoderS_path = os.path.join(self.opts.checkpoint_dir, '{}-Encoder_S.ckpt'.format(resume_iters))
            self.EncoderS.load_state_dict(torch.load(EncoderS_path, map_location=lambda storage, loc: storage))

            Student_LR_path = os.path.join(self.opts.checkpoint_dir, '{}-Student_LR.ckpt'.format(resume_iters))
            self.Student_LR.load_state_dict(torch.load(Student_LR_path, map_location=lambda storage, loc: storage))

            Student_SR_path = os.path.join(self.opts.checkpoint_dir, '{}-Student_SR.ckpt'.format(resume_iters))
            self.Student_SR.load_state_dict(torch.load(Student_SR_path, map_location=lambda storage, loc: storage))

    def reset_grad(self):
        """
        Reset the gradient buffers.
        """
        self.student_optimizer.zero_grad()
        self.teacher_optimizer.zero_grad()

    def loss_GmTE_Net(self, real, predicted, metric):
        """
        Compute topological losses.
        """
        self.MAE = torch.nn.L1Loss()
        if metric == 'global_topology':
            return self.MAE(real, predicted)
        elif metric == 'local_topology':
            if real.shape[1] == 595:
                size=35
            else:
                size=116
            real_topology = topological_measures(real,  size)
            fake_topology = topological_measures(predicted, size)
            # 0:CC    1:EC   2:PC
            return torch.tensor(mean_absolute_error(fake_topology[0], real_topology[0]), requires_grad=True)
        else:
            assert False, '[*] loss not implemented.'

    def train(self):
        """
        Test both Teacher and Student networks of our GmTE-Net
        """
        t0_iter_T = iter(self.F_4_t0_T)
        t0_iter_S = iter(self.A_100_t0_S)

        tn_morph_iters_T = []
        for loader in self.teacher_M_tn_loaders:
            tn_morph_iters_T.append(iter(loader))

        tn_func_iters_T = []
        for loader in self.teacher_F_tn_loaders:
            tn_func_iters_T.append(iter(loader))

        tn_morph_iters_S = []
        for loader in self.student_M_tn_loaders:
            tn_morph_iters_S.append(iter(loader))

        tn_func_iters_S = []
        for loader in self.student_F_tn_loaders:
            tn_func_iters_S.append(iter(loader))

        # Start training.
        start_time = time.time()
        start_iters = 0
        print(" 1. Train the Teacher for LR and SR")
        for i in range(start_iters, self.opts.num_iters):
            print("-------------iteration-{}-------------".format(i))
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            #---ENCODER---TEACHER---#
            # Prepare the input to the encoder of the Teacher
            # It is a matrix of 4 real complete subjects with low-resolution graphs at t0
            try:
                t0_morph_encoder_T = next(t0_iter_T)
            except:
                t0_iter_T = iter(self.F_4_t0_T)
                t0_morph_encoder_T = next(t0_iter_T)

            t0_M_encoder_T = t0_morph_encoder_T[0].to(self.device)

            #---ENCODER---STUDENT---#
            # Prepare the input to the encoder of the Student
            # It is a matrix of 100 augmented subjects with low-resolution graphs at t0
            try:
                t0_morph_encoder_S = next(t0_iter_S)
            except:
                t0_iter_S = iter(self.A_100_t0_S)
                t0_morph_encoder_S = next(t0_iter_S)
            t0_M_encoder_S = t0_morph_encoder_S[0].to(self.device)

            #---Graph trajectory decoder-1----TEACHER---#
            # Prepare the real data to compute the loss
            # It is a matrix of 4 real complete subjects with trajectory of low-resolution graphs (t0 ... tn)
            M_tgt_GT_T = []
            for tn_morph_idx in range(len(tn_morph_iters_T)):
                try:
                    M_tgt_GT_i = next(tn_morph_iters_T[tn_morph_idx])
                    M_tgt_GT_T.append(M_tgt_GT_i)
                except:
                    tn_morph_iters_T[tn_morph_idx] = iter(self.teacher_M_tn_loaders[tn_morph_idx])
                    M_tgt_GT_i = next(tn_morph_iters_T[tn_morph_idx])
                    M_tgt_GT_T.append(M_tgt_GT_i)
            for tn_morph_idx in range(len(M_tgt_GT_T)):
                M_tgt_GT_T[tn_morph_idx] = M_tgt_GT_T[tn_morph_idx][0].to(self.device)

            #---Graph trajectory decoder-2----TEACHER---#
            # Prepare the real data to compute the loss
            # It is a matrix of 4 real complete subjects with trajectory of super-resolution graphs (t0 ... tn)
            F_tgt_GT_T = []
            for tn_func_idx in range(len(tn_func_iters_T)):
                try:
                    F_tgt_GT_i = next(tn_func_iters_T[tn_func_idx])
                    F_tgt_GT_T.append(F_tgt_GT_i)
                except:
                    tn_func_iters_T[tn_func_idx] = iter(self.teacher_F_tn_loaders[tn_func_idx])
                    F_tgt_GT_i = next(tn_func_iters_T[tn_func_idx])
                    F_tgt_GT_T.append(F_tgt_GT_i)

            for tn_func_idx in range(len(F_tgt_GT_T)):
                F_tgt_GT_T[tn_func_idx] = F_tgt_GT_T[tn_func_idx][0].to(self.device)

            #---Graph trajectory decoder-1---STUDENT---#
            # Prepare the ground truth 100 augmented data to compute the loss
            # It is a matrix of the ground truth 100 augmented data with trajectory of low-resolution graphs (t0 ... tn)
            M_tgt_GT_S = []
            for tn_morph_idx in range(len(tn_morph_iters_S)):
                try:
                    M_tgt_GT_i = next(tn_morph_iters_S[tn_morph_idx])
                    M_tgt_GT_S.append(M_tgt_GT_i)
                except:
                    tn_morph_iters_S[tn_morph_idx] = iter(self.student_M_tn_loaders[tn_morph_idx])
                    M_tgt_GT_i = next(tn_morph_iters_S[tn_morph_idx])
                    M_tgt_GT_S.append(M_tgt_GT_i)
            for tn_morph_idx in range(len(M_tgt_GT_S)):
                M_tgt_GT_S[tn_morph_idx] = M_tgt_GT_S[tn_morph_idx][0].to(self.device)

            #---Graph trajectory decoder-2---STUDENT---#
            # Prepare the ground truth 100 augmented data to compute the loss
            # It is a matrix of the ground truth 100 augmented data with trajectory of super-resolution graphs (t0 ... tn)
            F_tgt_GT_S = []
            for tn_func_idx in range(len(tn_func_iters_S)):
                try:
                    F_tgt_GT_i = next(tn_func_iters_S[tn_func_idx])
                    F_tgt_GT_S.append(F_tgt_GT_i)
                except:
                    tn_func_iters_S[tn_func_idx] = iter(self.student_F_tn_loaders[tn_func_idx])
                    F_tgt_GT_i = next(tn_func_iters_S[tn_func_idx])
                    F_tgt_GT_S.append(F_tgt_GT_i)

            for tn_func_idx in range(len(F_tgt_GT_S)):
                F_tgt_GT_S[tn_func_idx] = F_tgt_GT_S[tn_func_idx][0].to(self.device)

            # =================================================================================== #
            #          2. Train the Teacher for multi-trajectory evolution prediction             #
            # =================================================================================== #
            teacher_LR_loss = 0
            adj = torch.eye(t0_M_encoder_T.shape[0]).to(self.device)
            embedding = self.EncoderT(t0_M_encoder_T, adj).detach()
            M_fake_i = self.Teacher_LR(embedding, adj)

            teacher_SR_loss = 0
            adj = torch.eye(t0_M_encoder_T.shape[0]).to(self.device)
            F_fake_i = self.Teacher_SR(embedding, adj)

            for timepoint in range(0, self.nb_timepoints - 1):
                ### teacher loss
                teacher_loss_ti_LR = self.loss_GmTE_Net(M_tgt_GT_T[timepoint], M_fake_i[timepoint], "global_topology")
                teacher_loss_ti_SR = self.loss_GmTE_Net(F_tgt_GT_T[timepoint], F_fake_i[timepoint], "global_topology")
                teacher_LR_loss += (teacher_loss_ti_LR)
                teacher_SR_loss += (teacher_loss_ti_SR)

            teacher_LR_loss = torch.mean(teacher_LR_loss)
            teacher_SR_loss = torch.mean(teacher_SR_loss)

            teachers_loss = (teacher_LR_loss + teacher_SR_loss) / 2
            self.reset_grad()
            teachers_loss.backward()
            self.teacher_optimizer.step()

            # Logging.
            loss = {}
            loss['Teacher/loss'] = teachers_loss.item()

            # =================================================================================== #
            #                                   3. Miscellaneous                                  #
            # =================================================================================== #
            # print out training information.
            if (i + 1) % self.opts.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, self.opts.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

            # save model checkpoints.
            if (i + 1) % self.opts.model_save_step == 0:
                EncoderT_path = os.path.join(self.opts.checkpoint_dir, '{}-Encoder_T.ckpt'.format(i + 1))
                torch.save(self.EncoderT.state_dict(), EncoderT_path)

                Teacher_LR_path = os.path.join(self.opts.checkpoint_dir, '{}-Teacher_LR.ckpt'.format(i + 1))
                torch.save(self.Teacher_LR.state_dict(), Teacher_LR_path)

                Teacher_SR_path = os.path.join(self.opts.checkpoint_dir, '{}-Teacher_SR.ckpt'.format(i + 1))
                torch.save(self.Teacher_SR.state_dict(), Teacher_SR_path)

                print('Saved model checkpoints into {}...'.format(self.opts.checkpoint_dir))

                print('============================')
                print("End of Training the Teacher")
                print('============================')

            if i == (self.opts.num_iters - 1):
                # ============================================================================================= #
                #       4. Restore the trained Teacher network from the last iteration to train the Student     #
                # ============================================================================================= #

                self.restore_model(self.opts.test_iters, model_name="teacher")
                self.EncoderT.eval()
                self.Teacher_LR.eval()
                self.Teacher_SR.eval()

                with torch.no_grad():
                    adj = torch.eye(t0_M_encoder_S.shape[0]).to(self.device)
                    embedding_T = self.EncoderT(t0_M_encoder_S, adj).detach()
                    predicted_Trajectory_LR = []
                    predicted_Trajectory_SR = []
                    M_fake_i = self.Teacher_LR(embedding_T, adj)
                    F_fake_i = self.Teacher_SR(embedding_T, adj)
                    for timepoint in range(0, self.nb_timepoints - 1):
                        # the below list is the ground truth to the student
                        predicted_Trajectory_LR.append(M_fake_i[timepoint])
                        predicted_Trajectory_SR.append(F_fake_i[timepoint])

                print(" 2. Train the Student LR and SR ")
                for j in range(start_iters, self.opts.num_iters):
                    print("-------------iteration{}-------------".format(j))
                    # =================================================================================== #
                    #            5. Train the Student for multi-trajectory evolution prediction           #
                    # =================================================================================== #
                    student_LR_loss = 0
                    adj = torch.eye(t0_M_encoder_S.shape[0]).to(self.device)
                    embedding_S = self.EncoderS(t0_M_encoder_S, adj).detach()
                    M_fake_i_s = self.Student_LR(embedding_S, adj)

                    student_SR_loss = 0
                    adj = torch.eye(t0_M_encoder_S.shape[0]).to(self.device)
                    F_fake_i_s = self.Student_SR(embedding_S, adj)

                    for timepoint in range(0, self.nb_timepoints - 1):
                        student_loss_ti_LR = self.loss_GmTE_Net(predicted_Trajectory_LR[timepoint], M_fake_i_s[timepoint],
                                                           "local_topology")
                        student_loss_ti_SR = self.loss_GmTE_Net(predicted_Trajectory_SR[timepoint], F_fake_i_s[timepoint],
                                                           "local_topology")
                        student_LR_loss += (student_loss_ti_LR)
                        student_SR_loss += (student_loss_ti_SR)

                    student_LR_loss = torch.mean(student_LR_loss)
                    student_SR_loss = torch.mean(student_SR_loss)

                    students_loss = (student_LR_loss + student_SR_loss) / 2 + self.loss_GmTE_Net(embedding_S, embedding_T, self.loss).to(self.device)
                    self.reset_grad()
                    students_loss.backward()
                    self.student_optimizer.step()
                    # Logging.
                    loss = {}
                    loss['Student/loss'] = students_loss.item()

                    # save model checkpoints.
                    if (j + 1) % self.opts.model_save_step == 0:
                        EncoderS_path = os.path.join(self.opts.checkpoint_dir, '{}-Encoder_S.ckpt'.format(j + 1))
                        torch.save(self.EncoderS.state_dict(), EncoderS_path)

                        Student_LR_path = os.path.join(self.opts.checkpoint_dir, '{}-Student_LR.ckpt'.format(j + 1))
                        torch.save(self.Student_LR.state_dict(), Student_LR_path)

                        Student_SR_path = os.path.join(self.opts.checkpoint_dir, '{}-Student_SR.ckpt'.format(j + 1))
                        torch.save(self.Student_SR.state_dict(), Student_SR_path)

                        print('Saved model checkpoints into {}...'.format(self.opts.checkpoint_dir))

                        print('===========================')
                        print("End of Training the Student")
                        print('===========================')

    # =================================================================================== #
    #                              6. Test with a new dataset                             #
    # =================================================================================== #
    def test(self):
        """
        Test both trained Teacher and Student networks of our GmTE-Net
        """
        self.restore_model(self.opts.test_iters, model_name="teacher")
        self.EncoderT.eval()
        self.Teacher_LR.eval()
        self.Teacher_SR.eval()

        self.restore_model(self.opts.test_iters, model_name="student")
        self.EncoderS.eval()
        self.Student_LR.eval()
        self.Student_SR.eval()

        t0_M_encoder = next(iter(self.F_4_t0_T))
        t0_M_encoder = t0_M_encoder[0].to(self.device)

        tn_morph_iters = []
        for loader in self.all_times_loaders_morphological:
            tn_morph_iters.append(iter(loader))
        M_tgt_GT = []
        for tn_morph_idx in range(len(tn_morph_iters)):
            M_tgt_GT_i = next(tn_morph_iters[tn_morph_idx])
            M_tgt_GT.append(M_tgt_GT_i)
        for tn_morph_idx in range(len(M_tgt_GT)):
            M_tgt_GT[tn_morph_idx] = M_tgt_GT[tn_morph_idx][0].to(self.device)

        tn_func_iters = []
        for loader in self.all_times_loaders_functional:
            tn_func_iters.append(iter(loader))
        F_tgt_GT = []
        for tn_func_idx in range(len(tn_func_iters)):
            F_tgt_GT_i = next(tn_func_iters[tn_func_idx])
            F_tgt_GT.append(F_tgt_GT_i)
        for tn_func_idx in range(len(F_tgt_GT)):
            F_tgt_GT[tn_func_idx] = F_tgt_GT[tn_func_idx][0].to(self.device)

        with torch.no_grad():
            adj = torch.eye(t0_M_encoder.shape[0]).to(self.device)
            embedding_T = self.EncoderT(t0_M_encoder, adj).detach()
            embedding = self.EncoderS(t0_M_encoder, adj).detach()
            predicted_Trajectory_LR_from_student = []
            predicted_Trajectory_SR_from_student = []
            predicted_Trajectory_LR_from_teacher = []
            predicted_Trajectory_SR_from_teacher = []
            M_fake_i_T = self.Teacher_LR(embedding_T, adj)
            F_fake_i_T = self.Teacher_SR(embedding_T, adj)
            M_fake_i = self.Student_LR(embedding, adj)
            F_fake_i = self.Student_SR(embedding, adj)
            for timepoint in range(0, self.nb_timepoints - 1):
                # the below list is the ground truth to the student
                predicted_Trajectory_LR_from_teacher.append(M_fake_i_T[timepoint])
                predicted_Trajectory_SR_from_teacher.append(F_fake_i_T[timepoint])
                predicted_Trajectory_LR_from_student.append(M_fake_i[timepoint])
                predicted_Trajectory_SR_from_student.append(F_fake_i[timepoint])

        return predicted_Trajectory_LR_from_teacher, predicted_Trajectory_SR_from_teacher, predicted_Trajectory_LR_from_student, predicted_Trajectory_SR_from_student

