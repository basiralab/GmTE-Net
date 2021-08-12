import torch
import numpy as np
import math
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import functional as F

class GCN(Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        x = self.weight
        # print("weight",x.size())
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

        return output

class Encoder(nn.Module):
    """
    Encoder Network
    """
    def __init__(self, nfeat, nhid1, nhid2, dropout):

        super(Encoder, self).__init__()

        self.dropout = dropout
        self.gc1 = GCN(nfeat, nhid1)
        self.gc2 = GCN(nhid1, nhid2)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        return x


class Teacher_SR(nn.Module):
    """
    Teacher Network for Super-Resolution Brain Evolution Trajectory
    """
    def __init__(self, nhid1, nhid2, noutSR, dropout, timepoints):

        super(Teacher_SR, self).__init__()

        self.dropout = dropout
        self.timepoints = timepoints

        self.gc1 = GCN(nhid1, nhid2)
        self.gc2 = GCN(nhid2, noutSR)
        self.gc3 = GCN(noutSR, noutSR)

    def forward(self, x, adj):
        timepoints_prediction = []

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        timepoints_prediction.append(x)

        for t in range(1, self.timepoints):
            x = F.relu(self.gc3(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            timepoints_prediction.append(x)

        return timepoints_prediction

class Teacher_LR(nn.Module):
    """
    Teacher Network for Low-Resolution Brain Evolution Trajectory
    """
    def __init__(self, nhid1, nhid2, nhid3, dropout, timepoints):

        super(Teacher_LR, self).__init__()

        self.dropout = dropout
        self.timepoints = timepoints

        self.gc1 = GCN(nhid1, nhid2)
        self.gc2 = GCN(nhid2, nhid3)
        self.gc3 = GCN(nhid3, nhid3)

    def forward(self, x, adj):

            timepoints_prediction = []
            x = F.relu(self.gc1(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)

            x = F.relu(self.gc2(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            timepoints_prediction.append(x)

            for t in range(1, self.timepoints):
                x = F.relu(self.gc3(x, adj))
                x = F.dropout(x, self.dropout, training=self.training)
                timepoints_prediction.append(x)

            return timepoints_prediction

class Student_SR(nn.Module):
    """
    Student Network for Super-Resolution Brain Evolution Trajectory
    """
    def __init__(self, nhid1, nhid2, noutSR, dropout, timepoints):

        super(Student_SR, self).__init__()

        self.dropout = dropout
        self.timepoints = timepoints

        self.gc1 = GCN(nhid1, nhid2)
        self.gc2 = GCN(nhid2, noutSR)
        self.gc3 = GCN(noutSR, noutSR)

    def forward(self, x, adj):

        timepoints_prediction = []
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        timepoints_prediction.append(x)

        for t in range(1, self.timepoints):
            x = F.relu(self.gc3(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            timepoints_prediction.append(x)

        return timepoints_prediction

class Student_LR(nn.Module):
    """
    Student Network for Low-Resolution Brain Evolution Trajectory
    """
    def __init__(self, nhid1, nhid2, nhid3, dropout, timepoints):

        super(Student_LR, self).__init__()

        self.dropout = dropout
        self.timepoints = timepoints

        self.gc1 = GCN(nhid1, nhid2)
        self.gc2 = GCN(nhid2, nhid3)
        self.gc3 = GCN(nhid3, nhid3)

    def forward(self, x, adj):

            timepoints_prediction = []
            x = F.relu(self.gc1(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)

            x = F.relu(self.gc2(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
            timepoints_prediction.append(x)

            for t in range(1, self.timepoints):
                x = F.relu(self.gc3(x, adj))
                x = F.dropout(x, self.dropout, training=self.training)
                timepoints_prediction.append(x)

            return timepoints_prediction