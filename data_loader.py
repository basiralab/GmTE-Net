import torch.utils.data as data_utils
import numpy as np
import torch
import os
import pandas as pd



# read csv file
def readcsv(filename):
    data = pd.read_csv(filename)
    c = []
    data = np.array(data)
    for i in range(0,data.shape[0]):
        a = data[i][0]
        b = np.array(list(a.split(" ")))
        c.append(b)

    return(np.array(c))

def get_loader(features, batch_size, train_test, num_workers=1):
    """
    Build and return a data loader.
    """
    if train_test == "train":
        dataset = data_utils.TensorDataset(torch.Tensor(features))
        loader = data_utils.DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle = True, #set to True in case of training and False when testing the model
                            num_workers=num_workers
                            )
    else:
        dataset = data_utils.TensorDataset(torch.Tensor(features))
        loader = data_utils.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=False,  # set to True in case of training and False when testing the model
                                       num_workers=num_workers
                                       )
    
    return loader



def create_dirs_if_not_exist(dir_list):
    if isinstance(dir_list, list):
        for dir in dir_list:
            if not os.path.exists(dir):
                os.makedirs(dir)
    else:
        if not os.path.exists(dir_list):
            os.makedirs(dir_list)