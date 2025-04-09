import itertools
import os
import time
import networkx
import numpy as np
import torch.nn as nn
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from torch.nn import Parameter
import warnings

root_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(root_path)


class DADDataset(Dataset):
    def __init__(self, root_path=root_path, feature='dad', phase='training', toTensor=True, device=torch.device('cuda')):
        self.feature_path = os.path.join(yours)
        self.feature_files = self.get_filelist(self.feature_path)
        self.toTensor=toTensor
        self.device=device
        self.text_feature_path=os.path.join(yours)

        
        
    def get_filelist(self, featurefilepath):
        assert os.path.exists(featurefilepath), "Directory does not exist: %s"%(featurefilepath)
        file_list = []
        for filename in sorted(os.listdir(featurefilepath)):
            file_list.append(filename)
        return file_list
    
    def __len__(self):
        data_len = len(self.feature_files)
        return data_len
    
    def __getitem__(self, index):
        
        data_file = os.path.join(self.feature_path,self.feature_files[index])
        
        try:
            # Load feature data file
            data = np.load(data_file,allow_pickle=True)
            labels = torch.tensor(data['label'])
            labels = labels.long()
            labels = torch.nn.functional.one_hot(labels, num_classes=2)
            ID=data['id']
            
            
        except:
            raise IOError('Load data error! File: %s' % (data_file))
        


        # Load det file
        if labels[1] > 0:
            toa = [90]
            dir = 'positive'
        else:
            toa = [101]
            dir = 'negative'

        IDs=str(ID)+'.npz'

        
        features=data['ffeat']


        text_feature=np.load(self.text_feature_path)['text_features']

       

        if self.toTensor:
            features = np.array(features)
            labels = np.array(labels)
            toa = np.array(toa)
            text_feature = np.array(text_feature)

            # Convert to tensors
            features = torch.tensor(features).to(self.device)
            labels = torch.tensor(labels).to(self.device)
            toa = torch.Tensor(toa).to(self.device)
            text_feature = torch.Tensor(text_feature).to(self.device)


        return features, labels, toa,text_feature
    