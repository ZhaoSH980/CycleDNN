import sys
sys.path.append("..")
from PIL import Image
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import torch
import torch.nn as nn
from torch.utils.data import  Dataset, DataLoader
import pandas as pd


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=10, out_features=500)
        self.fc2 = torch.nn.Linear(in_features=500, out_features=500)
        self.fc3 = torch.nn.Linear(in_features=500, out_features=500)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        x = x.unsqueeze(1)
        input = self.fc1(x)
        input = self.relu(input)
        input = self.dropout(input)
        input = self.fc2(input)
        input = self.relu(input)
        input = self.fc3(input)
        output = self.relu(input)
        return output

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc4 = torch.nn.Linear(in_features=500, out_features=500)
        self.fc5 = torch.nn.Linear(in_features=500, out_features=500)
        self.fc6 = torch.nn.Linear(in_features=500, out_features=10)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        x = x.unsqueeze(1)
        input = self.fc4(x)
        input = self.relu(input)
        input = self.fc5(input)
        input = self.relu(input)
        input = self.dropout(input)
        output = self.fc6(input)
        return output

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=10, out_features=500)
        self.fc2 = torch.nn.Linear(in_features=500, out_features=500)
        self.fc3 = torch.nn.Linear(in_features=500, out_features=500)
        self.fc4 = torch.nn.Linear(in_features=500, out_features=500)
        self.fc5 = torch.nn.Linear(in_features=500, out_features=500)
        self.fc6 = torch.nn.Linear(in_features=500, out_features=10)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        input = self.fc1(x)
        input = self.relu(input)
        input = self.dropout(input)
        input = self.fc2(input)
        input = self.relu(input)
        input = self.fc3(input)
        input = self.relu(input)
        input = self.fc4(input)
        input = self.relu(input)
        input = self.fc5(input)
        input = self.relu(input)
        input = self.dropout(input)
        output = self.fc6(input)
        return output