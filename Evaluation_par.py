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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def MAPE(prediction,truth):
    MAPE_results = (torch.mean(torch.abs((prediction-truth)/(truth)), dim=1)).float().sum().item()
    return MAPE_results

def MSE(prediction,truth):
    MSE_results = (torch.mean(((prediction-truth)**2), dim=1)).float().sum().item()
    return MSE_results

def RMSE(prediction,truth):
    RMSE_results = (torch.sqrt(torch.mean(((prediction-truth)**2), dim=1))).float().sum().item()
    return RMSE_results

def MAE(prediction,truth):
    MAE_results = (torch.mean(torch.abs(prediction-truth))).float().sum().item()
    return MAE_results

    



def net_accurary(data_iter, loss_function, net):
    net.eval()
    MAPE_test_sum, loss= 0.0, 0.0
    RMSE_test_sum = 0.0
    MSE_test_sum = 0.0
    MAE_test_sum = 0.0
    n1 = 0
    for X, y in data_iter:
        if torch.cuda.is_available():
            X = X.to(device)
            y = y.to(device)

        y_hat = net(X)
        loss += loss_function(y_hat, y).item()
        loss = loss
        y_hat = y_hat.squeeze(1)
        MAPE_test_sum += MAPE(y_hat,y)
        MSE_test_sum += MSE(y_hat,y)
        RMSE_test_sum += RMSE(y_hat,y)
        MAE_test_sum += MAE(y_hat,y)
        n1 += y.shape[0]
    return MAPE_test_sum/n1,MSE_test_sum/n1 ,RMSE_test_sum/n1,MAE_test_sum/n1, loss / n1

def net_accurary2(data_iter, loss_function, netA,netB,k1,k2):
    netA.eval()
    netB.eval()
    MAPE_test_sum, loss= 0.0, 0.0
    RMSE_test_sum = 0.0
    MSE_test_sum = 0.0
    MAE_test_sum = 0.0
    n1 = 0
    for X, Y in data_iter:
        if torch.cuda.is_available():
            X = X.to(device)
            Y = Y.to(device)

        x = X[:,k1,:]
        y = Y[:,k2,:]

        output1 = netA(x)
        output1 = output1.squeeze(1)
        y_hat = netB(output1)
        y_hat = y_hat.squeeze(1)
        loss += loss_function(y_hat, y).item()
        loss = loss
        y_hat = y_hat.squeeze(1)
        MAPE_test_sum += MAPE(y_hat,y)
        MSE_test_sum += MSE(y_hat,y)
        RMSE_test_sum += RMSE(y_hat,y)
        MAE_test_sum += MAE(y_hat,y)
        n1 += y.shape[0]
    return MAPE_test_sum/n1,MSE_test_sum/n1 ,RMSE_test_sum/n1,MAE_test_sum/n1, loss / n1


def net_accurary3(data_iter, loss_function, netA,netB, trans):
    netA.eval()
    netB.eval()
    MAPE_test_sum, loss= 0.0, 0.0
    RMSE_test_sum = 0.0
    MSE_test_sum = 0.0
    MAE_test_sum = 0.0
    n1 = 0
    for Xo, y in data_iter:
        X = torch.tensor(trans.transform(Xo)).float()
        if torch.cuda.is_available():
            X = X.to(device)
            y = y.to(device)

        output1 = netA(X)
        output1 = output1.squeeze(1)
        y_hat = netB(output1)
        y_hat = y_hat.squeeze(1)

        y_hat_t = y_hat.cpu().detach().numpy()
        y_hat_t = trans.inverse_transform(y_hat_t)
        y_hat_t = torch.tensor(y_hat_t).float().cuda()

        loss += loss_function(y_hat_t, y).item()
        loss = loss
        
        MAPE_test_sum += MAPE(y_hat_t,y)
        MSE_test_sum += MSE(y_hat_t,y)
        RMSE_test_sum += RMSE(y_hat_t,y)
        MAE_test_sum += MAE(y_hat_t,y)
        n1 += y.shape[0]
    return MAPE_test_sum/n1,MSE_test_sum/n1 ,RMSE_test_sum/n1,MAE_test_sum/n1, loss / n1


def ex_accurary(data_iter, loss_function,k1,k2):
    # net.eval()
    MAPE_test_sum, loss= 0.0, 0.0
    RMSE_test_sum = 0.0
    MSE_test_sum = 0.0
    MAE_test_sum = 0.0
    n1 = 0
    for X, Y in data_iter:
        if torch.cuda.is_available():
            X = X.to(device)
            Y = Y.to(device)

        x = X[:,k1,:]
        y = Y[:,k2,:]

        MAPE_test_sum += MAPE(x,y)
        MSE_test_sum += MSE(x,y)
        RMSE_test_sum += RMSE(x,y)
        MAE_test_sum += MAE(x,y)
        n1 += y.shape[0]
    return MAPE_test_sum/n1,MSE_test_sum/n1 ,RMSE_test_sum/n1,MAE_test_sum/n1