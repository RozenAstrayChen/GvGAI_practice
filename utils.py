import torch
import torch.nn as nn
import numpy as np
import cv2


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def swapAxis(array):
    return np.moveaxis(array, -1, 0)

def convertVariable(matrix):
    return torch.from_numpy(matrix).float().to(device)

def preProcess(img, size=(120, 80)):
    x = cv2.resize(img, size)
    return x