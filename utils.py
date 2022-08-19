# 功能：导入训练集和测试集,并且计算miou和biou
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import torch
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix

#定义HorseDataset类，并将图像变为224x224的标准形式以输入EfficientDet
class HorseDataset(Dataset):
    def __init__(self,horse_list,mask_list):
        self.horses=horse_list
        self.masks=mask_list

        # horse预处理
        self.transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        # mask的预处理
        self.transformergt=transforms.Compose([
            transforms.Resize((224,224)),
        ])

    def __len__(self):
        return len(self.horses)

    def __getitem__(self, item):#
        horse_path=self.horses[item]
        mask_path=self.masks[item]
        horse = self.transformer(Image.open(horse_path)) #图像预处理
        mask =torch.from_numpy(np.array(self.transformergt(Image.open(mask_path)))).type(torch.FloatTensor) #numpy转tensor，不要使用totensor，否则会改变标签值
        return horse,mask

#获取训练集/测试集的horse和mask图片所在的路径
def getdir(horse_dir,mask_dir):
    horse_list=[]
    mask_list=[]
    for path,_,horses in os.walk(horse_dir):
        for horse in horses:
            horsedir=os.path.join(path,horse)
            horse_list.append(horsedir)  #horse
    for path,_,masks in os.walk(mask_dir):
        for mask in masks:
            maskdir=os.path.join(path,mask)
            mask_list.append(maskdir)   #mask
    return horse_list,mask_list

#计算miou
def get_miou(a,b):
    a = a.reshape((224 * 224))
    b = b.reshape((224 * 224))
    matrix = confusion_matrix(a, b) #混淆矩阵
    TP=matrix[1][1]
    FN=matrix[1][0]
    FP=matrix[0][1]
    TN=matrix[0][0]
    miou = (TP/(TP+FP+FN) + TN/(TN+FN+FP))/2 #0类1类的iou然后求平均
    return miou

#先填充0后腐蚀得到图像边界
def bmap(input):
    padding = cv2.copyMakeBorder(input, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0).astype(float)
    kernel = np.ones((11,11), np.uint8)
    erode = cv2.erode(padding,kernel,6)
    erode_horse = erode[1:input.shape[0] + 1, 1:input.shape[1]+ 1]
    return input-erode_horse

#计算biou

def get_biou(a,b):
    a=bmap(a)
    b=bmap(b)
    a=a.reshape((224*224))
    b=b.reshape((224 * 224))
    matrix=confusion_matrix(a,b)
    TP=matrix[1][1]
    FN=matrix[1][0]
    FP=matrix[0][1]
    TN=matrix[0][0]
    biou = TP/(TP+FP+FN)
    return biou


