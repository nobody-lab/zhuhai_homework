from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from efficientdet import EfficientDet
import torch
import torch.nn as nn
import numpy as np
from utils import getdir,HorseDataset,get_biou,get_miou

#超参数
epochs=60
batchsize=8
lr=1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#优先使用GPU训练、
torch.backends.cudnn.deterministic = True #防止使用GPU使由于随机性导致同输入不同输出
#路径
train_horse_path = r'weizmann_horse_db\train_horse' #训练集图像
train_mask_path = r'weizmann_horse_db\train_mask' #训练集mask
val_horse_path = r'weizmann_horse_db\val_horse' #测试集图像
val_mask_path = r'weizmann_horse_db\val_mask' #测试集mask

#将所有训练集图像与mask存至列表
train_horse_dir,train_mask_dir=getdir(train_horse_path,train_mask_path)

TrainHorse_Dataset = HorseDataset(train_horse_dir,train_mask_dir)
TrainHorse_Dataloader = DataLoader(TrainHorse_Dataset,batch_size=batchsize,shuffle=True)
#将所有测试集图像与mask存至列表
val_horse_dir,val_mask_dir=getdir(val_horse_path,val_mask_path)

ValHorse_Dataset = HorseDataset(val_horse_dir,val_mask_dir)
ValHorse_Dataloader = DataLoader(ValHorse_Dataset,batch_size=1,shuffle=True)

model=EfficientDet()  #使用EfficientDet B0训练
Loss = nn.CrossEntropyLoss() #交叉熵损失
model=model.to(device) 
Loss=Loss.to(device)

#保存数据用于画图分析
train_loss_list = []#训练集损失
test_loss_list = []#测试集损失
train_miou_list=[]#训练集miou
test_miou_list=[]#测试集miou
train_biou_list=[]#训练集biou
test_biou_list=[]#测试集biou
epoch_list=[] #保存训练进行的epoch数
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
#开始训练
for epoch in range(0,epochs):
    print('Epoch:',epoch+1,'training')
    iter=0
    iter1=0
    train_mious=0
    test_mious = 0
    train_bious=0
    test_bious=0
    trainloss=0
    testloss=0
    for x,y in TrainHorse_Dataloader:
        iter1=iter1+1
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        output = model(x)
        y_=y
        y=y.long()
        loss = Loss(output,y)
        trainloss+=loss.item()
        loss.backward()
        optimizer.step()
        iter_temp=iter+batchsize
        if iter_temp>278:
            iter_temp=278
        print(iter_temp,'/278') #训练进度
        for i in range(0,x.shape[0]):
            iter = iter + 1
            img = x[i]
            pre = output[i][1]
            pre = pre.cpu().detach().numpy()
            pre = np.around(pre)
            y=y_
            y = y[i].cpu()
            y = y.numpy()
            y = y.astype(int)
            miou = get_miou(pre,y)
            biou = get_biou(pre, y)
            train_mious = train_mious + miou
            train_bious=train_bious+biou
    trainloss=trainloss/iter1
    train_mious=train_mious/iter_temp
    train_bious=train_bious/iter_temp
    iter=0


    print('Epoch:',epoch+1,'testing')
    for x,y in ValHorse_Dataloader:
        x=x.to(device)
        y=y.to(device)
        iter=iter+1
        output = model(x)
        y_=y
        y=y.long()
        loss = Loss(output,y)
        testloss+=loss.item()
        print(iter,'/49')#测试进度
        img = x[0]
        pre = output[0][1]
        pre = pre.cpu().detach().numpy()
        pre = np.around(pre)
        y=y_
        y = y[0].cpu()
        y=y.numpy()
        y = y.astype(int)
        miou = get_miou(pre,y)
        biou=get_biou(pre,y)
        test_bious=test_bious+biou
        test_mious = test_mious + miou
    testloss=testloss/iter
    test_mious=test_mious/iter
    test_bious=test_bious/iter
    print('trainloss:',trainloss,'testloss',testloss)
    print('trainmiou:',train_mious,'testmiou',test_mious)
    print('trainbiou:',train_bious, 'testbiou', test_bious)
    #保存loss miou biou
    train_loss_list.append(trainloss)
    test_loss_list.append(testloss)
    train_miou_list.append(train_mious)
    test_miou_list.append(test_mious)
    train_biou_list.append(train_bious)
    test_biou_list.append(test_bious)
    epoch_list.append(epoch + 1)


#保存模型参数
torch.save(model.state_dict(), 'model_%s.pth'%epochs)
print('训练结束,模型model_%s.pth已保存'%epochs)

#loss，miou，biou画图
plt.figure()
plt.title('train')
plt.plot(epoch_list,train_loss_list,label='train_loss')
plt.plot(epoch_list,train_miou_list,label='train_miou')
plt.plot(epoch_list,train_biou_list,label='train_biou')
plt.legend(['train_loss','train_miou','train_biou'],loc='upper left')
plt.savefig('train.png')

plt.figure()
plt.title('test')
plt.plot(epoch_list,test_loss_list,label='test_loss')
plt.plot(epoch_list,test_miou_list,label='test_miou')
plt.plot(epoch_list,test_biou_list,label='test_biou')
plt.legend(['test_loss','test_miou','test_biou'],loc='upper left')
plt.savefig('test.png')