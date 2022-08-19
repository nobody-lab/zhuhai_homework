#导入已经训练好的模型输出预测图片
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from efficientdet import EfficientDet
import torch
import cv2
import numpy as np
from utils import HorseDataset,getdir,get_biou,get_miou

val_horse_dir = r'weizmann_horse_db/val_horse'
val_mask_dir = r'weizmann_horse_db/val_mask'
model_dir='model_60.pth'
model = EfficientDet()  # 初始化模型
#导入模型参数
model.load_state_dict(torch.load(model_dir),strict=False)
#导入测试集
horse_dir,mask_dir=getdir(val_horse_dir,val_mask_dir)
ValHorse_Dataset = HorseDataset(horse_dir,mask_dir)
ValHorse_Dataloader = DataLoader(ValHorse_Dataset,batch_size=1,shuffle=False)

iter=0
mious=0
bious=0
print('预测并保存结果中：')
for x, y in ValHorse_Dataloader:
    iter = iter + 1
    output = model(x)
    print(iter, '/49')
    horse = x[0]
    p = output[0][1]
    p=p.detach().numpy()
    p=np.around(p)
    y=y[0].numpy()
    y=y.astype(int)
    miou=get_miou(p,y)
    biou=get_biou(p,y)
    bious=bious+biou
    mious=mious+miou
    #以此保存原图，mask二值化后的图以及预测图
    save_image(horse, 'predict/' + str(iter) + 'horse.png')
    cv2.imwrite('predict/'+str(iter)+'mask.png',y*255)
    cv2.imwrite('predict/'+str(iter)+'pre.png',p*255)
mious=mious/iter
bious=bious/iter
print('miou:',mious,'biou:',bious)