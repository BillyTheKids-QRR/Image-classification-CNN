import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
from skimage import io
import time
import torch
from torch import optim
from torch.utils.data import dataset
from skimage import transform
from torchvision import transforms, utils,datasets
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import math
import shutil
from collections import Counter
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from Ghost import IGhosts
from GhostNet import GhostNet
from CGhost import CGhosts
from GoogleLeNet import GoogLeNet
from Ghost import IGhosts
from AlexNet import AlexNet
from ResNet34 import ResNet
from MobileNetV3 import MobileNetV3_Large , MobileNetV3_Small
from Vgg19 import VGG
from CGhost_modify import CGhosts_modify
from LeNet5 import LeNet5
#过滤警告信息
import warnings
warnings.filterwarnings("ignore")


start = time.perf_counter()

data_transform = transforms.Compose([
    transforms.Resize(224),
    # transforms.RandomHorizontalFlip(),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.17], std=[0.05])
])

    # 读取数据，随机分为训练集和验证集
train_dataset = datasets.ImageFolder('D:/CWRU7200/CWRU32+7200/1hp', transform=data_transform)
test_dataset = datasets.ImageFolder('D:/CWRU7200/CWRU32+7200+03/1hp', transform=data_transform)

train_size = int(len(train_dataset))
val_size = int(len(test_dataset))
print('训练样本数量：', train_size, '验证集样本数量：', val_size)
    # train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)


train_data_iter =iter(train_loader)
train_image,train_label = train_data_iter.next()
val_data_iter = iter(val_loader)
val_image, val_label = val_data_iter.next()

model = GhostNet().cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0015)

epoch = 30
train_set_loss = []
val_set_loss = []
train_set_acc = []
val_set_acc = []
for ep in range(epoch):  # loop over the dataset multiple times

    running_loss = 0
    trainset_right = 0
    for step, (x, y) in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
        inputs = Variable(x).cuda()
        labels = Variable(y).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs1 = model(inputs)

        predict_y1 = torch.max(outputs1, dim=1)[1]
        trainset_right += torch.eq(predict_y1, labels).sum().item()

        train_loss = loss_function(outputs1, labels)
        train_loss.backward()
        optimizer.step()

        # validation,这里本来是每n个step输出一次数据，因为通常来讲训练集是成千上万张
        # 但是我们的训练集非常小，我就直接一个step输出一次，这里验证集batchsize=val_size
        # 如果你的验证集数量特别多，建议编写一个evaluate函数用于验证
        running_loss += train_loss.item()
        with torch.no_grad():
            outputs1_5 = model(train_image.cuda()).cpu()
            outputs2 = model(val_image.cuda()).cpu()
            predict_y1_5 = torch.max(outputs1_5, dim=1)[1]
            predict_y2 = torch.max(outputs2, dim=1)[1]
            val_loss = loss_function(outputs2,predict_y2)
            train_acc = torch.eq(predict_y1_5, train_label).sum().item() / train_label.size(0)
            accuracy = torch.eq(predict_y2, val_label).sum().item() / val_label.size(0)
            print('epoch:[%d/%d] step:[%d/%d]  train_loss: %.3f  train_acc: %.3f  val_loss: %.3f val_accuracy: %.3f' %
                  (ep + 1, epoch, step + 1, len(train_loader), train_loss.item(), train_acc,val_loss, accuracy))

    train_set_acc.append(trainset_right/train_size)
    loss_per_epoch = running_loss / len(train_loader)
    train_set_loss.append(loss_per_epoch)

    val_set_acc.append(accuracy)
    val_set_loss.append(val_loss)

    print(train_set_loss)
    print(val_set_loss)
    print(train_set_acc)
    print(val_set_acc)

    list = [val_set_acc]

data = pd.DataFrame([list])
data.to_csv('.\\experiment\\experiment_result.csv', mode='a', header=False, index=False)

print('Finished Training')
# 可视化
# 你也可以用vison,tensorbord等可视化工具实时获取损失图像


end = time.perf_counter()
runTime = end - start
runTime_ms = runTime * 1000
print("运行时间：", runTime, "秒")
print("运行时间：", runTime_ms, "毫秒")


epo = np.arange(0, len(train_set_acc), 1)
plt.subplot(2,1,1)
plt.plot(epo, train_set_loss,color='red',label='train set')
plt.plot(epo, val_set_loss, color='blue',label='validation set')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('loss function of train set and validation set')
plt.subplots_adjust(hspace=0.5)
plt.subplot(2,1,2)
plt.plot(epo, train_set_acc,color='red',label='train set')
plt.plot(epo, val_set_acc, color='blue',label='validation set')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('accuracy function of train set and validation set')
plt.show()




