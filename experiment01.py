import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
#以上语句是由于python与torch版本不匹配才加的与加载数据无关
import torchvision
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn import metrics
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms, utils,datasets
import numpy
from PIL import Image
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
#过滤警告信息
import warnings
warnings.filterwarnings("ignore")
import sklearn
from sklearn.model_selection import train_test_split
from Ghost import IGhosts
from GoogleLeNet import GoogLeNet
from GhostNet import GhostNet
from CGhost import CGhosts
from Ghost import IGhosts
from inceptionV3 import Inception3
from AlexNet import AlexNet
from ResNet34 import ResNet
from MobileNetV3 import MobileNetV3_Large , MobileNetV3_Small
from Vgg import vgg_19
from Vgg19 import VGG
from CGhost_modify import CGhosts_modify
from LeNet5 import LeNet5
from drawMatrix import DrawConfusionMatrix
from metrics import Metric
import itertools



# def plot_matrix(y_true, y_pred, labels_name, title=None, thresh=0.8, axis_labels=None):
# # 利用sklearn中的函数生成混淆矩阵并归一化
#     cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # 生成混淆矩阵
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
#
# # 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
#     plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
#     plt.colorbar()  # 绘制图例
#
# # 图像标题
#     if title is not None:
#         plt.title(title)
# # 绘制坐标
#     num_local = np.array(range(len(labels_name)))
#     if axis_labels is None:
#         axis_labels = labels_name
#     plt.xticks(num_local, axis_labels, rotation=45)  # 将标签印在x轴坐标上， 并倾斜45度
#     plt.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#
# # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
#     for i in range(np.shape(cm)[0]):
#         for j in range(np.shape(cm)[1]):
#             if int(cm[i][j] * 100 + 0.5) > 0:
#                 plt.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
#                         ha="center", va="center",
#                         color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
# # 显示
#     plt.show()


df = pd.DataFrame(columns=['val_set_loss','val_set_acc'])
df.to_csv(".\\experiment\\experiment_result.csv",index=False)

start = time.perf_counter()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize(256), # 缩放图片(Image)，保持长宽比不变，最短边为32像素
    #transforms.CenterCrop(32), # 从图片中间切出32*32的图片
    transforms.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
    transforms.Normalize(mean=[0.2580339, 0.16180208, 0.42759213], std=[0.04127504, 0.16801909, 0.07414886]) # 标准化至[-1, 1]，规定均值和标准差

])
# transforms.Normalize(mean=[0.492, 0.461, 0.417], std=[0.256, 0.248, 0.251]
data= datasets.ImageFolder(root="D:/ImbalanceData/Graf128/1_100",
           transform=data_transform) #导入数据集

train_size = int(0.8 * len(data))
print(train_size)
val_size = len(data) - train_size
print('训练样本数量：', train_size, '验证集样本数量：', val_size)


img, label = data[887] #将启动魔法方法__getitem__(0)
"""这个15000，表示所有文件夹排序后的第15001张图片，0是第一张图片"""
print(label)   #查看标签
"""这里的0表示cat，1表示dog；因为是按文件夹排列的顺序，如果有第三个文件夹pig则2表示pig"""
print(img.size())
print(img)

#处理后的图片信息
for img, label in data:
    print("图像img的形状{},标签label的值{}".format(img.shape, label))
    print("图像数据预处理后：\n",img)
    break


train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64 , shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)


train_data_iter =iter(train_loader)
train_image,train_label = train_data_iter.next()
val_data_iter = iter(val_loader)
val_image, val_label = val_data_iter.next()

model = CGhosts_modify().cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



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
        correct = 0
        with torch.no_grad():
            outputs1_5 = model(train_image.cuda()).cpu()
            outputs2 = model(val_image.cuda()).cpu()
            predict_y1_5 = torch.max(outputs1_5,dim=1)[1]
            predict_y2 = torch.max(outputs2, dim=1)[1]
            val_loss = loss_function(outputs2, predict_y2)
            train_acc = torch.eq(predict_y1_5,train_label).sum().item() / train_label.size(0)
            accuracy = torch.eq(predict_y2, val_label).sum().item() / val_label.size(0)
            predict_np = np.argmax(outputs2.detach().numpy(),axis = -1)
            labels_np = val_label.numpy
            # drawconfusionmatrix.update(predict_np,labels_np)

            print('epoch:[%d/%d] step:[%d/%d]  train_loss: %.3f  train_acc: %.3f  val_loss: %.3f val_accuracy: %.3f' %
                  (ep + 1, epoch, step + 1, len(train_loader), train_loss.item(), train_acc,val_loss, accuracy))


    train_set_acc.append(trainset_right / train_size)
    loss_per_epoch = running_loss / len(train_loader)
    train_set_loss.append(loss_per_epoch)

    val_set_acc.append(accuracy)
    val_set_loss.append(val_loss)


    print(train_set_loss)
    print(val_set_loss)
    print(train_set_acc)
    print(val_set_acc)

    list = [val_set_loss,val_set_acc]

data = pd.DataFrame([list])
data.to_csv('.\\experiment\\experiment_result.csv',mode='a',header=False,index=False)

# plot_matrix(val_label,val_set_acc,[0,1,2,3,4,5,6],title='confusion_matrix',
#                 axis_labels=['0','1','2','3','4','5','6'])
print(list)
print('Finished Training')



end = time.perf_counter()
runTime = end - start
runTime_ms = runTime * 1000
print("运行时间：", runTime, "秒")
print("运行时间：", runTime_ms, "毫秒")


# 可视化
# 你也可以用vison,tensorbord等可视化工具实时获取损失图像
epo = numpy.arange(0, len(train_set_acc), 1)
plt.subplot(2, 1, 1)
plt.plot(epo, train_set_loss, color='red', label='train set')
plt.plot(epo, val_set_loss, color='blue', label='validation set')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('loss function of train set and validation set')
plt.subplots_adjust(hspace=0.5)
plt.subplot(2, 1, 2)
plt.plot(epo, train_set_acc, color='red', label='train set')
plt.plot(epo, val_set_acc, color='blue', label='validation set')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('accuracy function of train set and validation set')
plt.show()

# labels_name=['0','1','2','3','4','5','6']
# drawconfusionmatrix = DrawConfusionMatrix(labels_name=labels_name)
# for index,(val_label,val_image) in (val_loader):
#     labels_pd = model(val_image)
#     predict_np = np.argmax(labels_pd.cpu().detach().numpy(), axis=-1)
#     labels_np = val_label.numpy()
#     drawconfusionmatrix.update(labels_np, predict_np)
#
# drawconfusionmatrix.drawMatrix()
#
# confusion_mat=drawconfusionmatrix.getMatrix()
# print(confusion_mat)

# save_path = '.\\experment\\experiment01.pth'
# torch.save(model.state_dict(), save_path)

# checkpoint_save_path = "./experment/Baseline.ckpt"
# if os.path.exists(checkpoint_save_path + '.index'):
#     print('-------------load the model-----------------')
#     model.state_dict(checkpoint_save_path)







