import torch
import torch.nn as nn

class BasicBlock(nn.Module):    #定义一个BasicBlock类，继承自nn.model，适用于resnet18、34的残差结构
    expansion = 1   #指定扩张因子为1，主分支的卷积核不发生变化
    #初始化函数，定义网络层和一些参数
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock).__init__()
        #传入输入通道数，输出通道数，卷积核大小默认为3，步长默认为1，下采样默认为None
        self.cov1=nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU()
        self.cov1=nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False)