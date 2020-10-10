﻿# 利用Unet网络实现钢腐蚀区域的图像分割（PYTORCH）

------

以下我将为您说明：

> * 训练平台
> * 数据集简介
> * 代码简介
> * 测试结果

------

## 训练平台

> * Python 3.6
> * Pytorch 3.5.1
> * 操作系统：windows10家庭中文版； 版本号：1909； 操作系统版本：18363.1082。

## 数据集简介
数据集存放在data的子文件夹下，train文件夹下存放的是训练数据，共200张；val文件夹存放的是验证数据共12张，text文件下存放的是测试图片。

## 代码简介
> * dataset.py是数据集代码。
> * unet.py存放的是网络结构代码。
> * main.py里面是训练和测试代码，需要在终端中运行。
> * predict.py是预测代码。

## 测试结果
1、改动前：使用unet网络进行训练：训练时，将图片的分辨率降到（224，224）大小在输入到网络中，在测试时会发现测试效果不好，测试时如果提高图片的输入分辨率，会使预测结果时间变慢，cpu、内存负荷增加，但输出的结果更清晰。（详见result.pdf）
2、改动后：使用unet网络进行训练：训练时，将图片的分辨率先降到（598，448），在以中心裁切到（448，448）大小，极大的减轻了预处理时高度方向上的失真，并且较原来，图片的分辨率增加（输入网络中的参数增加）。（详见result_update.pdf）

